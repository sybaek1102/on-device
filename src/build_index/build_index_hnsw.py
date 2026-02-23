#!/usr/bin/env python3
"""
build_index_hnsw.py
──────────────────────────────────────────────────────────────────────
IndexHNSWPQ 기반 인덱스 빌드 스크립트.
build_index.py와 동일한 흐름이지만 Base PQ Index를 IndexHNSWPQ로 대체.

[IndexHNSWPQ 구조 차이]
  - faiss.IndexHNSWPQ(d, pq_M, hnsw_M, pq_nbits)
    ├─ HNSW 그래프: 근사 최근방 탐색
    └─ storage (IndexPQ): PQ 코드 저장
  - PQ 코드/센트로이드 접근: index.storage 사용
  - 복원 벡터(residual 계산용): index.storage.reconstruct_n()
  - IndexHNSWPQ는 reconstruct_n 지원하지 않으므로
    내부 storage.pq.decode() 로 직접 복원

[저장 파일]
  data/index/{CREATION_DATE}_hnswpq.index
  data/index/{CREATION_DATE}_residual_pq_hnsw.index
  data/raw/{CREATION_DATE}_base10M.fvecs
  data/features/{CREATION_DATE}_residual_norm_sq_hnsw.npz
  output/metric/{CREATION_DATE}_build_time_hnsw.json
"""

import faiss
faiss.omp_set_num_threads(8)

import numpy as np
import os
import time
import json
from utils.print_log import info, warn, error

# =========================================================
# 🔹 Configuration
# =========================================================
DATA_DIR   = "/home/syback/vectorDB/ann_datasets/sift1B"
BASE_FILE  = os.path.join(DATA_DIR, "bigann_base.bvecs")
QUERY_FILE = os.path.join(DATA_DIR, "bigann_query.bvecs")
LEARN_FILE = os.path.join(DATA_DIR, "bigann_learn.bvecs")

NUM_BASE  = 10_000_000
NUM_LEARN =  1_000_000

CREATION_DATE = "2026022007"   # 고정

INDEX_SAVE_PATH       = f"/home/syback/vectorDB/on-device/data/index/{CREATION_DATE}_hnswpq.index"
RESIDUAL_INDEX_PATH   = f"/home/syback/vectorDB/on-device/data/index/{CREATION_DATE}_residual_pq_hnsw.index"
BASE10M_SAVE_PATH     = f"/home/syback/vectorDB/on-device/data/raw/{CREATION_DATE}_base10M.fvecs"
RESIDUAL_NORM_SQ_PATH = f"/home/syback/vectorDB/on-device/data/features/{CREATION_DATE}_residual_norm_sq_hnsw.npz"
METRIC_SAVE_PATH      = f"/home/syback/vectorDB/on-device/output/metric/{CREATION_DATE}_build_time_hnsw.json"

DIM      = 128
PQ_M     = 16       # PQ subquantizer 수
PQ_NBITS = 8        # 코드북 bit 수
HNSW_M   = 32       # HNSW 그래프 연결도 (M): 크면 정확도↑, 메모리↑

# =========================================================
# 🔹 Helper functions
# =========================================================
def load_bvecs(fname, num_vectors=None):
    with open(fname, "rb") as f:
        d = np.frombuffer(f.read(4), dtype="int32")[0]
    filesize    = os.path.getsize(fname)
    record_size = 4 + d
    total       = filesize // record_size
    n = min(num_vectors, total) if num_vectors else total
    mm   = np.memmap(fname, dtype="uint8", mode="r")[:n * record_size]
    return mm.reshape(n, record_size)[:, 4:].astype("float32")

def save_fvecs(fname, data):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    n, d = data.shape
    with open(fname, "wb") as f:
        for i in range(n):
            f.write(np.array([d], dtype="int32").tobytes())
            f.write(data[i].astype("float32").tobytes())
    info(f"  - Saved {n} vectors to: {fname}")

def reconstruct_from_hnswpq(index, n):
    """
    IndexHNSWPQ 에서 복원 벡터를 얻는 함수.
    index.storage 를 downcast_index()로 IndexPQ 타입으로 변환한 뒤 pq.decode() 로 복원.
    """
    storage = faiss.downcast_index(index.storage)  # storage → IndexPQ
    pq      = storage.pq                           # ProductQuantizer
    M       = pq.M
    codes   = faiss.vector_to_array(storage.codes).reshape(n, M)   # (n, M) uint8
    # Python 바인딩: pq.decode(codes) → (n * d,) float32
    reconstructed = pq.decode(codes).reshape(n, pq.d)
    return reconstructed   # (n, DIM)

# =========================================================
# 🔹 Main
# =========================================================
if __name__ == "__main__":

    # 1. Data Load
    info("1. Data Load")
    xb = load_bvecs(BASE_FILE, NUM_BASE)
    xq = load_bvecs(QUERY_FILE)
    xt = load_bvecs(LEARN_FILE, NUM_LEARN)

    info(f"  - Base vectors:  {xb.shape}")
    info(f"  - Query vectors: {xq.shape}")
    info(f"  - Learn vectors: {xt.shape}")

    save_fvecs(BASE10M_SAVE_PATH, xb)

    # 2. Train & Add — IndexHNSWPQ (인덱스 파일 존재 시 로드 후 건너뜀)
    if os.path.exists(INDEX_SAVE_PATH):
        info(f"2. HNSWPQ index already exists — loading from file")
        index = faiss.read_index(INDEX_SAVE_PATH)
        info(f"  - Loaded: {INDEX_SAVE_PATH}  (ntotal={index.ntotal:,})")
        train_time_ms = 0.0
        add_time_ms   = 0.0
    else:
        info("2. Train Index & Add Index (IndexHNSWPQ)")
        index = faiss.IndexHNSWPQ(DIM, PQ_M, HNSW_M, PQ_NBITS)
        info(f"  - Index type: IndexHNSWPQ (PQ_M={PQ_M}, HNSW_M={HNSW_M}, nbits={PQ_NBITS})")
        index.hnsw.efConstruction = 80
        info(f"  - efConstruction: {index.hnsw.efConstruction}")

        train_start = time.perf_counter_ns()
        index.train(xt)
        train_end   = time.perf_counter_ns()
        train_time_ms = (train_end - train_start) / 1e6
        info(f"  - Training done in {train_time_ms:.3f} ms")

        add_start = time.perf_counter_ns()
        index.add(xb)
        add_end   = time.perf_counter_ns()
        add_time_ms = (add_end - add_start) / 1e6
        info(f"  - Adding done in {add_time_ms:.3f} ms")
        info(f"  - Total vectors in index: {index.ntotal:,}")

        # 3. Save Index
        info("3. Save Index")
        os.makedirs(os.path.dirname(INDEX_SAVE_PATH), exist_ok=True)
        faiss.write_index(index, INDEX_SAVE_PATH)
        info(f"  - HNSWPQ index saved to: {INDEX_SAVE_PATH}")

    # 4. Build Residual PQ Index
    #    IndexHNSWPQ의 내부 storage(IndexPQ)에서 복원 벡터를 얻어
    #    잔차(residual)를 계산한 뒤 일반 IndexPQ로 구성
    info("4. Build Residual PQ Index (based on HNSWPQ reconstruction)")

    info("  - Reconstructing PQ approximations from HNSWPQ storage...")
    pq_reconstructed = reconstruct_from_hnswpq(index, index.ntotal)
    residual_vectors = xb - pq_reconstructed
    info(f"  - Residual vectors shape: {residual_vectors.shape}")

    residual_index = faiss.IndexPQ(DIM, PQ_M, PQ_NBITS)
    info(f"  - Residual Index type: IndexPQ (M={PQ_M}, nbits={PQ_NBITS})")

    res_train_start = time.perf_counter_ns()
    residual_index.train(residual_vectors)
    res_train_end   = time.perf_counter_ns()
    res_train_time_ms = (res_train_end - res_train_start) / 1e6
    info(f"  - Residual training done in {res_train_time_ms:.3f} ms")

    res_add_start = time.perf_counter_ns()
    residual_index.add(residual_vectors)
    res_add_end   = time.perf_counter_ns()
    res_add_time_ms = (res_add_end - res_add_start) / 1e6
    info(f"  - Residual adding done in {res_add_time_ms:.3f} ms")
    info(f"  - Total residual vectors in index: {residual_index.ntotal:,}")

    # 5. Save Residual Index
    info("5. Save Residual Index")
    os.makedirs(os.path.dirname(RESIDUAL_INDEX_PATH), exist_ok=True)
    faiss.write_index(residual_index, RESIDUAL_INDEX_PATH)
    info(f"  - Residual index saved to: {RESIDUAL_INDEX_PATH}")

    # 6. Calculate and Save Residual Norm Squared
    info("6. Calculate and Save Residual Norm Squared")
    residual_norm_sq = np.sum(residual_vectors ** 2, axis=1)   # (10M,)
    info(f"  - Residual norm squared shape: {residual_norm_sq.shape}")

    os.makedirs(os.path.dirname(RESIDUAL_NORM_SQ_PATH), exist_ok=True)
    np.savez_compressed(RESIDUAL_NORM_SQ_PATH, residual_norm_sq=residual_norm_sq)
    info(f"  - Residual norm squared saved to: {RESIDUAL_NORM_SQ_PATH}")

    # 7. Save Timing Metrics
    info("7. Save Timing Metrics")
    metrics = {
        "creation_date": CREATION_DATE,
        "index_type": "IndexHNSWPQ",
        "pq_m": PQ_M,
        "pq_nbits": PQ_NBITS,
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": index.hnsw.efConstruction,
        "base_hnswpq": {
            "train_time_ms": round(train_time_ms, 3),
            "add_time_ms":   round(add_time_ms, 3),
            "total_time_ms": round(train_time_ms + add_time_ms, 3),
        },
        "residual_pq": {
            "train_time_ms": round(res_train_time_ms, 3),
            "add_time_ms":   round(res_add_time_ms, 3),
            "total_time_ms": round(res_train_time_ms + res_add_time_ms, 3),
        },
        "total_build_time_ms": round(
            train_time_ms + add_time_ms + res_train_time_ms + res_add_time_ms, 3),
    }

    os.makedirs(os.path.dirname(METRIC_SAVE_PATH), exist_ok=True)
    with open(METRIC_SAVE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    info(f"  - Timing metrics saved to: {METRIC_SAVE_PATH}")
