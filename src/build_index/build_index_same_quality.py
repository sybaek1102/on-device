#!/usr/bin/env python3
"""
build_index_same_quality.py
──────────────────────────────────────────────────────────────
기존 build_index.py와 동일하지만 PQ_M=32로 설정한 버전.
Step 1~3 (Data Load → Train & Add Index → Save Index) 까지만 실행.
Residual PQ / Residual Norm Sq 는 생성하지 않음.

[변경 사항]
  - CREATION_DATE = "2026022007"  (고정)
  - PQ_M = 32
  - Step 1~3 실행 후 metric 저장
  - metric 파일명: 2026022007_build_time_same_quality.json
"""

import faiss
faiss.omp_set_num_threads(8)  # core 수 제한

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

NUM_BASE  = 10_000_000   # 10M base vectors
NUM_LEARN =  1_000_000   # 1M learn vectors

CREATION_DATE = "2026022007"   # 고정

INDEX_SAVE_PATH  = f"/home/syback/vectorDB/on-device/data/index/{CREATION_DATE}_same_quality_pq.index"
BASE10M_SAVE_PATH = f"/home/syback/vectorDB/on-device/data/raw/{CREATION_DATE}_base10M.fvecs"
METRIC_SAVE_PATH  = f"/home/syback/vectorDB/on-device/output/metric/{CREATION_DATE}_build_time_same_quality.json"

DIM      = 128
PQ_M     = 32    # ← 변경 (기존 16 → 32)
PQ_NBITS = 8

# =========================================================
# 🔹 Helper functions
# =========================================================
def load_bvecs(fname, num_vectors=None):
    with open(fname, "rb") as f:
        d = np.frombuffer(f.read(4), dtype="int32")[0]

    filesize    = os.path.getsize(fname)
    record_size = 4 + d
    total_vectors = filesize // record_size

    if num_vectors is not None:
        num_vectors = min(num_vectors, total_vectors)
    else:
        num_vectors = total_vectors

    mm   = np.memmap(fname, dtype="uint8", mode="r")
    mm   = mm[: num_vectors * record_size]
    data = mm.reshape(num_vectors, record_size)[:, 4:]
    return data.astype("float32")

def save_fvecs(fname, data):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    n, d = data.shape
    with open(fname, "wb") as f:
        for i in range(n):
            f.write(np.array([d], dtype="int32").tobytes())
            f.write(data[i].astype("float32").tobytes())
    info(f"  - Saved {n} vectors to: {fname}")

# =========================================================
# 🔹 Main
# =========================================================
if __name__ == "__main__":

    # 1. Data Load
    info("1. Data Load")
    xb = load_bvecs(BASE_FILE, NUM_BASE)    # base vectors (10M)
    xq = load_bvecs(QUERY_FILE)             # query vectors (all)
    xt = load_bvecs(LEARN_FILE, NUM_LEARN)  # learn/train vectors (1M)

    info(f"  - Base vectors:  {xb.shape}")
    info(f"  - Query vectors: {xq.shape}")
    info(f"  - Learn vectors: {xt.shape}")

    save_fvecs(BASE10M_SAVE_PATH, xb)

    # 2. Train Index & Add Index
    info("2. Train Index & Add Index")
    index = faiss.IndexPQ(DIM, PQ_M, PQ_NBITS)
    info(f"  - Index type: PQ (M={PQ_M}, nbits={PQ_NBITS})")

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
    info(f"  - Index saved to: {INDEX_SAVE_PATH}")

    # 4. Save Timing Metrics
    info("4. Save Timing Metrics")
    metrics = {
        "creation_date": CREATION_DATE,
        "pq_m": PQ_M,
        "pq_nbits": PQ_NBITS,
        "base_pq": {
            "train_time_ms":  round(train_time_ms, 3),
            "add_time_ms":    round(add_time_ms, 3),
            "total_time_ms":  round(train_time_ms + add_time_ms, 3),
        },
        "total_build_time_ms": round(train_time_ms + add_time_ms, 3),
    }

    os.makedirs(os.path.dirname(METRIC_SAVE_PATH), exist_ok=True)
    with open(METRIC_SAVE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    info(f"  - Timing metrics saved to: {METRIC_SAVE_PATH}")
