#!/usr/bin/env python3
"""
create_residual_model_train_data_hnsw.py
─────────────────────────────────────────────────────────────────────
build_index_hnsw.py 결과물을 이용해 Residual Feature / Label을 생성합니다.
create_residual_model_train_data.py 의 HNSW 버전.

[IndexHNSWPQ 구조 차이]
  - pq_index.reconstruct_n() 미지원 → storage.pq.decode(codes) 로 복원
  - PQ object 접근: faiss.downcast_index(hnsw_index.storage).pq
  - PQ codes 접근: faiss.vector_to_array(storage.codes)

[출력]
  residual_features_hnsw.npz  : shape (160000, 16, 9)
  residual_label_hnsw.npz     : shape (160000, 1)

[저장 위치]
  data/model/residual_hnsw/

[의존 파일]
  - {CREATION_DATE}_hnswpq.index         : Base HNSW PQ Index (faiss)
  - {CREATION_DATE}_residual_pq_hnsw.index : Residual PQ Index (faiss)
  - bigann_base.bvecs                    : Base vectors
  - bigann_query.bvecs                   : Query vectors
"""

import faiss
faiss.omp_set_num_threads(8)

import numpy as np
import os
import time
from tqdm import tqdm

# =============================================================================
# 🔹 Configuration
# =============================================================================
CREATION_DATE = "2026022007"

DATA_DIR   = "/home/syback/vectorDB/ann_datasets/sift1B"
BASE_FILE  = os.path.join(DATA_DIR, "bigann_base.bvecs")
QUERY_FILE = os.path.join(DATA_DIR, "bigann_query.bvecs")

INDEX_DIR       = "/home/syback/vectorDB/on-device/data/index"
HNSW_INDEX_PATH = os.path.join(INDEX_DIR, f"{CREATION_DATE}_hnswpq.index")
RES_INDEX_PATH  = os.path.join(INDEX_DIR, f"{CREATION_DATE}_residual_pq_hnsw.index")

FEATURE_SAVE_DIR  = "/home/syback/vectorDB/on-device/data/model/residual_hnsw"
FEATURE_SAVE_PATH = os.path.join(FEATURE_SAVE_DIR, "residual_features_hnsw.npz")
LABEL_SAVE_PATH   = os.path.join(FEATURE_SAVE_DIR, "residual_label_hnsw.npz")

NUM_BASE      = 10_000_000
NUM_QUERY     = 10_000
CANDIDATES    = 16
DIM           = 128
NUM_SUBSPACES = 16
SUB_DIM       = DIM // NUM_SUBSPACES   # 8

# HNSW Search 품질 (학습 데이터 생성용: 높게 설정)
HNSW_EF_SEARCH = 256

# =============================================================================
# 🔹 Helper functions
# =============================================================================
def load_bvecs(fname, num_vectors=None):
    with open(fname, "rb") as f:
        d = np.frombuffer(f.read(4), dtype="int32")[0]
    filesize    = os.path.getsize(fname)
    record_size = 4 + d
    total       = filesize // record_size
    n = min(num_vectors, total) if num_vectors else total
    mm = np.memmap(fname, dtype="uint8", mode="r")[:n * record_size]
    return mm.reshape(n, record_size)[:, 4:].astype("float32")

def decode_pq(pq_obj, codes):
    """
    pq_obj.decode(codes) → (n, d) float32
    codes: (n, M) uint8
    """
    return pq_obj.decode(codes).reshape(len(codes), pq_obj.d)

# =============================================================================
# 🔹 Main
# =============================================================================
if __name__ == "__main__":
    t0 = time.perf_counter()

    # -------------------------------------------------------------------------
    # Step 1. 파일 확인
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("  Residual Feature Generation  [HNSW version]")
    print("=" * 70)
    print(f"\n[Config]")
    print(f"  CREATION_DATE   : {CREATION_DATE}")
    print(f"  HNSW Index      : {HNSW_INDEX_PATH}")
    print(f"  Residual Index  : {RES_INDEX_PATH}")
    print(f"  Base File       : {BASE_FILE}")
    print(f"  Query File      : {QUERY_FILE}")
    print(f"  efSearch        : {HNSW_EF_SEARCH}  (학습 데이터용 고품질 탐색)")

    for path in [HNSW_INDEX_PATH, RES_INDEX_PATH, BASE_FILE, QUERY_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    print("\n  ✓ 모든 입력 파일 확인 완료\n")

    # -------------------------------------------------------------------------
    # Step 2. 데이터 로드
    # -------------------------------------------------------------------------
    print(">>> [1/5] 데이터 로딩 중...")
    xb = load_bvecs(BASE_FILE, NUM_BASE)    # (10M, 128)
    xq = load_bvecs(QUERY_FILE, NUM_QUERY)  # (10000, 128)
    print(f"    - Base  vectors : {xb.shape}")
    print(f"    - Query vectors : {xq.shape}")

    # -------------------------------------------------------------------------
    # Step 3. Faiss Index 로드 및 PQ 정보 추출
    # -------------------------------------------------------------------------
    print("\n>>> [2/5] Faiss Index 로딩 및 PQ Centroid 추출 중...")

    hnsw_index = faiss.read_index(HNSW_INDEX_PATH)
    res_index  = faiss.read_index(RES_INDEX_PATH)
    hnsw_index.hnsw.efSearch = HNSW_EF_SEARCH
    print(f"    - HNSW Index ntotal    : {hnsw_index.ntotal:,}")
    print(f"    - Residual Index ntotal: {res_index.ntotal:,}")

    # ── PQ object 접근 (IndexHNSWPQ 구조 차이) ──────────────────────────────
    # IndexHNSWPQ: storage = IndexPQ → faiss.downcast_index(storage) 필요
    hnsw_storage = faiss.downcast_index(hnsw_index.storage)   # IndexPQ
    pq_obj       = hnsw_storage.pq                             # ProductQuantizer

    res_index_down = faiss.downcast_index(res_index)
    res_pq_obj     = res_index_down.pq

    M_pq = pq_obj.M; K_pq = pq_obj.ksub; dsub = pq_obj.dsub

    pq_centroids  = faiss.vector_to_array(pq_obj.centroids).reshape(M_pq, K_pq, dsub)
    res_centroids = faiss.vector_to_array(res_pq_obj.centroids).reshape(M_pq, K_pq, dsub)
    print(f"    - PQ  Centroids shape  : {pq_centroids.shape}")
    print(f"    - Res Centroids shape  : {res_centroids.shape}")

    # ── HNSW 내장 PQ codes 전체 추출 ────────────────────────────────────────
    # IndexHNSWPQ: 코드는 storage.codes 에 저장
    pq_codes_all = faiss.vector_to_array(hnsw_storage.codes).reshape(
        hnsw_index.ntotal, M_pq).copy()   # (10M, 16) uint8
    print(f"    - HNSW PQ codes shape  : {pq_codes_all.shape}  (from storage)")

    # -------------------------------------------------------------------------
    # Step 4. Base 벡터의 PQ 복원 및 Residual PQ code 계산
    # -------------------------------------------------------------------------
    print("\n>>> [3/5] Base 벡터 PQ 복원 및 Residual code 계산 중...")

    # IndexHNSWPQ는 reconstruct_n() 미지원 → storage.pq.decode() 로 직접 복원
    # Residual code는 원본과 동일하게 res_pq_obj.compute_codes() 사용
    res_codes_base = np.zeros((NUM_BASE, M_pq), dtype=np.uint8)

    BATCH = 500_000
    for start in tqdm(range(0, NUM_BASE, BATCH), desc="  Encoding Residual"):
        end   = min(start + BATCH, NUM_BASE)
        chunk = xb[start:end]

        # PQ 복원: storage.pq.decode(codes_chunk)
        codes_chunk = pq_codes_all[start:end]       # (n, M) uint8
        pq_recon    = decode_pq(pq_obj, codes_chunk) # (n, 128)
        residuals   = chunk - pq_recon

        # Residual PQ code
        res_codes_base[start:end] = res_pq_obj.compute_codes(residuals)

    print(f"    - pq_codes_all   : {pq_codes_all.shape}  dtype={pq_codes_all.dtype}")
    print(f"    - res_codes_base : {res_codes_base.shape}  dtype={res_codes_base.dtype}")

    # -------------------------------------------------------------------------
    # Step 5. Query Search — top-16 후보 추출 (HNSW search)
    # -------------------------------------------------------------------------
    print(f"\n>>> [4/5] Query Search (top-{CANDIDATES}, efSearch={HNSW_EF_SEARCH}) 수행 중...")

    _, I = hnsw_index.search(xq, CANDIDATES)   # I: (10000, 16)
    print(f"    - Search result I : {I.shape}")

    # -------------------------------------------------------------------------
    # Step 6. Feature 생성  → (160000, 16, 9) / Label → (160000, 1)
    # -------------------------------------------------------------------------
    print("\n>>> [5/5] Feature Engineering 수행 중...")

    N_total  = NUM_QUERY * CANDIDATES   # 160000
    flat_idx = I.flatten()              # (160000,)

    Q_exp  = np.repeat(xq, CANDIDATES, axis=0)   # (160000, 128)
    X_cand = xb[flat_idx]                         # (160000, 128)
    pq_c   = pq_codes_all[flat_idx]               # (160000, 16) ← HNSW storage에서
    res_c  = res_codes_base[flat_idx]             # (160000, 16)

    features_list = []
    labels_acc    = np.zeros((N_total, 1), dtype=np.float32)

    for m in tqdm(range(NUM_SUBSPACES), desc="  Subspace Feature"):
        s = m * SUB_DIM; e = (m + 1) * SUB_DIM

        Q_sub    = Q_exp[:, s:e]               # (160000, 8)
        P_sub    = pq_centroids[m][pq_c[:, m]] # (160000, 8)
        diff_vec = Q_sub - P_sub               # Q - P

        res_reconstructed = res_centroids[m][res_c[:, m]]  # (160000, 8)

        # Feature
        product_vec            = diff_vec * res_reconstructed              # (160000, 8)
        feat_res_norm          = np.sqrt(np.sum(res_reconstructed ** 2, axis=1, keepdims=True)) / np.sqrt(SUB_DIM)
        feat_m                 = np.concatenate([product_vec, feat_res_norm], axis=1)  # (160000, 9)
        features_list.append(feat_m)

        # Label: dot(Q-P, X-P)
        X_sub     = X_cand[:, s:e]
        true_res  = X_sub - P_sub
        dot_qp_xp = np.sum(diff_vec * true_res, axis=1, keepdims=True)
        labels_acc += dot_qp_xp

    X_final = np.stack(features_list, axis=1)   # (160000, 16, 9)
    y_final = labels_acc                         # (160000, 1)
    print(f"\n    - Final Feature Shape : {X_final.shape}  (Expected: ({N_total}, 16, 9))")
    print(f"    - Final Label   Shape : {y_final.shape}  (Expected: ({N_total}, 1))")

    # -------------------------------------------------------------------------
    # Step 7. 저장
    # -------------------------------------------------------------------------
    os.makedirs(FEATURE_SAVE_DIR, exist_ok=True)
    np.savez_compressed(FEATURE_SAVE_PATH, data=X_final)
    print(f"\n    ✓ Saved Feature: {FEATURE_SAVE_PATH}")
    np.savez_compressed(LABEL_SAVE_PATH, data=y_final)
    print(f"    ✓ Saved Label  : {LABEL_SAVE_PATH}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("=" * 70)
