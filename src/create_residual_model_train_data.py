#!/usr/bin/env python3
"""
create_residual_model_train_data.py
────────────────────────────────────
build_index.py 결과물을 이용해 Residual Feature / Label을 생성합니다.

[출력]
  residual_features.npz  : shape (160000, 16, 9)
                           - 9 dims per subspace, 16 subspaces
                           - 10000 queries x 16 candidates = 160000 rows
  residual_label.npz     : shape (160000, 1)
                           - label = dot(Q-P, X-P)  (전체 128차원 합산)

[Feature 구성 (9 dims per subspace)]
  product_vec (8 dims): (Q - P) * res_reconstructed  element-wise
  feat_res_norm (1 dim): ||res_reconstructed|| / sqrt(8)

[의존 파일]
  - {creation_date}_pq.index          : Base PQ Index (faiss)
  - {creation_date}_residual_pq.index : Residual PQ Index (faiss)
  - bigann_base.bvecs                 : Base vectors
  - bigann_query.bvecs                : Query vectors
"""

import faiss
faiss.omp_set_num_threads(8)

import numpy as np
import os
import time
from tqdm import tqdm

# =============================================================================
# 🔹 Configuration — 여기서 직접 설정하세요
# =============================================================================

# build_index.py 실행 시 찍힌 creation_date 값 (예: "2026022007")
CREATION_DATE = "2026022007"

# 데이터 경로
DATA_DIR        = "/home/syback/vectorDB/ann_datasets/sift1B"
BASE_FILE       = os.path.join(DATA_DIR, "bigann_base.bvecs")
QUERY_FILE      = os.path.join(DATA_DIR, "bigann_query.bvecs")

# Index 경로 (build_index.py 가 저장한 파일)
INDEX_DIR       = "/home/syback/vectorDB/on-device/data/index"
PQ_INDEX_PATH   = os.path.join(INDEX_DIR, f"{CREATION_DATE}_pq.index")
RES_INDEX_PATH  = os.path.join(INDEX_DIR, f"{CREATION_DATE}_residual_pq.index")

# 출력 경로
FEATURE_SAVE_DIR  = "/home/syback/vectorDB/on-device/data/model/residual"
FEATURE_SAVE_PATH = os.path.join(FEATURE_SAVE_DIR, "residual_features.npz")
LABEL_SAVE_PATH   = os.path.join(FEATURE_SAVE_DIR, "residual_label.npz")

# 검색/모델 파라미터
NUM_BASE        = 10_000_000   # Base 벡터 수
NUM_QUERY       = 10_000       # Query 벡터 수 (전체)
CANDIDATES      = 16           # Query당 후보 수 (top-k)
DIM             = 128          # 원본 차원
NUM_SUBSPACES   = 16           # PQ subspace 수
SUB_DIM         = DIM // NUM_SUBSPACES  # subspace 당 차원 (= 8)

# =============================================================================
# 🔹 Helper: bvecs 로더
# =============================================================================
def load_bvecs(fname, num_vectors=None):
    """bigann .bvecs 파일을 float32 ndarray로 로드합니다."""
    with open(fname, "rb") as f:
        d = np.frombuffer(f.read(4), dtype="int32")[0]
    filesize    = os.path.getsize(fname)
    record_size = 4 + d
    total       = filesize // record_size

    if num_vectors is not None:
        num_vectors = min(num_vectors, total)
    else:
        num_vectors = total

    mm   = np.memmap(fname, dtype="uint8", mode="r")
    mm   = mm[: num_vectors * record_size]
    data = mm.reshape(num_vectors, record_size)[:, 4:]
    return data.astype("float32")


# =============================================================================
# 🔹 Main
# =============================================================================
if __name__ == "__main__":
    t0 = time.perf_counter()

    # -------------------------------------------------------------------------
    # Step 1. 파일 존재 확인
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("  Residual Feature Generation")
    print("=" * 70)
    print(f"\n[Config]")
    print(f"  CREATION_DATE : {CREATION_DATE}")
    print(f"  PQ Index      : {PQ_INDEX_PATH}")
    print(f"  Residual Index: {RES_INDEX_PATH}")
    print(f"  Base File     : {BASE_FILE}")
    print(f"  Query File    : {QUERY_FILE}")

    for path in [PQ_INDEX_PATH, RES_INDEX_PATH, BASE_FILE, QUERY_FILE]:
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

    pq_index  = faiss.read_index(PQ_INDEX_PATH)
    res_index = faiss.read_index(RES_INDEX_PATH)
    print(f"    - PQ Index ntotal      : {pq_index.ntotal:,}")
    print(f"    - Residual Index ntotal: {res_index.ntotal:,}")

    # Base PQ Centroid 테이블  (16 subspaces × 256 centroids × 8 dims)
    pq_obj  = faiss.downcast_index(pq_index).pq    # IndexPQ → pq 필드
    res_pq_obj = faiss.downcast_index(res_index).pq

    M_pq  = pq_obj.M        # 16
    K_pq  = pq_obj.ksub     # 256
    dsub  = pq_obj.dsub     # 8

    # Centroid 배열 추출 : (M, K, dsub)
    pq_centroids = faiss.vector_to_array(pq_obj.centroids).reshape(M_pq, K_pq, dsub)
    res_centroids = faiss.vector_to_array(res_pq_obj.centroids).reshape(M_pq, K_pq, dsub)
    print(f"    - PQ  Centroids shape  : {pq_centroids.shape}")
    print(f"    - Res Centroids shape  : {res_centroids.shape}")

    # -------------------------------------------------------------------------
    # Step 4. Base 벡터의 PQ code 및 Residual code 계산
    # -------------------------------------------------------------------------
    print("\n>>> [3/5] Base 벡터의 PQ code / Residual code 계산 중...")

    # Base PQ codes  (N, M)  — reconstruct 없이 encode로 추출
    pq_codes_base = np.zeros((NUM_BASE, M_pq), dtype=np.uint8)
    res_codes_base = np.zeros((NUM_BASE, M_pq), dtype=np.uint8)

    BATCH = 500_000
    for start in tqdm(range(0, NUM_BASE, BATCH), desc="  Encoding Base"):
        end   = min(start + BATCH, NUM_BASE)
        chunk = xb[start:end]

        # Base PQ code  — compute_codes() 는 (n, M) uint8 numpy 배열 반환
        pq_codes_base[start:end] = pq_obj.compute_codes(chunk)  # (n, M)

        # PQ 재구성 벡터 → Residual 계산
        pq_recon  = pq_index.reconstruct_n(start, end - start)  # (n, 128)
        residuals = chunk - pq_recon

        # Residual PQ code
        res_codes_base[start:end] = res_pq_obj.compute_codes(residuals)  # (n, M)

    print(f"    - pq_codes_base  : {pq_codes_base.shape}  dtype={pq_codes_base.dtype}")
    print(f"    - res_codes_base : {res_codes_base.shape}  dtype={res_codes_base.dtype}")

    # -------------------------------------------------------------------------
    # Step 5. Query Search — top-16 후보 추출
    # -------------------------------------------------------------------------
    print(f"\n>>> [4/5] Query Search (top-{CANDIDATES}) 수행 중...")

    _, I = pq_index.search(xq, CANDIDATES)   # I: (10000, 16)
    print(f"    - Search result I : {I.shape}")

    # -------------------------------------------------------------------------
    # Step 6. Feature 생성  → (160000, 16, 9) / Label → (160000, 1)
    # -------------------------------------------------------------------------
    print("\n>>> [5/5] Feature Engineering 수행 중...")

    N_total = NUM_QUERY * CANDIDATES   # 160000
    flat_idx = I.flatten()             # (160000,)

    # Query를 16번 반복 확장
    Q_exp = np.repeat(xq, CANDIDATES, axis=0)   # (160000, 128)

    # 후보 Base 벡터 gather
    X_cand = xb[flat_idx]             # (160000, 128)
    pq_c   = pq_codes_base[flat_idx]  # (160000, 16)
    res_c  = res_codes_base[flat_idx] # (160000, 16)

    # subspace별 feature를 리스트에 담고 마지막에 stack
    features_list = []  # 각 원소: (160000, 9)
    # label: dot(Q-P, X-P) — 128차원 전체 dot product를 subspace 단위로 합산 → (160000, 1)
    labels_acc = np.zeros((N_total, 1), dtype=np.float32)

    for m in tqdm(range(NUM_SUBSPACES), desc="  Subspace Feature"):
        start_col = m * SUB_DIM
        end_col   = (m + 1) * SUB_DIM

        Q_sub    = Q_exp[:, start_col:end_col]   # (160000, 8)
        P_sub    = pq_centroids[m][pq_c[:, m]]  # (160000, 8)  — base PQ 재구성
        diff_vec = Q_sub - P_sub                 # (160000, 8)  — Q - P

        # Residual 재구성  (Residual PQ centroid lookup)
        res_reconstructed = res_centroids[m][res_c[:, m]]  # (160000, 8)

        # [Feature 1] product_vec: (Q-P) * res_reconstructed  element-wise
        product_vec = diff_vec * res_reconstructed  # (160000, 8)

        # [Feature 2] feat_res_norm: ||res_reconstructed|| / sqrt(8)
        feat_res_norm_sq       = np.sum(res_reconstructed ** 2, axis=1, keepdims=True)  # (160000, 1)
        feat_res_norm_div_sqrt8 = np.sqrt(feat_res_norm_sq) / np.sqrt(SUB_DIM)          # (160000, 1)

        # 이 subspace의 feature (160000, 9)
        feat_m = np.concatenate([product_vec, feat_res_norm_div_sqrt8], axis=1)
        features_list.append(feat_m)

        # [Label] dot(Q-P, X-P) — true residual (실제 원본 사용)
        X_sub     = X_cand[:, start_col:end_col]                          # (160000, 8)
        true_res  = X_sub - P_sub                                         # (160000, 8)  X - P
        dot_qp_xp = np.sum(diff_vec * true_res, axis=1, keepdims=True)    # (160000, 1)
        labels_acc += dot_qp_xp

    # (160000, 16, 9) — axis=1 에 subspace 쌓기
    X_final = np.stack(features_list, axis=1)  # (160000, 16, 9)
    y_final = labels_acc                        # (160000, 1)
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

    print("\n" + "=" * 70)
    print("[최종 Feature 구성 (160000, 16, 9)]")
    print("  axis=0 : 160000 = 10000 queries × 16 candidates")
    print("  axis=1 : 16 subspaces")
    print("  axis=2 : 9 dims per subspace")
    print("    Index 0~7 : product_vec  = (Q-P) * res_reconstructed  element-wise")
    print("    Index 8   : feat_res_norm = ||res_reconstructed|| / sqrt(8)")
    print("")
    print("[Label 구성 (160000, 1)]")
    print("  label = dot(Q-P, X-P)")
    print("        = sum over 16 subspaces of dot(Q_sub-P_sub, X_sub-P_sub)")
    print("        * X-P = true residual (실제 base 벡터 - PQ centroid)")
    print("=" * 70)
