#!/usr/bin/env python3
"""
create_reranking_model_train_data.py
────────────────────────────────────
build_index.py 결과물 + residual OOF 예측으로 Re-ranking Feature/Label을 생성합니다.

[출력]
  re-ranking_features.npz : shape (10000, 33)  →  data/model/re-ranking/
                             - PQ Distance   (16 dims) : ||Q-P||² per subspace
                             - Residual Dist (16 dims) : ||X-P||² - 2·pred(dot(Q-P,X-P))
                             - Label          (1 dim)  : 0/1 binary

[Label 정의 (01_create_re-ranking_label.py 동일)]
  0: pred_top1 == gt_top1          (이미 1등이 정답)
  1: 그 외                          (re-ranking 필요 or 정답 없음)
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

# build_index.py 실행 시 찍힌 creation_date 값
CREATION_DATE = "2026022007"

# Base / Query bvecs
DATA_DIR   = "/home/syback/vectorDB/ann_datasets/sift1B"
BASE_FILE  = os.path.join(DATA_DIR, "bigann_base.bvecs")
QUERY_FILE = os.path.join(DATA_DIR, "bigann_query.bvecs")

# Ground Truth ivecs
GT_FILE = os.path.join(DATA_DIR, "gnd", "idx_10M.ivecs")

# Faiss Index
INDEX_DIR     = "/home/syback/vectorDB/on-device/data/index"
PQ_INDEX_PATH = os.path.join(INDEX_DIR, f"{CREATION_DATE}_pq.index")

# Residual OOF 예측 (model_train_residual.py 결과)
OOF_PRED_PATH = "/home/syback/vectorDB/on-device/data/model/residual/oof_pred.npz"

# 출력 경로
OUTPUT_DIR  = "/home/syback/vectorDB/on-device/data/model/re-ranking"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "re-ranking_features.npz")

# 파라미터
NUM_BASE      = 10_000_000
NUM_QUERY     = 10_000
CANDIDATES    = 16
DIM           = 128
NUM_SUBSPACES = 16
SUB_DIM       = DIM // NUM_SUBSPACES   # 8

# =============================================================================
# 🔹 Helper: bvecs 로더
# =============================================================================
def load_bvecs(fname, num_vectors=None):
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

def load_ivecs(fname):
    """ivecs 파일 로드 → (N, k) int32"""
    mm          = np.memmap(fname, dtype="int32", mode="r")
    k           = mm[0]
    record_size = k + 1
    nvecs       = mm.shape[0] // record_size
    return mm.reshape(nvecs, record_size)[:, 1:].copy()

# =============================================================================
# 🔹 Main
# =============================================================================
if __name__ == "__main__":
    t0 = time.perf_counter()

    print("=" * 70)
    print("  Re-ranking Feature & Label Generation")
    print("=" * 70)
    print(f"\n[Config]")
    print(f"  CREATION_DATE : {CREATION_DATE}")
    print(f"  PQ Index      : {PQ_INDEX_PATH}")
    print(f"  OOF Pred      : {OOF_PRED_PATH}")
    print(f"  GT File       : {GT_FILE}")

    for path in [PQ_INDEX_PATH, OOF_PRED_PATH, BASE_FILE, QUERY_FILE, GT_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    print("\n  ✓ 모든 입력 파일 확인 완료\n")

    # -------------------------------------------------------------------------
    # Step 1. 데이터 로드
    # -------------------------------------------------------------------------
    print(">>> [1/6] 데이터 로딩 중...")
    xb = load_bvecs(BASE_FILE, NUM_BASE)    # (10M, 128)
    xq = load_bvecs(QUERY_FILE, NUM_QUERY)  # (10000, 128)
    print(f"    - Base  vectors : {xb.shape}")
    print(f"    - Query vectors : {xq.shape}")

    # -------------------------------------------------------------------------
    # Step 2. GT 로드
    # -------------------------------------------------------------------------
    print("\n>>> [2/6] Ground Truth 로드 중...")
    gt_idx = load_ivecs(GT_FILE)            # (10000, k_gt)
    gt_top1 = gt_idx[:, 0]                 # (10000,)  GT 1등 index
    print(f"    - GT shape      : {gt_idx.shape}")

    # -------------------------------------------------------------------------
    # Step 3. PQ Index 로드 + Query Search
    # -------------------------------------------------------------------------
    print("\n>>> [3/6] PQ Index 로드 & Query Search 중...")
    pq_index   = faiss.read_index(PQ_INDEX_PATH)
    pq_obj     = faiss.downcast_index(pq_index).pq
    M_pq       = pq_obj.M       # 16
    K_pq       = pq_obj.ksub    # 256
    dsub       = pq_obj.dsub    # 8

    # Centroid 테이블 추출 (16, 256, 8)
    pq_centroids = faiss.vector_to_array(pq_obj.centroids).reshape(M_pq, K_pq, dsub)

    # Query search → I: (10000, 16)
    D_pq, I = pq_index.search(xq, CANDIDATES)
    print(f"    - PQ Index ntotal : {pq_index.ntotal:,}")
    print(f"    - Search I shape  : {I.shape}")
    print(f"    - Search D shape  : {D_pq.shape}")

    # -------------------------------------------------------------------------
    # Step 4. OOF Residual 예측 로드
    # -------------------------------------------------------------------------
    print("\n>>> [4/6] OOF Residual 예측 로드 중...")
    with np.load(OOF_PRED_PATH) as f:
        oof_preds = f["pred"]               # (160000, 1)
    oof_preds_2d = oof_preds.reshape(NUM_QUERY, CANDIDATES)  # (10000, 16)
    print(f"    - OOF pred shape  : {oof_preds.shape} → reshaped: {oof_preds_2d.shape}")

    # -------------------------------------------------------------------------
    # Step 5. Feature 계산
    # -------------------------------------------------------------------------
    print("\n>>> [5/6] Feature 계산 중...")

    flat_idx = I.flatten()              # (160000,)

    # --- [Feature A] PQ Distance per subspace ---
    # pq_dist[i, m] = ||Q_m - P_m||²  (이미 D_pq 가 전체 거리의 합이므로 subspace별 재계산)
    pq_codes_cand = pq_obj.compute_codes(xb[flat_idx])  # (160000, 16) uint8

    # Query expand
    Q_exp = np.repeat(xq, CANDIDATES, axis=0)           # (160000, 128)

    pq_dist_flat = np.zeros((NUM_QUERY * CANDIDATES, NUM_SUBSPACES), dtype=np.float32)
    for m in tqdm(range(NUM_SUBSPACES), desc="  PQ dist subspace"):
        start_col = m * SUB_DIM
        end_col   = (m + 1) * SUB_DIM
        Q_sub     = Q_exp[:, start_col:end_col]              # (160000, 8)
        P_sub     = pq_centroids[m][pq_codes_cand[:, m]]    # (160000, 8)
        diff      = Q_sub - P_sub
        pq_dist_flat[:, m] = np.sum(diff ** 2, axis=1)      # ||Q-P||² per subspace

    pq_dist_2d = pq_dist_flat.reshape(NUM_QUERY, CANDIDATES, NUM_SUBSPACES)  # (10000, 16, 16)
    # 후보별 subspace 합산 → (10000, 16)  (각 후보의 전체 PQ 거리)
    pq_dist_per_cand = pq_dist_2d.sum(axis=2)              # (10000, 16)

    print(f"    - PQ Distance shape : {pq_dist_per_cand.shape}")

    # --- [Feature B] ||X-P||² 직접 계산 ---
    X_cand = xb[flat_idx].reshape(NUM_QUERY, CANDIDATES, DIM)  # (10000, 16, 128)

    # PQ 재구성 벡터 계산
    pq_recon_flat = pq_index.reconstruct_n(0, 1)  # shape 확인용 dummy
    # 후보 벡터들의 PQ 재구성 벡터 계산
    X_flat  = xb[flat_idx]                                        # (160000, 128)
    pq_recon_flat = np.zeros_like(X_flat)
    ENCODE_BATCH = 500_000
    for start in tqdm(range(0, len(flat_idx), ENCODE_BATCH), desc="  Recon XP"):
        end = min(start + ENCODE_BATCH, len(flat_idx))
        codes_chunk = pq_obj.compute_codes(X_flat[start:end])     # (n, 16)
        # centroid lookup per subspace
        recon_chunk = np.zeros((end - start, DIM), dtype=np.float32)
        for m in range(NUM_SUBSPACES):
            s = m * SUB_DIM
            e = (m + 1) * SUB_DIM
            recon_chunk[:, s:e] = pq_centroids[m][codes_chunk[:, m]]
        pq_recon_flat[start:end] = recon_chunk

    residual_xp = X_flat - pq_recon_flat                          # (160000, 128)  X - P
    xp_normsq   = np.sum(residual_xp ** 2, axis=1)               # (160000,)  ||X-P||²
    xp_normsq_2d = xp_normsq.reshape(NUM_QUERY, CANDIDATES)      # (10000, 16)

    print(f"    - ||X-P||² shape    : {xp_normsq_2d.shape}")

    # --- [Feature B final] ||X-P||² - 2 * OOF_pred ---
    residual_feat = xp_normsq_2d - 2.0 * oof_preds_2d            # (10000, 16)
    print(f"    - Residual feat     : {residual_feat.shape}")

    # --- Feature 병합 (10000, 32) ---
    final_features = np.hstack([pq_dist_per_cand, residual_feat])  # (10000, 32)
    print(f"    - Final features    : {final_features.shape}")

    # -------------------------------------------------------------------------
    # Step 6. Label 생성 (01_create_re-ranking_label.py 동일 로직)
    # -------------------------------------------------------------------------
    print("\n>>> [6/6] Label 생성 중...")

    labels = np.zeros((NUM_QUERY, 1), dtype=np.int32)
    for i in range(NUM_QUERY):
        pred_top1   = I[i, 0]
        pred_all    = I[i, :]
        gt          = gt_top1[i]

        if pred_top1 == gt:
            labels[i, 0] = 0   # 이미 1등이 정답
        else:
            labels[i, 0] = 1   # re-ranking 필요 or 정답 없음

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    - Label {u}: {c:,} ({c/NUM_QUERY*100:.2f}%)")

    # -------------------------------------------------------------------------
    # 저장
    # -------------------------------------------------------------------------
    final_data = np.hstack([final_features, labels.astype(np.float32)])  # (10000, 33)
    print(f"\n    - Final data shape  : {final_data.shape}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, data=final_data)
    print(f"\n    ✓ Saved: {OUTPUT_PATH}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("[Feature 구성 (10000, 33)]")
    print("  Index  0~15 : PQ Distance    (||Q-P||²  per each of 16 candidates)")
    print("  Index 16~31 : Residual Dist  (||X-P||² - 2·pred(dot(Q-P,X-P)))")
    print("  Index 32    : Label          (0: 정답이 이미 1등 / 1: re-ranking 필요)")
    print("=" * 70)
