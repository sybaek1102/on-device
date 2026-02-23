#!/usr/bin/env python3
"""
query_search_selective_re-ranking_hnsw.py
──────────────────────────────────────────
Selective Re-ranking (HNSW버전): HNSWPQ Index로 top-16 후보를 추출하고
MLP이 re-ranking 필요 여부를 판단한 뒤 필요 시에만 L2 re-ranking을 수행합니다.

[OOF (Out-of-Fold) Inference 방식]
  - 10,000 쿼리를 1,000개씩 10구간으로 분할
  - 구간 k (0~9) 의 쿼리에는 fold k+1 모델만 사용
  - fold k 모델은 학습 시 해당 구간 쿼리를 validation set으로 사용
  - 모델 10개는 시작 시 한 번만 로드, 루프 내 인덱스 전환만 수행
    → 루프 중 추가 I/O/메모리 peak 없이 all/no 와 공평한 비교 가능

[파이프라인 (쿼리 1개)]
  1. PQ Search → top-16 candidates (D, I)
  2. Residual Feature 계산 (16 candidates × 16 subspaces × 9 dims)
  3. 해당 fold의 Residual MLP → pred_dot(Q-P, X-P) per candidate
  4. residual_norm_sq에서 ||X-P||² 조회
  5. residual_dist = ||X-P||² - 2·pred_dot
  6. 해당 fold의 Re-ranking MLP (fold별 scaler 적용) → 0 or 1
  7. 결과가 1 → base10M.fvecs에서 실제 L2 re-ranking
     결과가 0 → PQ top-1 그대로 반환

[저장]
  output/metric/{name}_query_search.json
"""

import faiss
faiss.omp_set_num_threads(1)

import numpy as np
import torch
import torch.nn as nn
import os
import time
import json
import subprocess

# =============================================================================
# 🔹 Configuration
# =============================================================================
CREATION_DATE  = "2026022007"
NAME           = "selective_hnsw"
THRESHOLD      = 0.5      # re-ranking MLP 임계값
HNSW_EF_SEARCH = 64       # HNSW 검색 품질

# =============================================================================
# 🔹 K-Fold 평가 설정
#    EVAL_FOLD : 1~10 중 하나를 지정하면 해당 fold의 쿼리 구간(1,000개)만 실행
#                → all / no 스크립트와 동일 쿼리셋으로 공평한 비교 가능
#                → selective는 해당 fold 모델만 사용 (OOF: 학습 미포함 보장)
# =============================================================================
EVAL_FOLD  = 1           # 1~10 (1-indexed)
NUM_FOLDS  = 10

DATA_DIR   = "/home/syback/vectorDB/ann_datasets/sift1B"
QUERY_FILE = os.path.join(DATA_DIR, "bigann_query.bvecs")
GT_FILE    = os.path.join(DATA_DIR, "gnd", "idx_10M.ivecs")

BASE10M_FILE = "/home/syback/vectorDB/on-device/data/raw/bigann_base10M.fvecs"

INDEX_DIR         = "/home/syback/vectorDB/on-device/data/index"
HNSW_INDEX_PATH   = os.path.join(INDEX_DIR, f"{CREATION_DATE}_hnswpq.index")
RES_INDEX_PATH    = os.path.join(INDEX_DIR, f"{CREATION_DATE}_residual_pq_hnsw.index")

RESIDUAL_NORM_SQ_PATH = f"/home/syback/vectorDB/on-device/data/features/{CREATION_DATE}_residual_norm_sq_hnsw.npz"

RESIDUAL_MODEL_DIR  = "/home/syback/vectorDB/on-device/data/model/residual_hnsw"
RERANKING_MODEL_DIR = "/home/syback/vectorDB/on-device/data/model/re-ranking_hnsw"

METRIC_DIR    = "/home/syback/vectorDB/on-device/output/metric"
METRIC_PATH   = os.path.join(METRIC_DIR, f"{NAME}_query_search_fold{EVAL_FOLD}.json")
TIMINGS_PATH  = os.path.join(METRIC_DIR, f"{NAME}_query_timings_fold{EVAL_FOLD}.json")

NUM_QUERY_TOTAL = 10_000
QUERIES_PER_FOLD = NUM_QUERY_TOTAL // NUM_FOLDS   # 1,000
Q_START   = (EVAL_FOLD - 1) * QUERIES_PER_FOLD    # inclusive
Q_END     = EVAL_FOLD * QUERIES_PER_FOLD           # exclusive
NUM_QUERY = QUERIES_PER_FOLD                       # 이번 실행에서 처리할 쿼리 수

CANDIDATES    = 16
DIM           = 128
NUM_SUBSPACES = 16
SUB_DIM       = DIM // NUM_SUBSPACES

# =============================================================================
# 🔹 Model Definitions (훈련 코드와 동일한 구조 필수)
# =============================================================================
FEATURE_DIM_RES  = 9
SHARED_HIDDEN    = 32
EMBED_DIM        = 8
GLOBAL_HIDDEN    = 64

class ResidualDistancePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(FEATURE_DIM_RES)
        self.shared_mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM_RES, SHARED_HIDDEN), nn.LeakyReLU(0.1),
            nn.Linear(SHARED_HIDDEN, EMBED_DIM),        nn.LeakyReLU(0.1)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(NUM_SUBSPACES * EMBED_DIM, GLOBAL_HIDDEN), nn.LeakyReLU(0.1),
            nn.Linear(GLOBAL_HIDDEN, 32),                         nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        b = x.size(0)
        x_flat = x.view(-1, FEATURE_DIM_RES)
        x_norm = self.input_norm(x_flat)
        emb    = self.shared_mlp(x_norm)
        return self.global_mlp(emb.view(b, -1))

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8,  1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# =============================================================================
# 🔹 Helper Functions
# =============================================================================
def load_bvecs(fname, num_vectors=None):
    with open(fname, "rb") as f:
        d = np.frombuffer(f.read(4), dtype="int32")[0]
    filesize    = os.path.getsize(fname)
    record_size = 4 + d
    total       = filesize // record_size
    n           = min(num_vectors, total) if num_vectors else total
    mm          = np.memmap(fname, dtype="uint8", mode="r")[:n * record_size]
    return mm.reshape(n, record_size)[:, 4:].astype("float32")

def load_ivecs(fname):
    mm = np.memmap(fname, dtype="int32", mode="r")
    k  = mm[0]; rs = k + 1
    return mm.reshape(mm.shape[0] // rs, rs)[:, 1:].copy()

def open_fvecs_memmap(fname):
    with open(fname, "rb") as f:
        d = np.frombuffer(f.read(4), dtype="int32")[0]
    total = os.path.getsize(fname) // ((1 + d) * 4)
    raw   = np.memmap(fname, dtype="float32", mode="r").reshape(total, 1 + d)
    return raw[:, 1:]

def get_io_bytes():
    try:
        with open(f"/proc/{os.getpid()}/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0

def get_rss_mb():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0

def get_peak_mb():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmPeak:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0

def drop_cache():
    try:
        r = subprocess.run(["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
                           capture_output=True, text=True, timeout=10)
        print("    ✓ 페이지 캐시 drop 완료" if r.returncode == 0
              else f"    ⚠ 캐시 drop 실패: {r.stderr.strip()}")
    except Exception as e:
        print(f"    ⚠ 캐시 drop 건너뜀: {e}")

def histogram_counts(values, bins):
    return {f"<={b:.1f}": int(np.sum(values <= b)) for b in bins}

# =============================================================================
# 🔹 Main
# =============================================================================
if __name__ == "__main__":
    BINS = [round(0.1 * i, 1) for i in range(1, 11)]
    DEVICE = torch.device("cpu")    # on-device: CPU

    print("=" * 70)
    print(f"  Query Search — Selective Re-ranking  (name='{NAME}')")
    print("=" * 70)

    total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Step 1. 데이터 로드
    # -------------------------------------------------------------------------
    print("\n>>> [1/6] 데이터 로드 중...")
    print(f"    - EVAL_FOLD : {EVAL_FOLD}  (쿼리 {Q_START}~{Q_END-1}, {NUM_QUERY}개)")
    xq_all  = load_bvecs(QUERY_FILE, Q_END)   # Q_END 까지만 읽기
    xq      = xq_all[Q_START:]                # (NUM_QUERY, 128)
    gt_all  = load_ivecs(GT_FILE)
    gt      = gt_all[Q_START:Q_END]           # (NUM_QUERY, k_gt)
    gt_top1 = gt[:, 0]
    del xq_all, gt_all
    print(f"    - Query : {xq.shape}")
    print(f"    - GT    : {gt.shape}")

    # Residual Norm Squared 전체 로드 (10M)
    with np.load(RESIDUAL_NORM_SQ_PATH) as f:
        res_norm_sq_all = f["residual_norm_sq"].astype(np.float32)  # (10M,)
    print(f"    - ResNormSq : {res_norm_sq_all.shape}")

    # -------------------------------------------------------------------------
    # Step 2. OS 캐시 비우기 + Index 로드
    # -------------------------------------------------------------------------
    print("\n>>> [2/6] OS 페이지 캐시 비우기...")
    drop_cache()

    print("\n    HNSW / Residual PQ Index 로드 중...")
    io_before_index  = get_io_bytes()
    mem_before_index = get_rss_mb()
    idx_load_start   = time.perf_counter()

    hnsw_index = faiss.read_index(HNSW_INDEX_PATH)
    hnsw_index.hnsw.efSearch = HNSW_EF_SEARCH
    res_index  = faiss.read_index(RES_INDEX_PATH)

    idx_load_end    = time.perf_counter()
    io_after_index  = get_io_bytes()
    mem_after_index = get_rss_mb()

    io_index_bytes = io_after_index - io_before_index
    index_load_ms  = (idx_load_end - idx_load_start) * 1000
    print(f"    - Load time       : {index_load_ms:.1f} ms")
    print(f"    - I/O (index)     : {io_index_bytes:,} bytes")
    print(f"    - Mem delta       : +{mem_after_index - mem_before_index:.1f} MB")
    print(f"    - efSearch        : {HNSW_EF_SEARCH}")

    # PQ / Residual PQ Centroid 추출
    # HNSWPQ: storage = IndexPQ → downcast 후 pq 접근
    hnsw_storage = faiss.downcast_index(hnsw_index.storage)   # IndexPQ
    pq_obj       = hnsw_storage.pq
    res_obj      = faiss.downcast_index(res_index).pq
    M_pq    = pq_obj.M; K_pq = pq_obj.ksub; dsub = pq_obj.dsub
    pq_centroids  = faiss.vector_to_array(pq_obj.centroids).reshape(M_pq, K_pq, dsub)
    res_centroids = faiss.vector_to_array(res_obj.centroids).reshape(M_pq, K_pq, dsub)

    # HNSW storage.codes 에서 PQ codes 추출 (10M × 16)
    pq_codes_all  = faiss.vector_to_array(hnsw_storage.codes
        ).reshape(hnsw_index.ntotal, M_pq).copy()   # (10M, 16) uint8
    res_codes_all = faiss.vector_to_array(
        faiss.downcast_index(res_index).codes
    ).reshape(res_index.ntotal, M_pq).copy()        # (10M, 16) uint8
    print(f"    - PQ codes   : {pq_codes_all.shape}  (HNSW storage 에서 추출)")
    print(f"    - Res codes  : {res_codes_all.shape}")

    # -------------------------------------------------------------------------
    # Step 3. MLP 모델 로드 (10개 앙상블)
    # -------------------------------------------------------------------------
    print("\n>>> [3/6] MLP 모델 로드 중...")

    residual_models  = []
    reranking_models = []
    scalers          = []   # (f1_mean, f1_std, f2_mean, f2_std) per fold

    for k in range(1, NUM_FOLDS + 1):
        # Residual MLP
        m_res = ResidualDistancePredictor().to(DEVICE)
        m_res.load_state_dict(torch.load(
            os.path.join(RESIDUAL_MODEL_DIR, f"model_k{k}.pt"), map_location=DEVICE))
        m_res.eval()
        residual_models.append(m_res)

        # Re-ranking MLP
        m_rer = SimpleMLP().to(DEVICE)
        m_rer.load_state_dict(torch.load(
            os.path.join(RERANKING_MODEL_DIR, f"model_k{k}.pt"), map_location=DEVICE))
        m_rer.eval()
        reranking_models.append(m_rer)

        # Scaler 파라미터
        sc = np.load(os.path.join(RERANKING_MODEL_DIR, f"scaler_k{k}.npz"))
        scalers.append((sc["f1_mean"], sc["f1_std"], sc["f2_mean"], sc["f2_std"]))

    # EVAL_FOLD에 해당하는 모델/scaler만 선택 (단일 fold inference)
    m_res_eval = residual_models[EVAL_FOLD - 1]
    m_rer_eval = reranking_models[EVAL_FOLD - 1]
    f1_mean, f1_std, f2_mean, f2_std = scalers[EVAL_FOLD - 1]
    print(f"    - Residual  MLP : fold {EVAL_FOLD} 사용 (학습 미포함 구간)")
    print(f"    - Re-ranking MLP: fold {EVAL_FOLD} 사용")
    print(f"    - Scaler        : fold {EVAL_FOLD} scaler 사용")

    # Base10M memmap
    base10m = open_fvecs_memmap(BASE10M_FILE)
    print(f"    - Base10M       : {base10m.shape}  (memmap)")

    # -------------------------------------------------------------------------
    # Step 4. 쿼리별 순차 처리
    # -------------------------------------------------------------------------
    print(f"\n>>> [4/6] 쿼리 {NUM_QUERY}개 순차 처리 중...")

    search_latencies  = []
    mlp_latencies     = []      # residual MLP + re-ranking MLP
    rerank_latencies  = []      # 실제 L2 re-ranking (수행 시에만)
    returned_ids      = []
    rerank_flags      = []      # 실제로 re-ranking 수행 여부
    t_mlp_list        = []      # 쿼리별 MLP 파이프라인 시간 (ms)
    t_io_list         = []      # 쿼리별 I/O 시간: 0=미수행, >0=실제 base10m 접근 시간 (ms)

    io_before = get_io_bytes()
    mem_before_search = get_rss_mb()

    for i in range(NUM_QUERY):
        q     = xq[i : i + 1]   # (1, 128)
        q_vec = xq[i]           # (128,)

        # ── HNSW Search → top-16 ────────────────────────────────────────────
        t_s = time.perf_counter()
        D_pq, I = hnsw_index.search(q, CANDIDATES)   # D: (1,16), I: (1,16)
        t_e = time.perf_counter()
        search_latencies.append((t_e - t_s) * 1000)

        cand_ids = I[0]           # (16,)
        pq_dists = D_pq[0]       # (16,) ||Q-P||²

        # ── MLP 파이프라인 ──────────────────────────────────────────────────
        t_m0 = time.perf_counter()

        # [A] Residual Feature 계산: (16 candidates, 16 subspaces, 9 dims)
        #     index에서 미리 추출한 codes를 바로 인덱싱 → base10m I/O 제로
        pq_codes_cand  = pq_codes_all[cand_ids]    # (16, 16)
        res_codes_cand = res_codes_all[cand_ids]   # (16, 16)

        feat_list = []   # 16 subspaces
        for m in range(NUM_SUBSPACES):
            s = m * SUB_DIM; e = (m + 1) * SUB_DIM
            Q_sub  = np.tile(q_vec[s:e], (CANDIDATES, 1))           # (16, 8)
            P_sub  = pq_centroids[m][pq_codes_cand[:, m]]           # (16, 8)
            diff_v = Q_sub - P_sub                                   # (16, 8)
            res_r  = res_centroids[m][res_codes_cand[:, m]]         # (16, 8)
            prod   = diff_v * res_r                                  # (16, 8)
            norm_r = np.sqrt(np.sum(res_r ** 2, axis=1, keepdims=True)) / np.sqrt(SUB_DIM)  # (16,1)
            feat_list.append(np.hstack([prod, norm_r]))             # (16, 9)

        # (CANDIDATES, NUM_SUBSPACES, 9) = (16, 16, 9)
        feat_tensor = torch.tensor(
            np.stack(feat_list, axis=1), dtype=torch.float32)

        # [B] EVAL_FOLD 모델로 pred_dot 계산: (16,)
        with torch.no_grad():
            pred_dot = m_res_eval(feat_tensor).numpy().flatten()   # (16,)

        # [C] residual_dist = ||X-P||² - 2 * pred_dot
        xp_norm_sq    = res_norm_sq_all[cand_ids]                # (16,)
        residual_dist = xp_norm_sq - 2.0 * pred_dot             # (16,)

        # [D] Re-ranking MLP feature: [pq_dists(16) || residual_dist(16)] = (32,)
        feat32 = np.hstack([pq_dists, residual_dist])           # (32,)

        # EVAL_FOLD scaler 적용
        feat32_f1 = (feat32[:16] - f1_mean) / (f1_std + 1e-8)
        feat32_f2 = (feat32[16:] - f2_mean) / (f2_std + 1e-8)
        feat32_scaled = np.hstack([feat32_f1, feat32_f2]).astype(np.float32)

        feat32_t = torch.tensor(feat32_scaled).unsqueeze(0)     # (1, 32)

        # [E] EVAL_FOLD Re-ranking MLP → binary prediction
        with torch.no_grad():
            prob = m_rer_eval(feat32_t).item()

        t_m1 = time.perf_counter()
        t_mlp_ms = (t_m1 - t_m0) * 1000
        mlp_latencies.append(t_mlp_ms)
        t_mlp_list.append(round(t_mlp_ms, 6))

        # ── Re-ranking 결정 ──────────────────────────────────────────────────
        if prob >= THRESHOLD:
            # 실제 L2 re-ranking
            t_r0 = time.perf_counter()
            cand_vecs = base10m[cand_ids]                          # (16, 128)
            dists     = np.sum((cand_vecs - q_vec) ** 2, axis=1)
            best      = int(cand_ids[np.argmin(dists)])
            t_r1 = time.perf_counter()
            t_io_ms = (t_r1 - t_r0) * 1000
            rerank_latencies.append(t_io_ms)
            t_io_list.append(round(t_io_ms, 6))
            rerank_flags.append(1)
        else:
            best = int(cand_ids[0])   # PQ top-1 그대로
            t_io_list.append(0.0)
            rerank_flags.append(0)

        returned_ids.append(best)

    io_after          = get_io_bytes()
    io_search_total   = io_after - io_before
    mem_after_search  = get_rss_mb()
    mem_peak          = get_peak_mb()

    search_latencies = np.array(search_latencies,  dtype=np.float64)
    mlp_latencies    = np.array(mlp_latencies,     dtype=np.float64)
    rerank_flags     = np.array(rerank_flags,      dtype=np.int32)
    returned_ids     = np.array(returned_ids,      dtype=np.int64)
    rerank_lat_arr   = np.array(rerank_latencies,  dtype=np.float64) if rerank_latencies else np.array([0.])

    n_reranked = int(rerank_flags.sum())
    print(f"    - Re-ranking 수행 쿼리 수 : {n_reranked:,} / {NUM_QUERY:,} ({n_reranked/NUM_QUERY*100:.1f}%)")
    print(f"    - PQ Search (avg)          : {search_latencies.mean():.4f} ms")
    print(f"    - MLP 파이프라인 (avg)     : {mlp_latencies.mean():.4f} ms")
    if len(rerank_latencies):
        print(f"    - L2 Re-ranking (avg)      : {rerank_lat_arr.mean():.4f} ms")

    # -------------------------------------------------------------------------
    # Step 5. 메트릭 계산
    # -------------------------------------------------------------------------
    print("\n>>> [5/6] 메트릭 계산 중...")

    recall_at_1 = float((returned_ids == gt_top1).mean())
    mrr_values  = []
    dr_values   = []

    for i in range(NUM_QUERY):
        ret_id = returned_ids[i]; gt_id = gt_top1[i]; q_vec = xq[i]
        gt_row   = gt[i]
        rank_pos = np.where(gt_row == ret_id)[0]
        rank     = int(rank_pos[0]) + 1 if len(rank_pos) > 0 else len(gt_row) + 1
        mrr_values.append(1.0 / rank)

        d_ret = float(np.sum((q_vec - base10m[ret_id]) ** 2))
        d_gt  = float(np.sum((q_vec - base10m[gt_id])  ** 2))
        dr    = (d_gt / d_ret) if d_ret > 1e-12 else 1.0
        dr_values.append(min(dr, 1.0))

    mrr_values = np.array(mrr_values, dtype=np.float64)
    dr_values  = np.array(dr_values,  dtype=np.float64)
    mrr        = float(mrr_values.mean())
    dr         = float(dr_values.mean())

    print(f"    - Recall@1        : {recall_at_1:.4f}")
    print(f"    - MRR             : {mrr:.4f}")
    print(f"    - Distance Ratio  : {dr:.4f}")

    mrr_hist = histogram_counts(mrr_values, BINS)
    dr_hist  = histogram_counts(dr_values,  BINS)

    # -------------------------------------------------------------------------
    # Step 6. JSON 저장
    # -------------------------------------------------------------------------
    total_end = time.perf_counter()
    total_ms  = (total_end - total_start) * 1000

    results = {
        "name": NAME,
        "creation_date": CREATION_DATE,
        "eval_fold": EVAL_FOLD,
        "query_range": [Q_START, Q_END - 1],
        "num_query": NUM_QUERY,
        "candidates": CANDIDATES,
        "reranked_queries": n_reranked,
        "reranked_ratio": round(n_reranked / NUM_QUERY, 4),
        "latency": {
            "total_ms":                    round(total_ms, 3),
            "query_search_total_ms":       round(float(search_latencies.sum()), 3),
            "query_search_avg_ms":         round(float(search_latencies.mean()), 6),
            "mlp_pipeline_total_ms":       round(float(mlp_latencies.sum()), 3),
            "mlp_pipeline_avg_ms":         round(float(mlp_latencies.mean()), 6),
            "re_ranking_total_ms":         round(float(rerank_lat_arr.sum()), 3),
            "re_ranking_avg_ms":           round(float(rerank_lat_arr.sum()) / NUM_QUERY, 6),
        },
        "disk_io": {
            "index_load_bytes":            io_index_bytes,
            "search_and_mlp_bytes":        io_search_total,
            "total_io_bytes":              io_index_bytes + io_search_total,
        },
        "memory_mb": {
            "before_index_load":           round(mem_before_index, 2),
            "after_index_load":            round(mem_after_index, 2),
            "index_load_delta":            round(mem_after_index - mem_before_index, 2),
            "after_search":                round(mem_after_search, 2),
            "search_delta":                round(mem_after_search - mem_before_search, 2),
            "peak":                        round(mem_peak, 2),
        },
        "metrics": {
            "recall_at_1":                 round(recall_at_1, 6),
            "mrr":                         round(mrr, 6),
            "distance_ratio":              round(dr, 6),
        },
        "histogram": {
            "mrr":            mrr_hist,
            "distance_ratio": dr_hist,
        }
    }

    os.makedirs(METRIC_DIR, exist_ok=True)
    with open(METRIC_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 쿼리별 T_mlp / T_I/O 저장 (평균 없이 쿼리 순서 그대로)
    timings = {
        "name": NAME,
        "creation_date": CREATION_DATE,
        "eval_fold": EVAL_FOLD,
        "query_range": [Q_START, Q_END - 1],
        "num_query": NUM_QUERY,
        "description": {
            "T_mlp": "Residual MLP + Re-ranking MLP 파이프라인 시간 (ms), 쿼리당 1개",
            "T_IO":  "실제 base10m I/O + L2 계산 시간 (ms); Re-ranking MLP가 0 예측 시 0.0"
        },
        "per_query": [
            {"query_idx": Q_START + i, "T_mlp": t_mlp_list[i], "T_IO": t_io_list[i]}
            for i in range(NUM_QUERY)
        ]
    }
    with open(TIMINGS_PATH, "w") as f:
        json.dump(timings, f, indent=2, ensure_ascii=False)

    print(f"\n    ✓ Saved: {METRIC_PATH}")
    print(f"    ✓ Saved: {TIMINGS_PATH}")

    print("\n" + "=" * 70)
    print("[결과 요약]")
    print(f"  Total Latency              : {total_ms:.1f} ms")
    print(f"  PQ Search (avg)            : {float(search_latencies.mean()):.4f} ms")
    print(f"  MLP 파이프라인 (avg)       : {float(mlp_latencies.mean()):.4f} ms")
    print(f"  L2 Re-ranking (avg)        : {rerank_lat_arr.mean():.4f} ms")
    print(f"  Re-ranking 수행 비율       : {n_reranked}/{NUM_QUERY} ({n_reranked/NUM_QUERY*100:.1f}%)")
    print(f"  I/O - Index Load           : {io_index_bytes:,} bytes")
    print(f"  I/O - Search+MLP+Rerank    : {io_search_total:,} bytes")
    print(f"  Mem - Index Load Delta     : {mem_after_index - mem_before_index:.1f} MB")
    print(f"  Recall@1                   : {recall_at_1:.4f}")
    print(f"  MRR                        : {mrr:.4f}")
    print(f"  Distance Ratio             : {dr:.4f}")
    print("=" * 70)
