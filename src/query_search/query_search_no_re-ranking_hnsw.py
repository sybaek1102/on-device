#!/usr/bin/env python3
"""
query_search_no_re-ranking_hnsw.py
───────────────────────────────────
Re-ranking 없이 HNSWPQ Index 검색 결과의 top-1을 바로 반환하고 성능을 평가합니다.
쿼리를 하나씩 순차 처리합니다 (on-device 환경 가정).

[측정 항목]
  - total_latency_ms
  - query_search_latency_ms   (전체 / 평균)
  - re_ranking_latency_ms     (0 — re-ranking 없음)
  - disk_io_bytes             (index 로딩 I/O / 그 외 I/O)
  - recall@1
  - MRR
  - Distance Ratio
  - MRR / Distance Ratio 구간별 히스토그램 (0.1 단위)

[저장]
  output/metric/{name}_query_search.json
"""

import faiss
faiss.omp_set_num_threads(1)   # on-device: 단일 스레드 가정

import numpy as np
import os
import time
import json
import resource

# =============================================================================
# 🔹 Configuration
# =============================================================================
CREATION_DATE = "2026022007"
NAME          = "no_hnsw"      # JSON 파일명 prefix

# =============================================================================
# 🔹 K-Fold 평가 설정
#    EVAL_FOLD : 1~10 중 하나를 지정하면 해당 fold의 쿼리 구간(1,000개)만 실행
#                → selective / all 스크립트와 동일 쿼리셋으로 공평한 비교 가능
# =============================================================================
EVAL_FOLD  = 1           # 1~10 (1-indexed)
NUM_FOLDS  = 10

DATA_DIR   = "/home/syback/vectorDB/ann_datasets/sift1B"
QUERY_FILE = os.path.join(DATA_DIR, "bigann_query.bvecs")
GT_FILE    = os.path.join(DATA_DIR, "gnd", "idx_10M.ivecs")

BASE10M_FILE = "/home/syback/vectorDB/on-device/data/raw/bigann_base10M.fvecs"

INDEX_DIR     = "/home/syback/vectorDB/on-device/data/index"
HNSW_EF_SEARCH = 64           # HNSW 검색 품질 (Ŀ을수록 Recall↑, 속도↓)
PQ_INDEX_PATH = os.path.join(INDEX_DIR, f"{CREATION_DATE}_hnswpq.index")

METRIC_DIR  = "/home/syback/vectorDB/on-device/output/metric"
METRIC_PATH = os.path.join(METRIC_DIR, f"{NAME}_query_search_fold{EVAL_FOLD}.json")

NUM_QUERY_TOTAL  = 10_000
QUERIES_PER_FOLD = NUM_QUERY_TOTAL // NUM_FOLDS   # 1,000
Q_START   = (EVAL_FOLD - 1) * QUERIES_PER_FOLD    # inclusive
Q_END     = EVAL_FOLD * QUERIES_PER_FOLD           # exclusive
NUM_QUERY = QUERIES_PER_FOLD                       # 이번 실행에서 처리할 쿼리 수

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
    mm          = np.memmap(fname, dtype="int32", mode="r")
    k           = mm[0]
    record_size = k + 1
    nvecs       = mm.shape[0] // record_size
    return mm.reshape(nvecs, record_size)[:, 1:].copy()

def open_fvecs_memmap(fname):
    """fvecs 파일을 memmap으로 열어 (N, D) float32 배열체럼 접근"""
    with open(fname, "rb") as f:
        d = np.frombuffer(f.read(4), dtype="int32")[0]
    total = os.path.getsize(fname) // ((1 + d) * 4)
    raw   = np.memmap(fname, dtype="float32", mode="r").reshape(total, 1 + d)
    return raw[:, 1:]

def get_io_bytes():
    """현재 프로세스의 누적 read I/O bytes (Linux /proc 기반)"""
    try:
        with open(f"/proc/{os.getpid()}/io", "r") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0

def get_rss_mb():
    """현재 프로세스의 RSS(Resident Set Size) 메모리 반환 (MB)"""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024   # KB → MB
    except Exception:
        pass
    return 0.0

def get_peak_mb():
    """현재 프로세스의 Peak RSS 메모리 반환 (MB)"""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmPeak:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0

def histogram_counts(values, bins):
    """values 배열에서 각 구간 (0, bins[i]] 에 속하는 개수 반환"""
    result = {}
    for b in bins:
        key = f"<={b:.1f}"
        result[key] = int(np.sum(values <= b))
    return result

# =============================================================================
# 🔹 Main
# =============================================================================
if __name__ == "__main__":
    BINS = [round(0.1 * i, 1) for i in range(1, 11)]   # 0.1 ~ 1.0

    print("=" * 70)
    print(f"  Query Search — No Re-ranking  (name='{NAME}')")
    print("=" * 70)

    total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Step 1. 데이터 로드 (bvecs, GT)
    # -------------------------------------------------------------------------
    print("\n>>> [1/4] 데이터 로드 (bvecs, GT)")
    print(f"    - EVAL_FOLD : {EVAL_FOLD}  (쿼리 {Q_START}~{Q_END-1}, {NUM_QUERY}개)")
    xq_all  = load_bvecs(QUERY_FILE, Q_END)
    xq      = xq_all[Q_START:]          # (NUM_QUERY, 128)
    gt_all  = load_ivecs(GT_FILE)
    gt      = gt_all[Q_START:Q_END]     # (NUM_QUERY, k_gt)
    gt_top1 = gt[:, 0]
    del xq_all, gt_all
    print(f"    - Query : {xq.shape}")
    print(f"    - GT    : {gt.shape}")

    # -------------------------------------------------------------------------
    # Step 2. OS 페이지 캐시 비우기 + Index 로드 — I/O & 메모리 측정
    # -------------------------------------------------------------------------
    print("\n>>> [2/4] OS 페이지 캐시 비우기...")
    import subprocess
    try:
        result = subprocess.run(
            ["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("    ✓ 페이지 캐시 drop 완료")
        else:
            print(f"    ⚠ 캐시 drop 실패 (sudo 권한 필요): {result.stderr.strip()}")
    except Exception as e:
        print(f"    ⚠ 캐시 drop 건너뜀: {e}")

    print("\n    PQ Index 로드 중...")
    io_before_index  = get_io_bytes()
    mem_before_index = get_rss_mb()
    index_load_start = time.perf_counter()

    index = faiss.read_index(PQ_INDEX_PATH)
    index.hnsw.efSearch = HNSW_EF_SEARCH   # 검색 품질 설정

    index_load_end  = time.perf_counter()
    io_after_index  = get_io_bytes()
    mem_after_index = get_rss_mb()

    index_load_ms  = (index_load_end - index_load_start) * 1000
    io_index_bytes = io_after_index - io_before_index
    print(f"    - Index ntotal    : {index.ntotal:,}")
    print(f"    - Load time       : {index_load_ms:.1f} ms")
    print(f"    - I/O (index)     : {io_index_bytes:,} bytes")
    print(f"    - efSearch        : {HNSW_EF_SEARCH}")
    print(f"    - Mem before load : {mem_before_index:.1f} MB")
    print(f"    - Mem after load  : {mem_after_index:.1f} MB  (+{mem_after_index - mem_before_index:.1f} MB)")

    # Base10M memmap (distance ratio 계산와 no에서는 xb 대신 사용)
    base10m = open_fvecs_memmap(BASE10M_FILE)
    print(f"    - Base10M         : {base10m.shape}  (memmap)")

    # -------------------------------------------------------------------------
    # Step 3. 쿼리별 순차 검색
    # -------------------------------------------------------------------------
    print(f"\n>>> [3/4] 쿼리 {NUM_QUERY}개 순차 검색 중...")

    search_latencies = []       # query당 검색 latency (ms)
    returned_ids     = []       # top-1 반환 index

    io_before_search = get_io_bytes()
    mem_before_search = get_rss_mb()

    for i in range(NUM_QUERY):
        q = xq[i : i + 1]          # (1, 128)
        t_s = time.perf_counter()
        _, I = index.search(q, 1)   # top-1
        t_e = time.perf_counter()

        search_latencies.append((t_e - t_s) * 1000)
        returned_ids.append(int(I[0, 0]))

    io_after_search  = get_io_bytes()
    io_search_bytes  = io_after_search - io_before_search
    mem_after_search = get_rss_mb()
    mem_peak         = get_peak_mb()

    search_latencies = np.array(search_latencies, dtype=np.float64)
    returned_ids     = np.array(returned_ids, dtype=np.int64)

    print(f"    - 평균 검색 latency : {search_latencies.mean():.4f} ms")
    print(f"    - 총  검색 latency  : {search_latencies.sum():.2f} ms")
    print(f"    - I/O (search)      : {io_search_bytes:,} bytes")

    # -------------------------------------------------------------------------
    # Step 4. 메트릭 계산
    # -------------------------------------------------------------------------
    print("\n>>> [4/4] 메트릭 계산 중...")

    # --- Recall@1 ---
    recall_arr = (returned_ids == gt_top1).astype(np.float32)
    recall_at_1 = float(recall_arr.mean())

    # --- MRR & Distance Ratio ---
    mrr_values  = []
    dr_values   = []

    # GT top1 벡터의 실제 거리 (xq와 xb[gt_top1] 의 L2)
    # Distance Ratio = d(q, gt_top1) / d(q, returned)
    for i in range(NUM_QUERY):
        ret_id = returned_ids[i]
        gt_id  = gt_top1[i]
        q_vec  = xq[i]

        # MRR: GT 배열에서 반환 벡터 순위
        gt_row   = gt[i]
        rank_pos = np.where(gt_row == ret_id)[0]
        if len(rank_pos) == 0:
            rank = len(gt_row) + 1
        else:
            rank = int(rank_pos[0]) + 1

        mrr_values.append(1.0 / rank)

        # 실제 L2 거리 계산
        d_ret = float(np.sum((q_vec - base10m[ret_id]) ** 2))
        d_gt  = float(np.sum((q_vec - base10m[gt_id])  ** 2))

        if d_ret < 1e-12:
            dr = 1.0
        else:
            dr = d_gt / d_ret

        dr_values.append(min(dr, 1.0))

    mrr_values = np.array(mrr_values, dtype=np.float64)
    dr_values  = np.array(dr_values,  dtype=np.float64)

    mrr = float(mrr_values.mean())
    dr  = float(dr_values.mean())

    print(f"    - Recall@1        : {recall_at_1:.4f}")
    print(f"    - MRR             : {mrr:.4f}")
    print(f"    - Distance Ratio  : {dr:.4f}")

    # 히스토그램
    mrr_hist = histogram_counts(mrr_values, BINS)
    dr_hist  = histogram_counts(dr_values,  BINS)

    # -------------------------------------------------------------------------
    # 최종 정리
    # -------------------------------------------------------------------------
    total_end    = time.perf_counter()
    total_ms     = (total_end - total_start) * 1000
    search_total = float(search_latencies.sum())

    results = {
        "name": NAME,
        "creation_date": CREATION_DATE,
        "eval_fold": EVAL_FOLD,
        "query_range": [Q_START, Q_END - 1],
        "num_query": NUM_QUERY,
        "latency": {
            "total_ms":                   round(total_ms, 3),
            "query_search_total_ms":      round(search_total, 3),
            "query_search_avg_ms":        round(float(search_latencies.mean()), 6),
            "query_search_min_ms":        round(float(search_latencies.min()), 6),
            "query_search_max_ms":        round(float(search_latencies.max()), 6),
            "re_ranking_total_ms":        0,
            "re_ranking_avg_ms":          0,
        },
        "disk_io": {
            "index_load_bytes":           io_index_bytes,
            "search_io_bytes":            io_search_bytes,
            "total_io_bytes":             io_index_bytes + io_search_bytes,
        },
        "memory_mb": {
            "before_index_load":          round(mem_before_index, 2),
            "after_index_load":           round(mem_after_index, 2),
            "index_load_delta":           round(mem_after_index - mem_before_index, 2),
            "after_search":               round(mem_after_search, 2),
            "search_delta":               round(mem_after_search - mem_before_search, 2),
            "peak":                       round(mem_peak, 2),
        },
        "metrics": {
            "recall_at_1":                round(recall_at_1, 6),
            "mrr":                        round(mrr, 6),
            "distance_ratio":             round(dr, 6),
        },
        "histogram": {
            "mrr": mrr_hist,
            "distance_ratio": dr_hist,
        }
    }

    os.makedirs(METRIC_DIR, exist_ok=True)
    with open(METRIC_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n    ✓ Saved: {METRIC_PATH}")

    print("\n" + "=" * 70)
    print("[결과 요약]")
    print(f"  Total Latency         : {total_ms:.1f} ms")
    print(f"  Query Search (avg)    : {float(search_latencies.mean()):.4f} ms")
    print(f"  Re-ranking            : 0 ms  (없음)")
    print(f"  I/O - Index Load      : {io_index_bytes:,} bytes")
    print(f"  I/O - Search          : {io_search_bytes:,} bytes")
    print(f"  Recall@1              : {recall_at_1:.4f}")
    print(f"  MRR                   : {mrr:.4f}")
    print(f"  Distance Ratio        : {dr:.4f}")
    print("=" * 70)
