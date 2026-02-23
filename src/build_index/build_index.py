#!/usr/bin/env python3
import faiss
faiss.omp_set_num_threads(8)  # core 수 제한

import numpy as np
import os
import time
import json
from datetime import datetime
from utils.print_log import info, warn, error

# =========================================================
# 🔹 Configuration
# =========================================================
DATA_DIR = "/home/syback/vectorDB/ann_datasets/sift1B"
BASE_FILE = os.path.join(DATA_DIR, "bigann_base.bvecs")
QUERY_FILE = os.path.join(DATA_DIR, "bigann_query.bvecs")
LEARN_FILE = os.path.join(DATA_DIR, "bigann_learn.bvecs")

NUM_BASE = 10_000_000  # 10M base vectors
NUM_LEARN = 1_000_000  # 1M learn vectors

creation_date = datetime.now().strftime("%Y%m%d%H")
INDEX_SAVE_PATH = f"/home/syback/vectorDB/on-device/data/index/{creation_date}_pq.index"
RESIDUAL_INDEX_SAVE_PATH = f"/home/syback/vectorDB/on-device/data/index/{creation_date}_residual_pq.index"
BASE10M_SAVE_PATH = f"/home/syback/vectorDB/on-device/data/raw/{creation_date}_base10M.fvecs"
RESIDUAL_NORM_SQ_PATH = f"/home/syback/vectorDB/on-device/data/features/{creation_date}_residual_norm_sq.npz"
METRIC_SAVE_PATH = f"/home/syback/vectorDB/on-device/output/metric/{creation_date}_build_time.json"

DIM = 128
PQ_M = 16
PQ_NBITS = 8

# =========================================================
# 🔹 Helper functions
# =========================================================
def load_bvecs(fname, num_vectors=None):
    with open(fname, "rb") as f:
        d = np.frombuffer(f.read(4), dtype='int32')[0]
    
    filesize = os.path.getsize(fname)
    record_size = 4 + d  # 4 bytes for dimension + d bytes for data
    total_vectors = filesize // record_size
    
    if num_vectors is not None:
        num_vectors = min(num_vectors, total_vectors)
    else:
        num_vectors = total_vectors
    
    mm = np.memmap(fname, dtype='uint8', mode='r')
    mm = mm[:num_vectors * record_size]
    
    data = mm.reshape(num_vectors, record_size)[:, 4:]
    
    return data.astype('float32')

def save_fvecs(fname, data):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    
    n, d = data.shape
    
    with open(fname, 'wb') as f:
        for i in range(n):
            f.write(np.array([d], dtype='int32').tobytes())
            f.write(data[i].astype('float32').tobytes())
    
    info(f"  - Saved {n} vectors to: {fname}")

# =========================================================
# 🔹 Main
# =========================================================
if __name__ == "__main__":
    
    # 1. Data Load
    info("1. Data Load")
    xb = load_bvecs(BASE_FILE, NUM_BASE)  # base vectors (10M)
    xq = load_bvecs(QUERY_FILE)  # query vectors (all)
    xt = load_bvecs(LEARN_FILE, NUM_LEARN)  # learn/train vectors (1M)
    
    info(f"  - Base vectors: {xb.shape}")
    info(f"  - Query vectors: {xq.shape}")
    info(f"  - Learn vectors: {xt.shape}")
    
    save_fvecs(BASE10M_SAVE_PATH, xb)
    
    # 2. Train Index & Add Index
    info("2. Train Index & Add Index")
    index = faiss.IndexPQ(DIM, PQ_M, PQ_NBITS)
    info(f"  - Index type: PQ (M={PQ_M}, nbits={PQ_NBITS})")
    
    train_start = time.perf_counter_ns()
    index.train(xt)
    train_end = time.perf_counter_ns()
    train_time_ms = (train_end - train_start) / 1e6
    info(f"  - Training done in {train_time_ms:.3f}ms")
    
    add_start = time.perf_counter_ns()
    index.add(xb)
    add_end = time.perf_counter_ns()
    add_time_ms = (add_end - add_start) / 1e6
    info(f"  - Adding done in {add_time_ms:.3f}ms")
    info(f"  - Total vectors in index: {index.ntotal:,}")
    
    # 3. Save Index
    info("3. Save Index")
    os.makedirs(os.path.dirname(INDEX_SAVE_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_SAVE_PATH)
    info(f"  - Index saved to: {INDEX_SAVE_PATH}")
    
    # 4. Build Residual PQ Index
    info("4. Build Residual PQ Index")
    pq_reconstructed = index.reconstruct_n(0, index.ntotal)  # Get PQ reconstructed vectors
    residual_vectors = xb - pq_reconstructed
    info(f"  - Residual vectors shape: {residual_vectors.shape}")
    
    residual_index = faiss.IndexPQ(DIM, PQ_M, PQ_NBITS)
    info(f"  - Residual Index type: PQ (M={PQ_M}, nbits={PQ_NBITS})")
    
    res_train_start = time.perf_counter_ns()
    residual_index.train(residual_vectors)
    res_train_end = time.perf_counter_ns()
    res_train_time_ms = (res_train_end - res_train_start) / 1e6
    info(f"  - Residual training done in {res_train_time_ms:.3f}ms")
    
    res_add_start = time.perf_counter_ns()
    residual_index.add(residual_vectors)
    res_add_end = time.perf_counter_ns()
    res_add_time_ms = (res_add_end - res_add_start) / 1e6
    info(f"  - Residual adding done in {res_add_time_ms:.3f}ms")
    info(f"  - Total residual vectors in index: {residual_index.ntotal:,}")
    
    # 5. Save Residual Index
    info(f"5. Save Residual Index")
    os.makedirs(os.path.dirname(RESIDUAL_INDEX_SAVE_PATH), exist_ok=True)
    faiss.write_index(residual_index, RESIDUAL_INDEX_SAVE_PATH)
    info(f"  - Residual index saved to: {RESIDUAL_INDEX_SAVE_PATH}")
    
    # 6. Calculate and Save Residual Norm Squared
    info("6. Calculate and Save Residual Norm Squared")
    residual_norm_sq = np.sum(residual_vectors ** 2, axis=1)  # (10M,)
    info(f"  - Residual norm squared shape: {residual_norm_sq.shape}")
    
    os.makedirs(os.path.dirname(RESIDUAL_NORM_SQ_PATH), exist_ok=True)
    np.savez_compressed(RESIDUAL_NORM_SQ_PATH, residual_norm_sq=residual_norm_sq)
    info(f"  - Residual norm squared saved to: {RESIDUAL_NORM_SQ_PATH}")
    
    # 7. Save Timing Metrics
    info("7. Save Timing Metrics")
    metrics = {
        "version": creation_date,
        "base_pq": {
            "train_time_ms": round(train_time_ms, 3),
            "add_time_ms": round(add_time_ms, 3),
            "total_time_ms": round(train_time_ms + add_time_ms, 3)
        },
        "residual_pq": {
            "train_time_ms": round(res_train_time_ms, 3),
            "add_time_ms": round(res_add_time_ms, 3),
            "total_time_ms": round(res_train_time_ms + res_add_time_ms, 3)
        },
        "total_build_time_ms": round(train_time_ms + add_time_ms + res_train_time_ms + res_add_time_ms, 3)
    }
    
    os.makedirs(os.path.dirname(METRIC_SAVE_PATH), exist_ok=True)
    with open(METRIC_SAVE_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    info(f"  - Timing metrics saved to: {METRIC_SAVE_PATH}")
