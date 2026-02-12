#!/usr/bin/env python3
"""
Create Residual PQ Index
- Load pq.index and bigann_base10M.fvecs
- Calculate residual vectors (original - PQ reconstructed)
- Train and build Residual PQ index
"""

import faiss
import numpy as np
import os
import time

# ===== Configuration =====
class Config:
    # Input paths
    INDEX_FILE = "/home/syback/vectorDB/on-device/data/index/pq.index"
    BASE_FILE = "/home/syback/vectorDB/ann_datasets/sift1B/bigann_base10M.fvecs"
    
    # Output paths
    OUTPUT_DIR = "/home/syback/vectorDB/on-device/data/index"
    DATA_DIR = "/home/syback/vectorDB/on-device/data"
    RESIDUAL_INDEX_FILE = os.path.join(OUTPUT_DIR, "residual_pq.index")
    RESIDUAL_NORMSQ_FILE = os.path.join(DATA_DIR, "residual_normsq.npy")
    
    # Index parameters (same as base PQ)
    DIM = 128
    PQ_M = 16        # Number of subquantizers
    PQ_NBITS = 8     # Bits per subquantizer
    
    # Data sizes
    NUM_BASE = 10_000_000   # 10M vectors


# ===== Data Loading Functions =====
def read_bvecs(filename, num_vectors, dim=128):
    """Read bvecs file (uint8 format)"""
    print(f"Reading {num_vectors:,} vectors from {filename}...")
    
    record_size = 4 + dim
    mm = np.memmap(filename, dtype='uint8', mode='r')
    
    bytes_to_read = num_vectors * record_size
    mm = mm[:bytes_to_read]
    
    data = mm.reshape(num_vectors, record_size)[:, 4:]
    
    print(f"  Loaded shape: {data.shape}, dtype: {data.dtype}")
    return data


def read_fvecs(filename, num_vectors=None):
    """Read fvecs file (float32 format)"""
    print(f"Reading vectors from {filename}...")
    
    with open(filename, "rb") as f:
        d = np.frombuffer(f.read(4), dtype='int32')[0]
    
    filesize = os.path.getsize(filename)
    n = filesize // ((d + 1) * 4)
    
    if num_vectors is not None:
        n = min(n, num_vectors)
    
    data = np.memmap(filename, dtype='float32', mode='r')
    data = data.reshape(-1, d + 1)[:n, 1:]
    
    print(f"  Loaded shape: {data.shape}, dtype: {data.dtype}")
    return data


# ===== Residual Calculation =====
def calculate_residuals(base_vectors, pq_index):
    """Calculate residual vectors (original - PQ reconstructed)"""
    print(f"\nCalculating residual vectors...")
    print(f"  Base vectors shape: {base_vectors.shape}")
    
    # Convert to float32 if needed
    if base_vectors.dtype != np.float32:
        base_vectors = base_vectors.astype('float32')
    
    # Get PQ codes
    pq = pq_index.pq
    codes = pq.compute_codes(base_vectors)
    codes_np = np.frombuffer(codes, dtype=np.uint8)
    codes_np = codes_np.reshape(base_vectors.shape[0], pq.code_size)
    
    print(f"  PQ codes shape: {codes_np.shape}")
    
    # Reconstruct vectors from PQ codes
    pq_reconstructed = pq.decode(codes_np)
    print(f"  PQ reconstructed shape: {pq_reconstructed.shape}")
    
    # Calculate residuals
    residuals = base_vectors - pq_reconstructed
    print(f"  Residuals shape: {residuals.shape}")
    
    # Statistics
    residual_norms = np.linalg.norm(residuals, axis=1)
    print(f"\n  Residual statistics:")
    print(f"    Mean norm: {residual_norms.mean():.4f}")
    print(f"    Std norm:  {residual_norms.std():.4f}")
    print(f"    Min norm:  {residual_norms.min():.4f}")
    print(f"    Max norm:  {residual_norms.max():.4f}")
    
    return residuals


# ===== Index Creation Functions =====
def create_residual_pq_index(dim, m, nbits):
    """Create Residual PQ index"""
    print(f"\nCreating Residual PQ index:")
    print(f"  Dimension: {dim}")
    print(f"  M (subquantizers): {m}")
    print(f"  nbits: {nbits}")
    
    index = faiss.IndexPQ(dim, m, nbits)
    return index


def train_index(index, train_residuals):
    """Train the residual index"""
    print(f"\nTraining Residual PQ index with {train_residuals.shape[0]:,} vectors...")
    
    if train_residuals.dtype != np.float32:
        train_residuals = train_residuals.astype('float32')
    
    start_time = time.time()
    index.train(train_residuals)
    elapsed = time.time() - start_time
    
    print(f"  Training completed in {elapsed:.2f} seconds")
    print(f"  Index is_trained: {index.is_trained}")
    
    return index


def add_vectors(index, residuals, batch_size=100_000):
    """Add residual vectors to the index in batches"""
    print(f"\nAdding {residuals.shape[0]:,} residual vectors to index...")
    
    if residuals.dtype != np.float32:
        residuals = residuals.astype('float32')
    
    start_time = time.time()
    
    for i in range(0, residuals.shape[0], batch_size):
        batch = residuals[i:i+batch_size]
        index.add(batch)
        
        if (i + batch_size) % 1_000_000 == 0 or i + batch_size >= residuals.shape[0]:
            print(f"  Added {min(i + batch_size, residuals.shape[0]):,} / {residuals.shape[0]:,} vectors")
    
    elapsed = time.time() - start_time
    
    print(f"  Adding completed in {elapsed:.2f} seconds")
    print(f"  Total vectors in index: {index.ntotal:,}")
    
    return index


def calculate_and_save_residual_normsq(residuals, filepath):
    """
    Calculate and save residual L2 squared norms
    
    Args:
        residuals: (N, D) residual vectors
        filepath: output file path
    
    Returns:
        residual_normsq: (N,) L2 squared norms
    """
    print(f"\nCalculating residual L2 squared norms...")
    
    # Calculate L2 squared norm for each vector
    # ||residual||^2 = sum(residual^2, axis=1)
    residual_normsq = np.sum(residuals ** 2, axis=1).astype(np.float32)
    
    print(f"  Shape: {residual_normsq.shape}")
    print(f"  Statistics:")
    print(f"    Mean: {residual_normsq.mean():.4f}")
    print(f"    Std:  {residual_normsq.std():.4f}")
    print(f"    Min:  {residual_normsq.min():.4f}")
    print(f"    Max:  {residual_normsq.max():.4f}")
    
    # Save to file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, residual_normsq)
    
    file_size = os.path.getsize(filepath)
    print(f"\n  Saved to: {filepath}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024**2:.2f} MB)")
    
    return residual_normsq


def save_index(index, filepath):
    """Save the index to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    print(f"\nSaving Residual PQ index to {filepath}...")
    faiss.write_index(index, filepath)
    
    file_size = os.path.getsize(filepath)
    print(f"  Index saved successfully ({file_size:,} bytes, {file_size / 1024**2:.2f} MB)")


# ===== Main Function =====
def main():
    print("=" * 70)
    print("Residual PQ Index Creation")
    print("=" * 70)
    
    # 1. Load base PQ index
    print("\n[Step 1] Loading base PQ index...")
    pq_index = faiss.read_index(Config.INDEX_FILE)
    print(f"  Base PQ index loaded: {pq_index.ntotal:,} vectors")
    
    # 2. Load base vectors
    print("\n[Step 2] Loading base vectors...")
    base_data = read_fvecs(Config.BASE_FILE, Config.NUM_BASE)
    
    # 3. Calculate base residuals
    print("\n[Step 3] Calculating residuals...")
    base_residuals = calculate_residuals(base_data, pq_index)
    
    # 4. Calculate and save residual norm squared
    print("\n[Step 4] Calculating and saving residual L2 squared norms...")
    residual_normsq = calculate_and_save_residual_normsq(base_residuals, Config.RESIDUAL_NORMSQ_FILE)
    
    # 5. Create Residual PQ index
    print("\n[Step 5] Creating Residual PQ index...")
    residual_index = create_residual_pq_index(Config.DIM, Config.PQ_M, Config.PQ_NBITS)
    
    # 6. Train index on base residuals (same as 15_create_residual_features_resD_pq.py)
    print("\n[Step 6] Training Residual PQ index on base residuals...")
    residual_index = train_index(residual_index, base_residuals)
    
    # 7. Add base residuals
    print("\n[Step 7] Adding base residuals to index...")
    residual_index = add_vectors(residual_index, base_residuals)
    
    # 8. Save index
    print("\n[Step 8] Saving Residual PQ index...")
    save_index(residual_index, Config.RESIDUAL_INDEX_FILE)
    
    # 9. Summary
    print("\n" + "=" * 70)
    print("Residual Data Creation Completed!")
    print("=" * 70)
    print(f"Base PQ index:         {Config.INDEX_FILE}")
    print(f"Residual PQ index:     {Config.RESIDUAL_INDEX_FILE}")
    print(f"Residual NormSq file:  {Config.RESIDUAL_NORMSQ_FILE}")
    print(f"Total vectors:         {residual_index.ntotal:,}")
    print(f"Dimension:             {Config.DIM}")
    print(f"PQ parameters:         M={Config.PQ_M}, nbits={Config.PQ_NBITS}")
    print("=" * 70)
    print("\n✅ This matches the approach in 15_create_residual_features_resD_pq.py")
    print("   - Train on full base residuals (10M vectors)")
    print("   - Add same residuals to index")


if __name__ == "__main__":
    main()
