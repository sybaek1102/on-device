#!/usr/bin/env python3
"""
model_train_residual.py
────────────────────────
create_residual_model_train_data.py 로 생성된 feature / label 로
10-Fold OOF MLP 모델을 학습합니다.

[입력]
  residual_features.npz : shape (160000, 16, 9)
  residual_label.npz    : shape (160000, 1)

[출력]
  model_k1.pt ~ model_k10.pt : 각 Fold best 모델
  train_log.csv               : epoch별 학습 로그
  oof_pred.npz                : 전체 OOF 예측값
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys

# =====================================================================
# 🔹 Configuration
# =====================================================================
FEATURE_PATH = "/home/syback/vectorDB/on-device/data/model/residual/residual_features.npz"
LABEL_PATH   = "/home/syback/vectorDB/on-device/data/model/residual/residual_label.npz"

MODEL_SAVE_DIR = "/home/syback/vectorDB/on-device/data/model/residual"
LOG_PATH       = os.path.join(MODEL_SAVE_DIR, "train_log.csv")
OOF_PATH       = os.path.join(MODEL_SAVE_DIR, "oof_pred.npz")

# 하이퍼파라미터
BATCH_SIZE    = 4096
LEARNING_RATE = 0.001
EPOCHS        = 100
NUM_FOLDS     = 10   # 10-Fold OOF

# 모델 구조
FEATURE_DIM   = 9    # 각 subspace feature 차원 (9 dims)
SHARED_HIDDEN = 32   # Shared MLP 중간 차원
EMBED_DIM     = 8    # Shared MLP 출력 차원
GLOBAL_HIDDEN = 64   # Global MLP 중간 차원

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Device: {DEVICE}")
print(f"📁 {NUM_FOLDS}-Fold OOF Training")
print(f"📊 Feature Dimension: (N, 16, {FEATURE_DIM})")

# =====================================================================
# 1. 데이터 로드 & 전처리
# =====================================================================
print("\n" + "="*70)
print("1️⃣  데이터 로드 & 전처리")
print("="*70)

if not os.path.exists(FEATURE_PATH) or not os.path.exists(LABEL_PATH):
    print(f"❌ Error: 파일을 찾을 수 없습니다.")
    print(f"   Feature: {FEATURE_PATH}")
    print(f"   Label:   {LABEL_PATH}")
    sys.exit(1)

X_np = np.load(FEATURE_PATH)["data"].astype(np.float32)  # (160000, 16, 9)
y_np = np.load(LABEL_PATH)["data"].astype(np.float32)    # (160000, 1)

print(f"✓ Feature Shape: {X_np.shape}")
print(f"✓ Label Shape:   {y_np.shape}")

# Global label (이미 16 subspace 합산된 값)
y_global = y_np  # (160000, 1)

print(f"\n📊 Label Statistics:")
print(f"   Mean: {y_global.mean():.4f}")
print(f"   Std:  {y_global.std():.4f}")
print(f"   Min:  {y_global.min():.4f}")
print(f"   Max:  {y_global.max():.4f}")

# Target 정규화를 위한 전역 통계 (전체 데이터 기준)
y_global_mean = float(y_global.mean())
y_global_std  = float(y_global.std())

print(f"\n✓ Normalization Stats: mean={y_global_mean:.4f}, std={y_global_std:.4f}")

num_samples = len(X_np)
oof_preds   = np.zeros((num_samples, 1), dtype=np.float32)

# =====================================================================
# 2. Fold Split 생성
# =====================================================================
print("\n" + "="*70)
print("2️⃣  10-Fold Split 생성")
print("="*70)

all_indices = np.arange(num_samples)
fold_chunks = np.array_split(all_indices, NUM_FOLDS)

print(f"✓ Total Samples:      {num_samples:,}")
print(f"✓ Samples per Fold: ~{len(fold_chunks[0]):,}")

# =====================================================================
# 3. Model 정의 (18번 참조 파일과 동일한 구조)
# =====================================================================
print("\n" + "="*70)
print("3️⃣  Model 설계")
print("="*70)

class ResidualDistancePredictor(nn.Module):
    """
    Shared MLP: 각 subspace (9 dims) → embed (8 dims)
    Global MLP: 16개 subspace embed concat (128 dims) → 1 (거리 예측)
    """
    def __init__(self):
        super(ResidualDistancePredictor, self).__init__()

        # Input normalization (subspace 단위)
        self.input_norm = nn.BatchNorm1d(FEATURE_DIM)

        # Shared MLP: (9) → (32) → (8)
        self.shared_mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM, SHARED_HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(SHARED_HIDDEN, EMBED_DIM),
            nn.LeakyReLU(0.1)
        )

        # Global MLP: 16 × 8 = 128 → 64 → 32 → 1
        global_input_dim = 16 * EMBED_DIM  # 128
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, GLOBAL_HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(GLOBAL_HIDDEN, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, 16, 9)
        batch_size = x.size(0)

        # Flatten for shared processing
        x_flat = x.view(-1, FEATURE_DIM)    # (batch*16, 9)

        # Input normalization
        x_norm = self.input_norm(x_flat)    # (batch*16, 9)

        # Shared encoding
        embeddings = self.shared_mlp(x_norm)  # (batch*16, 8)

        # Global prediction
        global_input = embeddings.view(batch_size, -1)  # (batch, 128)
        global_pred  = self.global_mlp(global_input)    # (batch, 1)

        return global_pred

# 모델 구조 출력 (한 번만)
_tmp = ResidualDistancePredictor()
print(_tmp)
print(f"\n✓ Total Parameters: {sum(p.numel() for p in _tmp.parameters()):,}")
del _tmp

# =====================================================================
# 4. Metric 계산 함수
# =====================================================================
def calculate_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

    y_range = y_true.max() - y_true.min()
    nrmse   = 1 - (rmse / y_range) if y_range > 0 else 0

    epsilon    = 1e-8
    mape       = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    mape_score = 1 / (1 + mape / 100)

    y_std = y_true.std()
    acc_01 = np.mean(np.abs(y_true - y_pred) < y_std * 0.1)
    acc_02 = np.mean(np.abs(y_true - y_pred) < y_std * 0.2)
    acc_03 = np.mean(np.abs(y_true - y_pred) < y_std * 0.3)
    acc_04 = np.mean(np.abs(y_true - y_pred) < y_std * 0.4)
    acc_05 = np.mean(np.abs(y_true - y_pred) < y_std * 0.5)

    return {
        'mse': mse, 'mae': mae, 'rmse': rmse,
        'r2': r2, 'corr': corr, 'nrmse': nrmse, 'mape_score': mape_score,
        'acc_like_0.1': acc_01, 'acc_like_0.2': acc_02, 'acc_like_0.3': acc_03,
        'acc_like_0.4': acc_04, 'acc_like_0.5': acc_05,
    }

# =====================================================================
# 5. 10-Fold OOF 학습
# =====================================================================
print("\n" + "="*70)
print("4️⃣  10-Fold OOF 학습 시작")
print("="*70)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
history = []

for fold in range(NUM_FOLDS):
    print(f"\n{'='*70}")
    print(f"📂 Fold {fold + 1}/{NUM_FOLDS}")
    print(f"{'='*70}")

    # ── Fold Split ──────────────────────────────────────────────────
    test_idx = fold_chunks[fold]
    val_idx  = fold_chunks[(fold + 1) % NUM_FOLDS]
    train_chunks = [fold_chunks[i]
                    for i in range(NUM_FOLDS)
                    if i != fold and i != (fold + 1) % NUM_FOLDS]
    train_idx = np.concatenate(train_chunks)

    print(f"✓ Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    # ── 데이터 추출 & 정규화 ─────────────────────────────────────────
    X_train = X_np[train_idx]
    X_val   = X_np[val_idx]
    X_test  = X_np[test_idx]

    y_global_train    = (y_global[train_idx] - y_global_mean) / y_global_std
    y_global_val      = (y_global[val_idx]   - y_global_mean) / y_global_std
    y_global_val_orig = y_global[val_idx]

    # ── DataLoader ───────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_global_train)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_global_val)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # ── Model 초기화 (Fold마다 새로 생성) ────────────────────────────
    model     = ResidualDistancePredictor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    best_val_loss    = float('inf')
    best_epoch       = 0
    best_model_state = None

    # ── Epoch Loop ───────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = nn.MSELoss()(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item()
        avg_train_loss = train_loss_sum / len(train_loader)

        # ── Validation ──
        model.eval()
        val_loss_sum  = 0.0
        all_val_preds = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                pred     = model(batch_X)
                loss     = nn.MSELoss()(pred, batch_y)
                val_loss_sum += loss.item()
                # denormalize
                pred_denorm = pred.cpu().numpy() * y_global_std + y_global_mean
                all_val_preds.append(pred_denorm)

        all_val_preds  = np.concatenate(all_val_preds)
        avg_val_loss   = val_loss_sum / len(val_loader)
        val_metrics    = calculate_metrics(y_global_val_orig, all_val_preds)

        scheduler.step(avg_val_loss)

        # ── Log ──
        history.append({
            'fold': fold + 1, 'epoch': epoch,
            'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
            'val_mse': val_metrics['mse'],       'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],      'val_r2': val_metrics['r2'],
            'val_corr': val_metrics['corr'],      'val_nrmse': val_metrics['nrmse'],
            'val_mape_score': val_metrics['mape_score'],
            'val_acc_like_0.1': val_metrics['acc_like_0.1'],
            'val_acc_like_0.2': val_metrics['acc_like_0.2'],
            'val_acc_like_0.3': val_metrics['acc_like_0.3'],
            'val_acc_like_0.4': val_metrics['acc_like_0.4'],
            'val_acc_like_0.5': val_metrics['acc_like_0.5'],
            'lr': optimizer.param_groups[0]['lr'],
        })

        # ── Best model 저장 (메모리) ──
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            best_epoch       = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # ── Console 출력 (10 epoch마다) ──
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{EPOCHS}] "
                  f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
                  f"R²: {val_metrics['r2']:.4f} | "
                  f"Corr: {val_metrics['corr']:.4f}")

    # ── Best model 디스크에 저장 ──────────────────────────────────────
    print(f"\n✓ Best Epoch: {best_epoch}  Best Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_state)

    model_save_path = os.path.join(MODEL_SAVE_DIR, f"model_k{fold + 1}.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"✓ Model saved: {model_save_path}")

    # ── Test (OOF) Prediction ─────────────────────────────────────────
    model.eval()
    X_test_tensor = torch.tensor(X_test).to(DEVICE)
    with torch.no_grad():
        test_pred = model(X_test_tensor)
        test_pred_denorm = test_pred.cpu().numpy() * y_global_std + y_global_mean
    oof_preds[test_idx] = test_pred_denorm
    print(f"✓ Fold {fold + 1} 완료!")

# =====================================================================
# 6. 결과 저장 & 출력
# =====================================================================
print("\n" + "="*70)
print("5️⃣  결과 저장 & 출력")
print("="*70)

# CSV 로그 저장
pd.DataFrame(history).to_csv(LOG_PATH, index=False)
print(f"✓ 학습 로그 저장: {LOG_PATH}")

# OOF 예측 저장
np.savez_compressed(OOF_PATH, pred=oof_preds)
print(f"✓ OOF 예측 저장:  {OOF_PATH}  shape={oof_preds.shape}")

# 전체 OOF 성능
oof_metrics = calculate_metrics(y_global, oof_preds)

print(f"\n{'='*70}")
print(f"🏆 Overall OOF Performance ({NUM_FOLDS}-Fold)")
print(f"{'='*70}")
print(f"  MSE:             {oof_metrics['mse']:.4f}")
print(f"  MAE:             {oof_metrics['mae']:.4f}")
print(f"  RMSE:            {oof_metrics['rmse']:.4f}")
print(f"  R² Score:        {oof_metrics['r2']:.4f}")
print(f"  Correlation:     {oof_metrics['corr']:.4f}")
print(f"  NRMSE:           {oof_metrics['nrmse']:.4f}")
print(f"  MAPE Score:      {oof_metrics['mape_score']:.4f}")
print(f"  Acc-like (10%):  {oof_metrics['acc_like_0.1']:.4f}")
print(f"  Acc-like (20%):  {oof_metrics['acc_like_0.2']:.4f}")
print(f"  Acc-like (30%):  {oof_metrics['acc_like_0.3']:.4f}")
print(f"  Acc-like (40%):  {oof_metrics['acc_like_0.4']:.4f}")
print(f"  Acc-like (50%):  {oof_metrics['acc_like_0.5']:.4f}")
print(f"{'='*70}")

print(f"\n✅ 학습 완료!")
print(f"   저장된 모델: model_k1.pt ~ model_k{NUM_FOLDS}.pt")
print(f"   저장 경로:   {MODEL_SAVE_DIR}")
