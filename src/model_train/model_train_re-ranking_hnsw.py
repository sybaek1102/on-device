#!/usr/bin/env python3
"""
model_train_re-ranking_hnsw.py
──────────────────────────────
create_reranking_model_train_data_hnsw.py 로 생성된 feature/label로
10-Fold OOF 이진 분류 MLP 모델을 학습합니다. (HNSW 버전)

[입력]
  re-ranking_features_hnsw.npz : shape (10000, 33)
                             - [:, :32] : features (PQ dist 16 + Residual dist 16)
                             - [:, 32]  : label (0/1)

[출력]
  model_k1.pt ~ model_k10.pt : 각 Fold val_loss best 모델
  train_log.csv               : epoch별 학습 로그
  oof_result.npz              : OOF 확률 / 예측 레이블

[저장 위치]
  data/model/re-ranking_hnsw/
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import os
import sys

# =============================================================================
# 🔹 Configuration
# =============================================================================
FEATURE_PATH = "/home/syback/vectorDB/on-device/data/model/re-ranking_hnsw/re-ranking_features_hnsw.npz"

MODEL_SAVE_DIR = "/home/syback/vectorDB/on-device/data/model/re-ranking_hnsw"
LOG_PATH       = os.path.join(MODEL_SAVE_DIR, "train_log.csv")
OOF_PATH       = os.path.join(MODEL_SAVE_DIR, "oof_result.npz")

# 하이퍼파라미터
BATCH_SIZE    = 128
LEARNING_RATE = 0.001
MAX_EPOCHS    = 100
THRESHOLD     = 0.5
NUM_FOLDS     = 10

# =============================================================================
# 1. 데이터 로드 & 전처리
# =============================================================================
print("\n" + "="*70)
print("📂 Re-ranking MLP 학습 (10-Fold OOF)")
print("="*70)
print("\n1️⃣  데이터 로드 & 전처리")

if not os.path.exists(FEATURE_PATH):
    print(f"❌ 파일이 존재하지 않습니다: {FEATURE_PATH}")
    sys.exit(1)

with np.load(FEATURE_PATH) as f:
    dataset = f["data"]                                # (10000, 33)

X_numpy = dataset[:, :-1].astype(np.float32)          # (10000, 32)
y_numpy = dataset[:,  -1].astype(np.float32).reshape(-1, 1)  # (10000, 1)

print(f"  ✓ Feature Shape : {X_numpy.shape}")
print(f"  ✓ Label  Shape  : {y_numpy.shape}")
print(f"  ✓ Label Dist    - 0: {int(np.sum(y_numpy == 0)):,}  /  1: {int(np.sum(y_numpy == 1)):,}")

# =============================================================================
# 2. OOF 결과 저장 배열 & Fold Split
# =============================================================================
num_samples = len(X_numpy)
all_indices = np.arange(num_samples)
oof_probs   = np.zeros((num_samples, 1), dtype=np.float32)

fold_chunks = np.array_split(all_indices, NUM_FOLDS)

print(f"\n  ✓ Total Samples:      {num_samples:,}")
print(f"  ✓ Samples per Fold: ~{len(fold_chunks[0]):,}")

# =============================================================================
# 3. Model 정의 (21번 참조 파일과 동일한 구조)
# =============================================================================
class SimpleMLP(nn.Module):
    """
    Input: 32 dims (PQ Dist 16 + OOF-based Residual Dist 16)
    Output: 1 dim (Sigmoid → re-ranking 필요 확률)
    """
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# 구조 확인용 출력
_tmp = SimpleMLP()
print(f"\n  ✓ Model Architecture:\n{_tmp}")
print(f"  ✓ Total Parameters: {sum(p.numel() for p in _tmp.parameters()):,}")
del _tmp

# =============================================================================
# 4. 10-Fold OOF 학습
# =============================================================================
print(f"\n4️⃣  학습 시작: {NUM_FOLDS}-Fold OOF, Max Epochs: {MAX_EPOCHS}")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
history = []

for fold in range(NUM_FOLDS):
    print(f"\n{'='*70}")
    print(f"📂 Fold {fold + 1}/{NUM_FOLDS}")
    print(f"{'='*70}")

    # ── Fold Split ──────────────────────────────────────────────────────────
    test_idx = fold_chunks[fold]
    val_idx  = fold_chunks[(fold + 1) % NUM_FOLDS]
    train_chunks = [fold_chunks[i]
                    for i in range(NUM_FOLDS)
                    if i != fold and i != (fold + 1) % NUM_FOLDS]
    train_idx = np.concatenate(train_chunks)

    print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    # ── Raw 슬라이싱 ─────────────────────────────────────────────────────────
    X_train_raw = X_numpy[train_idx]
    X_val_raw   = X_numpy[val_idx]
    X_test_raw  = X_numpy[test_idx]

    # ── Feature 그룹별 독립 Scaling ──────────────────────────────────────────
    # Group 1: PQ Distance (앞 16개)
    # Group 2: OOF-based Residual Dist (뒤 16개)
    scaler_f1 = StandardScaler()
    scaler_f2 = StandardScaler()

    X_train_f1 = scaler_f1.fit_transform(X_train_raw[:, :16])
    X_train_f2 = scaler_f2.fit_transform(X_train_raw[:, 16:])

    X_val_f1 = scaler_f1.transform(X_val_raw[:, :16])
    X_val_f2 = scaler_f2.transform(X_val_raw[:, 16:])

    X_test_f1 = scaler_f1.transform(X_test_raw[:, :16])
    X_test_f2 = scaler_f2.transform(X_test_raw[:, 16:])

    X_train_scaled = np.hstack([X_train_f1, X_train_f2])
    X_val_scaled   = np.hstack([X_val_f1,   X_val_f2])
    X_test_scaled  = np.hstack([X_test_f1,  X_test_f2])

    # ── Tensor 변환 ──────────────────────────────────────────────────────────
    X_train_t = torch.tensor(X_train_scaled)
    y_train_t = torch.tensor(y_numpy[train_idx])
    X_val_t   = torch.tensor(X_val_scaled)
    y_val_t   = torch.tensor(y_numpy[val_idx])
    X_test_t  = torch.tensor(X_test_scaled)

    # ── 모델 초기화 (Fold마다 새로 생성) ─────────────────────────────────────
    model     = SimpleMLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss    = float("inf")
    best_epoch       = 0
    best_model_state = None

    # ── Epoch Loop ────────────────────────────────────────────────────────────
    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        permutation    = torch.randperm(X_train_t.size(0))
        epoch_loss     = 0.0
        tr_probs_list  = []
        tr_labels_list = []

        for i in range(0, X_train_t.size(0), BATCH_SIZE):
            idx      = permutation[i: i + BATCH_SIZE]
            batch_x  = X_train_t[idx]
            batch_y  = y_train_t[idx]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss    = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tr_probs_list.append(outputs.detach().cpu().numpy())
            tr_labels_list.append(batch_y.detach().cpu().numpy())

        avg_train_loss     = epoch_loss / max(1, len(X_train_t) // BATCH_SIZE)
        train_probs_cat    = np.concatenate(tr_probs_list)
        train_labels_cat   = np.concatenate(tr_labels_list)
        train_auc          = roc_auc_score(train_labels_cat, train_probs_cat)
        train_preds        = (train_probs_cat >= THRESHOLD).astype(int)
        train_acc          = accuracy_score(train_labels_cat, train_preds)
        tr_prec, tr_rec, _, _ = precision_recall_fscore_support(
            train_labels_cat, train_preds, average=None, zero_division=0)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss    = criterion(val_outputs, y_val_t).item()
            val_probs   = val_outputs.cpu().numpy()
            val_labels  = y_val_t.cpu().numpy()

        val_auc  = roc_auc_score(val_labels, val_probs)
        val_preds= (val_probs >= THRESHOLD).astype(int)
        val_acc  = accuracy_score(val_labels, val_preds)
        val_prec, val_rec, _, _ = precision_recall_fscore_support(
            val_labels, val_preds, average=None, zero_division=0)

        # Log
        log_entry = {
            "fold": fold + 1, "epoch": epoch,
            "train_loss": avg_train_loss, "train_acc": train_acc, "train_auc": train_auc,
            "train_prec0": tr_prec[0],  "train_rec0": tr_rec[0],
            "train_prec1": tr_prec[1],  "train_rec1": tr_rec[1],
            "val_loss": val_loss, "val_acc": val_acc, "val_auc": val_auc,
            "val_prec0": val_prec[0],   "val_rec0": val_rec[0],
            "val_prec1": val_prec[1],   "val_rec1": val_rec[1],
        }
        history.append(log_entry)

        # Best model 체크 & 메모리 저장
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # 콘솔 출력 (10 epoch마다)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{MAX_EPOCHS}] "
                  f"Loss: {avg_train_loss:.4f}/{val_loss:.4f} | "
                  f"AUC: {train_auc:.4f}/{val_auc:.4f} | "
                  f"Acc: {train_acc:.4f}/{val_acc:.4f}")

    # ── Best model 디스크 저장 ────────────────────────────────────────────────
    print(f"\n  ✓ Best Epoch: {best_epoch}  Best Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_state)

    model_path = os.path.join(MODEL_SAVE_DIR, f"model_k{fold + 1}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  ✓ Model saved: {model_path}")

    # ── Scaler 파라미터 저장 (inference 재현용) ───────────────────────────────
    scaler_path = os.path.join(MODEL_SAVE_DIR, f"scaler_k{fold + 1}.npz")
    np.savez(scaler_path,
             f1_mean=scaler_f1.mean_,   f1_std=np.sqrt(scaler_f1.var_),
             f2_mean=scaler_f2.mean_,   f2_std=np.sqrt(scaler_f2.var_))
    print(f"  ✓ Scaler saved: {scaler_path}")

    # ── Test (OOF) Prediction (best model 기준) ───────────────────────────────
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        oof_probs[test_idx] = test_outputs.cpu().numpy()

    print(f"  ✓ Fold {fold + 1} 완료!")

# =============================================================================
# 5. 결과 저장
# =============================================================================
print("\n" + "="*70)
print("5️⃣  결과 저장")
print("="*70)

pd.DataFrame(history).to_csv(LOG_PATH, index=False)
print(f"✓ 학습 로그 저장: {LOG_PATH}")

oof_pred_labels = (oof_probs >= THRESHOLD).astype(np.float32)
np.savez_compressed(OOF_PATH, pred_prob=oof_probs, pred_label=oof_pred_labels)
print(f"✓ OOF 결과 저장:  {OOF_PATH}")
print(f"  - pred_prob  shape: {oof_probs.shape}")
print(f"  - pred_label shape: {oof_pred_labels.shape}")

# =============================================================================
# 6. 전체 OOF 성능 평가
# =============================================================================
print("\n" + "="*70)
print("6️⃣  전체 OOF 성능 (10-Fold)")
print("="*70)

oof_auc  = roc_auc_score(y_numpy, oof_probs)
oof_acc  = accuracy_score(y_numpy, oof_pred_labels)
oof_prec, oof_rec, _, _ = precision_recall_fscore_support(
    y_numpy, oof_pred_labels, average=None, zero_division=0)

print(f"  Overall OOF Accuracy  : {oof_acc:.4f}")
print(f"  Overall OOF AUC       : {oof_auc:.4f}")
print(f"  Class 0 Precision     : {oof_prec[0]:.4f}")
print(f"  Class 0 Recall        : {oof_rec[0]:.4f}")
print(f"  Class 1 Precision     : {oof_prec[1]:.4f}")
print(f"  Class 1 Recall        : {oof_rec[1]:.4f}")

print("\n" + "="*70)
print("[Feature 구성]")
print("  Index  0~15 : PQ Distance     (16 dims)  — ||Q-P||² per candidate")
print("  Index 16~31 : Residual Dist   (16 dims)  — ||X-P||² - 2·pred(dot(Q-P,X-P))")
print("  Total: 32 dims")
print("="*70)
print(f"\n✅ 학습 완료!")
print(f"   저장된 모델: model_k1.pt ~ model_k{NUM_FOLDS}.pt")
print(f"   저장 경로:   {MODEL_SAVE_DIR}")
