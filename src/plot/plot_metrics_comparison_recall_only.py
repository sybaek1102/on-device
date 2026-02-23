#!/usr/bin/env python3
"""
plot_metrics_comparison_recall_only.py
──────────────────────────────────────
Recall@1 비교 그래프.

[X축 구성] M=16 / M=32 / HNSW  (3개 그룹)
[막대 구성] 각 그룹마다 No / All / Proposed  (3개 막대)

[색상]
  No       : #D9D9D9  (연회색)
  All      : #666666  (진한 회색)
  Proposed : #8B1E2A  (와인색)

[저장]
  output/graph/metrics_comparison_fold{EVAL_FOLD}_recall_only.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# 🔹 Configuration
# =============================================================================
EVAL_FOLD  = 1
METRIC_DIR = "/home/syback/vectorDB/on-device/output/metric"
GRAPH_DIR  = "/home/syback/vectorDB/on-device/output/graph"

# =============================================================================
# 🔹 그룹 정의 — 3개 그룹 × 3개 막대
# =============================================================================
# 각 그룹: (그룹 라벨, {no_key, all_key, proposed_key})
GROUPS = [
    ("M=16",  "no",       "all",      "selective"),
    ("M=32",  "no_same",  "all_same", "selective_res"),
    ("HNSW",  "no_hnsw",  "all_hnsw", "selective_hnsw"),
]

# JSON 파일 경로
JSON_FILES = {
    "no":             os.path.join(METRIC_DIR, f"no_query_search_fold{EVAL_FOLD}.json"),
    "all":            os.path.join(METRIC_DIR, f"all_query_search_fold{EVAL_FOLD}.json"),
    "selective":      os.path.join(METRIC_DIR, f"selective_query_search_fold{EVAL_FOLD}.json"),
    "no_same":        os.path.join(METRIC_DIR, f"no_same_quality_query_search_fold{EVAL_FOLD}.json"),
    "all_same":       os.path.join(METRIC_DIR, f"all_same_quality_query_search_fold{EVAL_FOLD}.json"),
    "selective_res":  os.path.join(METRIC_DIR, f"selective_query_search_res_fold{EVAL_FOLD}.json"),
    "no_hnsw":        os.path.join(METRIC_DIR, f"no_hnsw_query_search_fold{EVAL_FOLD}.json"),
    "all_hnsw":       os.path.join(METRIC_DIR, f"all_hnsw_query_search_fold{EVAL_FOLD}.json"),
    "selective_hnsw": os.path.join(METRIC_DIR, f"selective_hnsw_query_search_fold{EVAL_FOLD}.json"),
}

# 막대 색상 (역할별)
C_NO       = "#D9D9D9"   # 연회색
C_ALL      = "#666666"   # 진한 회색
C_PROPOSED = "#8B1E2A"   # 와인색

# =============================================================================
# 🔹 Main
# =============================================================================
if __name__ == "__main__":
    os.makedirs(GRAPH_DIR, exist_ok=True)

    # JSON 로드
    data = {}
    for key, path in JSON_FILES.items():
        with open(path) as f:
            data[key] = json.load(f)

    def recall(key):
        return data[key]["metrics"]["recall_at_1"]

    # ── 그래프 설정 ────────────────────────────────────────────────────────
    n_groups  = len(GROUPS)
    bar_width = 0.22
    gap       = 0.03          # 같은 그룹 내 막대 간격
    spacing   = bar_width + gap

    # 그룹 중심 위치
    x = np.arange(n_groups)

    # 3개 막대의 오프셋 (중심 기준)
    offsets = np.array([-1, 0, 1]) * spacing

    fig, ax = plt.subplots(figsize=(9, 5))

    for gi, (grp_label, k_no, k_all, k_prop) in enumerate(GROUPS):
        vals   = [recall(k_no), recall(k_all), recall(k_prop)]
        colors = [C_NO, C_ALL, C_PROPOSED]
        keys   = [k_no, k_all, k_prop]

        for bi, (val, color) in enumerate(zip(vals, colors)):
            xpos = x[gi] + offsets[bi]
            bar  = ax.bar(
                xpos, val,
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            # 수치 레이블
            text_color = "white" if color == C_PROPOSED else "#333333"
            ax.text(
                xpos, val + 0.010,
                f"{val:.4f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color="#333333",
                rotation=0,
            )

    # ── 축 / 레이아웃 ───────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in GROUPS], fontsize=13, fontweight="bold")
    ax.set_xlim(-0.6, n_groups - 0.4)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall@1", fontsize=12)
    ax.set_title(
        f"Recall@1 Comparison — Fold {EVAL_FOLD}  (Query {(EVAL_FOLD-1)*1000}~{EVAL_FOLD*1000-1})",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 범례 ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=C_NO,       label="No Re-ranking",  edgecolor="white"),
        mpatches.Patch(color=C_ALL,      label="All Re-ranking", edgecolor="white"),
        mpatches.Patch(color=C_PROPOSED, label="Proposed",       edgecolor="white"),
    ]
    ax.legend(handles=legend_handles, fontsize=10.5, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc")

    # ── 저장 ────────────────────────────────────────────────────────────────
    plt.tight_layout()
    save_path = os.path.join(GRAPH_DIR, f"metrics_comparison_fold{EVAL_FOLD}_recall_only.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()
