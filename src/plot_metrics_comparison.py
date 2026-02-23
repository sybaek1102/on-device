#!/usr/bin/env python3
"""
plot_metrics_comparison.py
──────────────────────────
output/metric/ 의 3개 JSON 파일에서 metrics를 읽어
그룹 막대 그래프로 비교합니다.

[저장]
  output/graph/metrics_comparison_fold{EVAL_FOLD}.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# 🔹 Configuration
# =============================================================================
EVAL_FOLD  = 1   # 읽을 JSON의 fold 번호

METRIC_DIR = "/home/syback/vectorDB/on-device/output/metric"
GRAPH_DIR  = "/home/syback/vectorDB/on-device/output/graph"

JSON_FILES = {
    "no":        os.path.join(METRIC_DIR, f"no_query_search_fold{EVAL_FOLD}.json"),
    "selective": os.path.join(METRIC_DIR, f"selective_query_search_fold{EVAL_FOLD}.json"),
    "all":       os.path.join(METRIC_DIR, f"all_query_search_fold{EVAL_FOLD}.json"),
}

# 모델별 색상 (같은 모델 = 같은 색)
COLORS = {
    "no":        "#4C72B0",   # 파란계열
    "selective": "#DD8452",   # 주황계열
    "all":       "#55A868",   # 초록계열
}

LABELS = {
    "no":        "No Re-ranking",
    "selective": "Proposed",
    "all":       "All Re-ranking",
}

METRIC_KEYS  = ["recall_at_1", "mrr", "distance_ratio"]
METRIC_NAMES = ["Recall@1",    "MRR", "Distance Ratio"]

# =============================================================================
# 🔹 Main
# =============================================================================
if __name__ == "__main__":
    os.makedirs(GRAPH_DIR, exist_ok=True)

    # JSON 로드
    data = {}
    for name, path in JSON_FILES.items():
        with open(path) as f:
            data[name] = json.load(f)

    models = ["no", "all", "selective"]   # no → all → proposed
    n_models  = len(models)
    n_metrics = len(METRIC_KEYS)

    x = np.arange(n_metrics)          # 메트릭별 x 위치
    bar_width = 0.22
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(models):
        values = [data[model]["metrics"][k] for k in METRIC_KEYS]
        bars = ax.bar(
            x + offsets[i], values,
            width=bar_width,
            color=COLORS[model],
            label=LABELS[model],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        # 막대 위 수치 표시
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.4f}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                color="#333333",
            )

    # 축 / 레이아웃
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Metrics Comparison — Fold {EVAL_FOLD}  (Query {(EVAL_FOLD-1)*1000}~{EVAL_FOLD*1000-1})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(GRAPH_DIR, f"metrics_comparison_fold{EVAL_FOLD}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()
