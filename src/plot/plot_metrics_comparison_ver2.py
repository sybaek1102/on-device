#!/usr/bin/env python3
"""
plot_metrics_comparison.py
──────────────────────────
output/metric/ 의 4개 JSON 파일에서 metrics를 읽어
그룹 막대 그래프로 비교합니다.

[순서] no → all → no_same_quality → selective (proposed)

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
    "no":           os.path.join(METRIC_DIR, f"no_query_search_fold{EVAL_FOLD}.json"),
    "all":          os.path.join(METRIC_DIR, f"all_query_search_fold{EVAL_FOLD}.json"),
    "no_same":      os.path.join(METRIC_DIR, f"no_same_quality_query_search_fold{EVAL_FOLD}.json"),
    "all_same":     os.path.join(METRIC_DIR, f"all_same_quality_query_search_fold{EVAL_FOLD}.json"),
    "selective":    os.path.join(METRIC_DIR, f"selective_query_search_fold{EVAL_FOLD}.json"),
    "selective_res": os.path.join(METRIC_DIR, f"selective_query_search_res_fold{EVAL_FOLD}.json"),
    "no_hnsw":      os.path.join(METRIC_DIR, f"no_hnsw_query_search_fold{EVAL_FOLD}.json"),
    "all_hnsw":     os.path.join(METRIC_DIR, f"all_hnsw_query_search_fold{EVAL_FOLD}.json"),
}

COLORS = {
    "no":            "#D9D9D9",
    "all":           "#D9D9D9",
    "no_same":       "#777777",
    "all_same":      "#777777",
    "selective":     "#8B1E2A",
    "selective_res": "#8B1E2A",
    "no_hnsw":       "#444444",   # 진한 회색 (HNSW)
    "all_hnsw":      "#444444",   # 진한 회색 (HNSW)
}

LABELS = {
    "no":            "No Re-ranking (M=16)",
    "all":           "All Re-ranking (M=16)",
    "no_same":       "No Re-ranking (M=32)",
    "all_same":      "All Re-ranking (M=32)",
    "selective":     "Proposed (M=16)",
    "selective_res": "Proposed (M=32)",
    "no_hnsw":       "No Re-ranking (HNSW)",
    "all_hnsw":      "All Re-ranking (HNSW)",
}

MODELS = ["no", "all", "no_same", "all_same", "selective", "selective_res", "no_hnsw", "all_hnsw"]

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

    n_models  = len(MODELS)
    n_metrics = len(METRIC_KEYS)

    x         = np.arange(n_metrics)
    bar_width  = 0.11
    offsets    = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * (bar_width + 0.01)

    fig, ax = plt.subplots(figsize=(16, 5))

    for i, model in enumerate(MODELS):
        values = [data[model]["metrics"][k] for k in METRIC_KEYS]
        bars = ax.bar(
            x + offsets[i], values,
            width=bar_width,
            color=COLORS[model],
            label=LABELS[model],
            edgecolor="gray",
            linewidth=0.8,
            zorder=3,
        )
        # 막대 위 수치 표시
        for bar, val in zip(bars, values):
            txt_color = "white" if model == "selective" else "#444444"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.010,
                f"{val:.4f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color="#333333",
            )

    # 축 / 레이아웃
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES, fontsize=12)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Metrics Comparison — Fold {EVAL_FOLD}  (Query {(EVAL_FOLD-1)*1000}~{EVAL_FOLD*1000-1})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(GRAPH_DIR, f"metrics_comparison_fold{EVAL_FOLD}_ver2.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()
