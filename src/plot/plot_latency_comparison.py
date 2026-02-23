#!/usr/bin/env python3
"""
plot_latency_comparison_v1.py
───────────────────────────────────────────────────────────────────────
output/metric/ 의 3개 JSON 파일에서 latency를 읽어
그룹 막대 그래프(ms 단위)로 비교합니다.

항목:
  - PQ Search avg (ms)
  - Re-ranking avg (ms)  ← all: L2, selective: MLP+L2 합산
  - Total per query avg (ms)

[저장]
  output/graph/latency_comparison_fold{EVAL_FOLD}.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 🔹 Configuration
# =============================================================================
EVAL_FOLD  = 1

METRIC_DIR = "/home/syback/vectorDB/on-device/output/metric"
GRAPH_DIR  = "/home/syback/vectorDB/on-device/output/graph"

JSON_FILES = {
    "no":        os.path.join(METRIC_DIR, f"no_query_search_fold{EVAL_FOLD}.json"),
    "selective": os.path.join(METRIC_DIR, f"selective_query_search_fold{EVAL_FOLD}.json"),
    "all":       os.path.join(METRIC_DIR, f"all_query_search_fold{EVAL_FOLD}.json"),
}

COLORS = {
    "no":        "#4C72B0",
    "selective": "#DD8452",
    "all":       "#55A868",
}

LABELS = {
    "no":        "No Re-ranking",
    "selective": "Proposed",
    "all":       "All Re-ranking",
}

# =============================================================================
# 🔹 Main
# =============================================================================
if __name__ == "__main__":
    os.makedirs(GRAPH_DIR, exist_ok=True)

    data = {}
    for name, path in JSON_FILES.items():
        with open(path) as f:
            data[name] = json.load(f)

    def get_latency(name, d):
        lat = d["latency"]
        num_q = d["num_query"]
        pq_avg   = lat.get("query_search_avg_ms", 0)

        if name == "selective":
            # MLP + L2 re-ranking 합산 → "후처리" 전체 비용
            extra_avg = lat.get("mlp_pipeline_avg_ms", 0) + lat.get("re_ranking_avg_ms", 0)
        else:
            extra_avg = lat.get("re_ranking_avg_ms", 0)

        total_avg = (lat.get("query_search_total_ms", 0) +
                     lat.get("mlp_pipeline_total_ms", 0) +
                     lat.get("re_ranking_total_ms", 0)) / num_q

        return pq_avg, extra_avg, total_avg

    models   = ["no", "all", "selective"]   # no → all → proposed
    lat_data = {m: get_latency(m, data[m]) for m in models}

    # x축: PQ Search avg / Post-processing avg / Total per query
    group_labels = ["PQ Search\n(avg/query)", "re-ranking\n(avg/query)", "Total\n(per query)"]
    n_groups  = len(group_labels)
    n_models  = len(models)

    x        = np.arange(n_groups)
    bar_width = 0.22
    offsets   = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * bar_width

    all_vals = [v for m in models for v in lat_data[m]]
    y_max    = max(all_vals) * 1.22

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(models):
        values = list(lat_data[model])
        bars = ax.bar(
            x + offsets[i], values,
            width=bar_width,
            color=COLORS[model],
            label=LABELS[model],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.01,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title(
        f"Latency Comparison — Fold {EVAL_FOLD}  (Query {(EVAL_FOLD-1)*1000}~{EVAL_FOLD*1000-1})",
        fontsize=13, fontweight="bold", pad=12,
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(GRAPH_DIR, f"latency_comparison_fold{EVAL_FOLD}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()
