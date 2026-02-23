#!/usr/bin/env python3
"""
plot_diskio_comparison.py
──────────────────────────
output/metric/ 의 3개 JSON 파일에서 disk_io를 읽어
그룹 막대 그래프(MB 단위)로 비교합니다.

[저장]
  output/graph/diskio_comparison_fold{EVAL_FOLD}.png
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

def bytes_to_mb(b):
    return b / (1024 ** 2)

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

    # disk_io 항목 통일 — 키가 파일마다 다를 수 있으므로 안전하게 읽기
    def get_io(d):
        io = d["disk_io"]
        index_mb  = bytes_to_mb(io.get("index_load_bytes", 0))
        # search I/O 키 이름이 파일마다 다름
        search_mb = bytes_to_mb(
            io.get("search_and_reranking_bytes",
            io.get("search_and_mlp_bytes",
            io.get("search_io_bytes", 0)))
        )
        total_mb  = bytes_to_mb(io.get("total_io_bytes", 0))
        return index_mb, search_mb, total_mb

    models   = ["no", "all", "selective"]   # no → all → proposed
    io_items = {m: get_io(data[m]) for m in models}

    # ── 3개 서브플롯: Index Load / Search+Rerank / Total ──────────────────
    io_labels = ["Index Load", "Search / Rerank", "Total"]
    n_groups  = len(io_labels)
    n_models  = len(models)

    x        = np.arange(n_groups)
    bar_width = 0.22
    offsets   = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * bar_width

    # y 범위: 최대값에 여유 20%
    all_vals = [v for m in models for v in io_items[m]]
    y_max    = max(all_vals) * 1.22

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(models):
        idx_mb, srch_mb, tot_mb = io_items[model]
        values = [idx_mb, srch_mb, tot_mb]
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
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(io_labels, fontsize=12)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("I/O (MB)", fontsize=11)
    ax.set_title(
        f"Disk I/O Comparison — Fold {EVAL_FOLD}  (Query {(EVAL_FOLD-1)*1000}~{EVAL_FOLD*1000-1})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(GRAPH_DIR, f"diskio_comparison_fold{EVAL_FOLD}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()
