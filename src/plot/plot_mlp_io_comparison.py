#!/usr/bin/env python3
"""
plot_mlp_io_comparison.py
─────────────────────────────────────────────────────────────────────
9개 방식의 평균 Latency 비교 (stacked bar: T_search / T_mlp / T_IO)

[X축 그룹] M=16 / M=32 / HNSW  (3개 그룹, 각 그룹에 No / All / Selective 3개 막대)

[T 정의]
  T_search: query_search_avg_ms
  T_mlp:    mlp_pipeline_avg_ms  (selective만 존재, 나머지 0)
  T_IO:     re_ranking_avg_ms    (1,000개 기준 평균)

[저장]
  output/graph/mlp_io_comparison_fold{EVAL_FOLD}.png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# 설정
# =============================================================================
EVAL_FOLD  = 1
METRIC_DIR = "/home/syback/vectorDB/on-device/output/metric"
GRAPH_DIR  = "/home/syback/vectorDB/on-device/output/graph"
OUTPUT_PNG = os.path.join(GRAPH_DIR, f"mlp_io_comparison_fold{EVAL_FOLD}.png")

# =============================================================================
# 그룹 정의 — (그룹 X축 라벨,  no_key,    all_key,    sel_key)
# =============================================================================
GROUPS = [
    ("M=16", "no_m16",  "all_m16",  "sel_m16"),
    ("M=32", "no_m32",  "all_m32",  "sel_res"),
    ("HNSW", "no_hnsw", "all_hnsw", "sel_hnsw"),
]

JSON_FILES = {
    "no_m16":  os.path.join(METRIC_DIR, f"no_query_search_fold{EVAL_FOLD}.json"),
    "all_m16": os.path.join(METRIC_DIR, f"all_query_search_fold{EVAL_FOLD}.json"),
    "sel_m16": os.path.join(METRIC_DIR, f"selective_query_search_fold{EVAL_FOLD}.json"),
    "no_m32":  os.path.join(METRIC_DIR, f"no_same_quality_query_search_fold{EVAL_FOLD}.json"),
    "all_m32": os.path.join(METRIC_DIR, f"all_same_quality_query_search_fold{EVAL_FOLD}.json"),
    "sel_res": os.path.join(METRIC_DIR, f"selective_query_search_res_fold{EVAL_FOLD}.json"),
    "no_hnsw": os.path.join(METRIC_DIR, f"no_hnsw_query_search_fold{EVAL_FOLD}.json"),
    "all_hnsw":os.path.join(METRIC_DIR, f"all_hnsw_query_search_fold{EVAL_FOLD}.json"),
    "sel_hnsw":os.path.join(METRIC_DIR, f"selective_hnsw_query_search_fold{EVAL_FOLD}.json"),
}

# 막대 색상
COLOR_SEARCH = "#4CAF7D"    # 초록  (T_search)
COLOR_MLP    = "#4C9BE8"    # 파란  (T_mlp)
COLOR_IO     = "#E05C5C"    # 붉은  (T_IO)

# =============================================================================
# 데이터 로드
# =============================================================================
data = {}
for key, path in JSON_FILES.items():
    with open(path) as f:
        data[key] = json.load(f)

def get_latencies(key):
    lat = data[key]["latency"]
    t_search = lat.get("query_search_avg_ms", 0)
    t_mlp    = lat.get("mlp_pipeline_avg_ms", 0)
    t_io     = lat.get("re_ranking_avg_ms",   0)
    return t_search, t_mlp, t_io

latencies = {k: get_latencies(k) for k in JSON_FILES}
for k, (ts, tm, ti) in latencies.items():
    print(f"[{k:9s}] T_search={ts:.3f}  T_mlp={tm:.3f}  T_IO={ti:.3f}  total={ts+tm+ti:.3f}")

# =============================================================================
# 그래프
# =============================================================================
os.makedirs(GRAPH_DIR, exist_ok=True)

n_groups  = len(GROUPS)      # 3
bar_w     = 0.22
gap       = 0.03
spacing   = bar_w + gap

# 그룹 중심 위치
x_group = np.arange(n_groups)
# 그룹 내 3개 막대 오프셋
offsets = np.array([-1, 0, 1]) * spacing

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

all_totals = []

for gi, (grp_label, k_no, k_all, k_sel) in enumerate(GROUPS):
    for bi, key in enumerate([k_no, k_all, k_sel]):
        ts, tm, ti = latencies[key]
        total = ts + tm + ti
        all_totals.append(total)
        xp = x_group[gi] + offsets[bi]

        # T_search (초록, 맨 아래)
        ax.bar(xp, ts, width=bar_w, color=COLOR_SEARCH,
               zorder=3, edgecolor="white", linewidth=0.6)
        if ts >= 0.3:
            ax.text(xp, ts / 2, f"{ts:.2f}",
                    ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color="white")

        # T_mlp (파란, 중간) — selective만 존재
        if tm > 0:
            ax.bar(xp, tm, width=bar_w, bottom=ts, color=COLOR_MLP,
                   zorder=3, edgecolor="white", linewidth=0.6)
            if tm >= 0.1:
                ax.text(xp, ts + tm / 2, f"{tm:.3f}",
                        ha="center", va="center", fontsize=7.5,
                        fontweight="bold", color="white")

        # T_IO (붉은, 맨 위) — all/selective만 존재
        if ti > 0:
            ax.bar(xp, ti, width=bar_w, bottom=ts + tm, color=COLOR_IO,
                   zorder=3, edgecolor="white", linewidth=0.6)
            if ti >= 0.3:
                ax.text(xp, ts + tm + ti / 2, f"{ti:.3f}",
                        ha="center", va="center", fontsize=7.5,
                        fontweight="bold", color="white")

        # 상단 total
        ax.text(xp, total + 0.05,
                f"{total:.2f}",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#333333")


# ── 그룹 라벨 (X축) ──────────────────────────────────────────────────────────
ax.set_xticks(x_group)
ax.set_xticklabels([g[0] for g in GROUPS], fontsize=13, fontweight="bold")

# 그룹 내 No/All/Selective 위치 표시용 보조 tick
minor_ticks  = []
minor_labels = []
SUB_LABELS = ["No", "All", "Sel."]
for gi in range(n_groups):
    for bi, slabel in enumerate(SUB_LABELS):
        minor_ticks.append(x_group[gi] + offsets[bi])
        minor_labels.append(slabel)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(minor_labels, minor=True, fontsize=8, color="#555555")
ax.tick_params(axis="x", which="minor", length=0, pad=28)

# ── 그룹 구분선 ───────────────────────────────────────────────────────────────
for gi in range(n_groups - 1):
    sep = (x_group[gi] + x_group[gi + 1]) / 2
    ax.axvline(sep, color="#cccccc", linewidth=1.2, linestyle="--", zorder=1)

# ── 축 설정 ───────────────────────────────────────────────────────────────────
ax.set_ylabel("Average Latency (ms)", fontsize=11)
ax.set_title(f"Average Latency Comparison  (fold {EVAL_FOLD}, N=1,000 queries)",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlim(-0.55, n_groups - 0.45)
ax.set_ylim(0, max(all_totals) * 1.25)

ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# ── 범례 ─────────────────────────────────────────────────────────────────────
patch_search = mpatches.Patch(color=COLOR_SEARCH, label="T_search  (PQ / HNSW Search)")
patch_mlp    = mpatches.Patch(color=COLOR_MLP,    label="T_mlp     (Residual + Re-ranking MLP)")
patch_io     = mpatches.Patch(color=COLOR_IO,     label="T_IO      (L2 Re-ranking)")
ax.legend(handles=[patch_search, patch_mlp, patch_io],
          fontsize=9.5, loc="upper right",
          framealpha=0.88, edgecolor="#cccccc")

# ── 저장 ─────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\n✓ Saved: {OUTPUT_PNG}")
plt.close()
