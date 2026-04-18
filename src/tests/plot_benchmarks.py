#!/usr/bin/env python3
"""
PIM-DGP PageRank Benchmark Plots (Clean Version)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── Config ──────────────────────────────────────────────────────────────────
CSV_FILE = "benchmark_results.csv"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

DATASETS = {
    "cit-HepPh":  {"color": "#534AB7", "marker": "o", "label": "cit-HepPh"},
    "web-Google": {"color": "#1D9E75", "marker": "s", "label": "web-Google"},
    "roadNet-CA": {"color": "#D85A30", "marker": "^", "label": "roadNet-CA"},
}

# HBM2 peak bandwidth per stack (8 channels × 128 GB/s = ~256 GB/s per stack,
# 64-channel config = ~2 TB/s theoretical; use 307 GB/s per Samsung HBM2 spec)
# For a 64-channel HBM2 system: peak ≈ 307 GB/s × 8 stacks-equivalent ≈ varies.
# Use measured ceiling as reference (CPU hits ~711 GB/s in large N runs).
HBM2_PEAK_GBPS = 720.0  # empirical ceiling from CPU dense streaming results

# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)
sizes = sorted(df["N"].unique())
x_pos = np.arange(len(sizes))
bar_width = 0.22


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Speedup (CLEANED)
# ═══════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(7, 4.5))

for i, (ds_name, cfg) in enumerate(DATASETS.items()):
    subset = df[df["dataset"] == ds_name].sort_values("N")
    vals = [subset[subset["N"] == n]["speedup"].values[0] for n in sizes]

    ax1.bar(
        x_pos + i * bar_width, vals, bar_width,
        label=cfg["label"],
        color=cfg["color"],
        edgecolor="white",
        linewidth=0.5
    )

ax1.set_xlabel("Subgraph size (N)")
ax1.set_ylabel("PIM cycle speedup (×)")
ax1.set_title("PIM vs CPU speedup")
ax1.set_xticks(x_pos + bar_width)
ax1.set_xticklabels([f"N={n}" for n in sizes])
ax1.legend(loc="upper left", framealpha=0.9)
ax1.set_ylim(0, max(df["speedup"]) * 1.15)

# Key takeaway annotation
max_speedup = df["speedup"].max()
# ax1.text(
#     0.5, 0.92,
#     f"Up to {max_speedup:.1f}× speedup",
#     transform=ax1.transAxes,
#     ha="center",
#     fontsize=11,
#     fontweight="bold"
# )

fig1.tight_layout()
fig1.savefig(f"{OUTPUT_DIR}/fig1_speedup.png")
fig1.savefig(f"{OUTPUT_DIR}/fig1_speedup.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Data Movement (IMPROVED — log scale)
# ═══════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for ax, (ds_name, cfg) in zip(axes, DATASETS.items()):
    subset = df[df["dataset"] == ds_name].sort_values("N")
    cpu_mb = subset["cpu_data_mb"].values
    pim_mb = subset["pim_data_mb"].values
    x = np.arange(len(sizes))

    ax.bar(x - 0.18, cpu_mb, 0.35, label="CPU", color="#888780")
    ax.bar(x + 0.18, pim_mb, 0.35, label="PIM", color=cfg["color"])

    # Annotate only largest N (reduce clutter)
    j = len(sizes) - 1
    ratio = cpu_mb[j] / pim_mb[j]
    # ax.text(
    #     x[j] + 0.18,
    #     pim_mb[j] * 1.3,
    #     f"{ratio:.0f}×",
    #     ha="center",
    #     fontsize=9,
    #     fontweight="bold",
    #     color=cfg["color"]
    # )

    ax.set_title(cfg["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("N")
    ax.set_yscale("log")

axes[0].set_ylabel("Data moved (MB, log scale)")
axes[0].legend(loc="upper left")

fig2.suptitle("Memory traffic: CPU vs PIM", fontsize=13)
fig2.tight_layout()
fig2.savefig(f"{OUTPUT_DIR}/fig2_data_moved.png")
fig2.savefig(f"{OUTPUT_DIR}/fig2_data_moved.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Data Reduction (IMPROVED)
# ═══════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(6, 4))

for ds_name, cfg in DATASETS.items():
    subset = df[df["dataset"] == ds_name].sort_values("N")
    ax3.plot(
        subset["N"], subset["data_reduction"],
        marker=cfg["marker"],
        color=cfg["color"],
        label=cfg["label"],
        linewidth=2,
        markersize=7
    )

# Reference line (no improvement)
ax3.axhline(1, linestyle="--", color="gray", alpha=0.6)

ax3.set_xlabel("Subgraph size (N)")
ax3.set_ylabel("Data reduction (CPU / PIM)")
ax3.set_title("Memory traffic reduction")
ax3.set_xscale("log", base=2)
ax3.set_xticks(sizes)
ax3.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax3.set_ylim(1, max(df["data_reduction"]) * 1.1)
ax3.legend(loc="upper left")

fig3.tight_layout()
fig3.savefig(f"{OUTPUT_DIR}/fig3_data_reduction.png")
fig3.savefig(f"{OUTPUT_DIR}/fig3_data_reduction.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Iterations (FINAL CLEAN VERSION)
# ═══════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(7, 4.5))

for ds_name, cfg in DATASETS.items():
    subset = df[df["dataset"] == ds_name].sort_values("N")

    x = subset["N"].values
    cpu_it = subset["cpu_iters"].values
    pim_it = subset["pim_iters"].values

    color = cfg["color"]

    # CPU = same color but faded + dashed
    ax4.plot(x, cpu_it,
             linestyle="--",
             marker="o",
             color=color,
             alpha=0.35,
             linewidth=1.8,
             label=f"{cfg['label']} (CPU)")

    # PIM = bold solid
    ax4.plot(x, pim_it,
             linestyle="-",
             marker="o",
             color=color,
             linewidth=2.5,
             label=f"{cfg['label']} (PIM)")

# Labels
ax4.set_xlabel("Subgraph size (N)")
ax4.set_ylabel("Iterations")
ax4.set_title("Convergence Comparison (CPU vs PIM)")

ax4.set_xticks(sizes)

# Grid
ax4.grid(True, linestyle="--", alpha=0.4)

# Clean spines
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# 🔥 Move legend OUTSIDE (no overlap)
ax4.legend(loc="upper center",
           bbox_to_anchor=(0.5, 1.25),
           ncol=3,
           frameon=False,
           fontsize=9)

fig4.tight_layout()
fig4.savefig(f"{OUTPUT_DIR}/fig4_iterations.png", dpi=300, bbox_inches="tight")
fig4.savefig(f"{OUTPUT_DIR}/fig4_iterations.pdf", bbox_inches="tight")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Roofline / Bandwidth Utilization (REFINED)
# ═══════════════════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(7.5, 5))

# Average across datasets
agg = df.groupby("N").agg({
    "cpu_data_mb": "mean",
    "pim_data_mb": "mean",
    "cpu_bw_gbps": "mean",
    "pim_bw_gbps": "mean",
}).reset_index().sort_values("N")

# Stronger contrasting colors
CPU_COLOR = "#4C72B0"   # clean blue
PIM_COLOR = "#DD8452"   # strong orange

# HBM2 peak line (more subtle but visible)
ax5.axhline(
    HBM2_PEAK_GBPS,
    linestyle="--",
    color="black",
    alpha=0.6,
    linewidth=1.8,
    zorder=1
)
ax5.text(
    agg["cpu_data_mb"].max() * 1.4,
    HBM2_PEAK_GBPS * 1.02,
    f"HBM2 peak (~{HBM2_PEAK_GBPS:.0f} GB/s)",
    ha="right",
    va="bottom",
    fontsize=10,
    color="black",
)

# CPU trajectory (hollow markers + dashed)
ax5.plot(
    agg["cpu_data_mb"], agg["cpu_bw_gbps"],
    linestyle="--",
    color=CPU_COLOR,
    linewidth=2.2,
    alpha=0.9,
    zorder=2,
)
ax5.scatter(
    agg["cpu_data_mb"], agg["cpu_bw_gbps"],
    s=140,
    marker="o",
    facecolors="white",
    edgecolors=CPU_COLOR,
    linewidths=2.5,
    label="CPU (avg)",
    zorder=3,
)

# PIM trajectory (bold solid)
ax5.plot(
    agg["pim_data_mb"], agg["pim_bw_gbps"],
    linestyle="-",
    color=PIM_COLOR,
    linewidth=2.8,
    zorder=2,
)
ax5.scatter(
    agg["pim_data_mb"], agg["pim_bw_gbps"],
    s=140,
    marker="o",
    color=PIM_COLOR,
    edgecolors="white",
    linewidths=1.5,
    label="PIM (avg)",
    zorder=3,
)

# Annotate N values in black; move CPU labels below the line for N >= 512
# so they don't collide with the HBM2 peak line.
for _, row in agg.iterrows():
    n_val = int(row["N"])
    cpu_offset = (6, -14) if n_val >= 512 else (6, 8)
    ax5.annotate(
        f"N={n_val}",
        xy=(row["cpu_data_mb"], row["cpu_bw_gbps"]),
        xytext=cpu_offset,
        textcoords="offset points",
        fontsize=9,
        color="black",
    )
    ax5.annotate(
        f"N={n_val}",
        xy=(row["pim_data_mb"], row["pim_bw_gbps"]),
        xytext=(6, -10),
        textcoords="offset points",
        fontsize=9,
        color="black",
    )

# Axes
ax5.set_xscale("log")
ax5.set_xlabel("Data moved per run (MB, log scale)")
ax5.set_ylabel("Achieved memory bandwidth (GB/s)")
ax5.set_title("Data movement comparision: CPU vs PIM (Roofline)")

ax5.set_ylim(0, HBM2_PEAK_GBPS * 1.15)

# Legend (cleaner)
ax5.legend(loc="lower right", frameon=False, fontsize=10)

# Annotation boxes (more contrast, less gray)
# ax5.text(
#     0.02, 0.28,
#     "PIM:\n10–100× less data\nmoderate bandwidth",
#     transform=ax5.transAxes,
#     ha="left",
#     va="top",
#     fontsize=9.5,
#     bbox=dict(boxstyle="round,pad=0.4",
#               facecolor="white",
#               edgecolor=PIM_COLOR,
#               linewidth=1.5)
# )
# ax5.text(
#     0.98, 0.82,
#     "CPU:\napproaches HBM2 peak\n→ bandwidth-bound",
#     transform=ax5.transAxes,
#     ha="right",
#     va="top",
#     fontsize=9.5,
#     bbox=dict(boxstyle="round,pad=0.4",
#               facecolor="white",
#               edgecolor=CPU_COLOR,
#               linewidth=1.5)
# )

# Cleanup
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.grid(True, linestyle="--", alpha=0.35)

fig5.tight_layout()
fig5.savefig(f"{OUTPUT_DIR}/fig5_roofline.png", dpi=300, bbox_inches="tight")
fig5.savefig(f"{OUTPUT_DIR}/fig5_roofline.pdf", bbox_inches="tight")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Cycle breakdown — where do the cycles go?
#
# For each (dataset, N), show four bars: CPU-read, CPU-write, PIM-read,
# PIM-write. Cycles are attributed to each category proportionally to its
# transaction share:
#     read_cycles  = total_cycles × reads / (reads + writes)
#     write_cycles = total_cycles × writes / (reads + writes)
# Grouped (not stacked) so that tiny CPU writes remain visible on log scale —
# a stacked bar hides write slices smaller than a grid tick.
# ═══════════════════════════════════════════════════════════════════════════
fig6, ax6 = plt.subplots(figsize=(12, 5))

READ_COLOR  = "#4C72B0"   # blue for read traffic
WRITE_COLOR = "#DD8452"   # orange for write traffic

labels = []
cpu_read_cyc, cpu_write_cyc = [], []
pim_read_cyc, pim_write_cyc = [], []

for ds_name in DATASETS:
    subset = df[df["dataset"] == ds_name].sort_values("N")
    for _, row in subset.iterrows():
        labels.append(f"{ds_name}\nN={int(row['N'])}")

        cpu_txn = row["cpu_reads"] + row["cpu_writes"]
        pim_txn = row["pim_reads"] + row["pim_writes"]

        cpu_read_cyc.append(row["cpu_cycles"] * row["cpu_reads"] / cpu_txn)
        cpu_write_cyc.append(row["cpu_cycles"] * row["cpu_writes"] / cpu_txn)
        pim_read_cyc.append(row["pim_cycles"] * row["pim_reads"] / pim_txn)
        pim_write_cyc.append(row["pim_cycles"] * row["pim_writes"] / pim_txn)

cpu_read_cyc  = np.array(cpu_read_cyc)
cpu_write_cyc = np.array(cpu_write_cyc)
pim_read_cyc  = np.array(pim_read_cyc)
pim_write_cyc = np.array(pim_write_cyc)

n_groups = len(labels)
x = np.arange(n_groups)
bw = 0.2  # narrower so 4 bars fit per group

# Four side-by-side bars per group: CPU-R, CPU-W, PIM-R, PIM-W
ax6.bar(x - 1.5*bw, cpu_read_cyc,  bw, color=READ_COLOR,  edgecolor="white", linewidth=0.5, label="CPU: read traffic")
ax6.bar(x - 0.5*bw, cpu_write_cyc, bw, color=WRITE_COLOR, edgecolor="white", linewidth=0.5, label="CPU: write traffic")
ax6.bar(x + 0.5*bw, pim_read_cyc,  bw, color=READ_COLOR,  edgecolor="white", linewidth=0.5, hatch="//", alpha=0.85, label="PIM: read traffic")
ax6.bar(x + 1.5*bw, pim_write_cyc, bw, color=WRITE_COLOR, edgecolor="white", linewidth=0.5, hatch="//", alpha=0.85, label="PIM: write traffic")

# Total-cycle annotations above the CPU and PIM groups
# for i in range(n_groups):
#     cpu_total = cpu_read_cyc[i] + cpu_write_cyc[i]
#     pim_total = pim_read_cyc[i] + pim_write_cyc[i]
#     ax6.text(x[i] - bw, cpu_total * 1.6, f"{int(cpu_total):,}",
#              ha="center", va="bottom", fontsize=7.5, color="#333")
#     ax6.text(x[i] + bw, pim_total * 1.6, f"{int(pim_total):,}",
#              ha="center", va="bottom", fontsize=7.5, color="#333")

ax6.set_yscale("log")
ax6.set_ylabel("Simulated cycles (log scale)")
ax6.set_title("Cycle breakdown")
ax6.set_xticks(x)
ax6.set_xticklabels(labels, fontsize=8.5)

# Dataset dividers between the three groups of 4
per_ds = n_groups // len(DATASETS)
for i in range(1, len(DATASETS)):
    ax6.axvline(i * per_ds - 0.5, color="#ccc", linewidth=0.8, linestyle=":", zorder=0)

ax6.grid(True, axis="y", linestyle="--", alpha=0.4)
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)

ax6.legend(loc="upper center",
           bbox_to_anchor=(0.5, 1.22),
           ncol=4,
           frameon=False,
           fontsize=9)

# Extra headroom so total labels don't clip
ax6.set_ylim(top=ax6.get_ylim()[1] * 3)

fig6.tight_layout()
fig6.savefig(f"{OUTPUT_DIR}/fig6_cycle_breakdown.png", dpi=300, bbox_inches="tight")
fig6.savefig(f"{OUTPUT_DIR}/fig6_cycle_breakdown.pdf", bbox_inches="tight")

print(f"\nAll figures saved to {OUTPUT_DIR}/")
plt.show()