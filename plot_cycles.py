import matplotlib.pyplot as plt
import numpy as np

# Data from pim_cycle_sweep_edge_insertion test (N=64, avg_deg=6)
edges_inserted  = [1,  3,  7,  15, 31, 63]
cold_cycles     = [5140, 5140, 5140, 5140, 5140, 5140]
warm_cycles     = [2752, 2752, 2752, 2752, 2752, 2752]
cold_iters      = [2, 2, 2, 2, 2, 2]
warm_iters      = [1, 1, 1, 1, 1, 1]
reduction_pct   = [46.5] * 6

x = np.arange(len(edges_inserted))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("PIM PageRank: Cold vs Warm Restart after Edge Insertion (N=64)", fontsize=13)

# --- Plot 1: Cycle counts ---
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, cold_cycles, width, label="Cold restart", color="#d62728")
bars2 = ax1.bar(x + width/2, warm_cycles, width, label="Warm restart", color="#2ca02c")

ax1.set_xlabel("Cumulative edges inserted")
ax1.set_ylabel("Simulated PIM cycles")
ax1.set_title("Simulated Cycles: Cold vs Warm Restart")
ax1.set_xticks(x)
ax1.set_xticklabels(edges_inserted)
ax1.legend()
ax1.set_ylim(0, max(cold_cycles) * 1.3)

for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
             str(int(bar.get_height())), ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
             str(int(bar.get_height())), ha='center', va='bottom', fontsize=8)

ax1.axhline(y=np.mean(warm_cycles), color='green', linestyle='--', alpha=0.4)
ax1.axhline(y=np.mean(cold_cycles), color='red',   linestyle='--', alpha=0.4)

# --- Plot 2: Cycle reduction % ---
ax2 = axes[1]
ax2.plot(edges_inserted, reduction_pct, marker='o', color="#1f77b4", linewidth=2, markersize=7)
ax2.fill_between(edges_inserted, reduction_pct, alpha=0.15, color="#1f77b4")
ax2.set_xlabel("Cumulative edges inserted")
ax2.set_ylabel("Cycle reduction (%)")
ax2.set_title("Warm Start Cycle Reduction vs Cold Start")
ax2.set_ylim(0, 100)
ax2.set_xticks(edges_inserted)
ax2.axhline(y=46.5, color='gray', linestyle='--', alpha=0.5, label="~46.5% avg")
ax2.legend()

for xi, yi in zip(edges_inserted, reduction_pct):
    ax2.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                 xytext=(0, 8), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("pagerank_cycle_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: pagerank_cycle_comparison.png")
plt.show()
