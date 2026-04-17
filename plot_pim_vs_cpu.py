import matplotlib.pyplot as plt
import numpy as np

# Data from stats_cpu_dram_simulated test (N=256, same HBM2 DRAM simulator for both)
metrics = {
    "Iterations":        {"CPU": 8,       "PIM": 11},
    "Simulated\nCycles": {"CPU": 86802,   "PIM": 48528},
    "Memory Reads\n(txns)": {"CPU": 1050624, "PIM": 115456},
    "Memory Writes\n(txns)": {"CPU": 2048,   "PIM": 744128},
    "Data Moved\n(MB)":  {"CPU": 32.12,   "PIM": 26.23},
    "BW Used\n(GB/s)":   {"CPU": 388.07,  "PIM": 566.82},
}

speedup = 86802 / 48528  # 1.79x

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("PIM vs CPU Baseline — N=256, Both Routed Through HBM2 DRAM Simulator", fontsize=12)

# ── Plot 1: Cycles bar chart ──────────────────────────────────────────────────
ax1 = axes[0]
labels = ["CPU (no PIM)", "PIM"]
cycles = [86802, 48528]
colors = ["#d62728", "#2ca02c"]
bars = ax1.bar(labels, cycles, color=colors, width=0.4)
ax1.set_ylabel("Simulated HBM2 Cycles")
ax1.set_title(f"Simulated Cycles\n(PIM is {speedup:.2f}x faster)")
ax1.set_ylim(0, max(cycles) * 1.3)

for bar, val in zip(bars, cycles):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800,
             f"{val:,}", ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.annotate('', xy=(1, 48528), xytext=(0, 86802),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax1.text(0.5, (86802 + 48528) / 2 + 2000, f"{speedup:.2f}x speedup",
         ha='center', fontsize=10, color='black')

# ── Plot 2: Memory traffic breakdown ─────────────────────────────────────────
ax2 = axes[1]
x = np.arange(2)
width = 0.3

reads  = [1050624, 115456]
writes = [2048,    744128]

b1 = ax2.bar(x - width/2, reads,  width, label="Reads (txns)",  color="#1f77b4")
b2 = ax2.bar(x + width/2, writes, width, label="Writes (txns)", color="#ff7f0e")

ax2.set_xticks(x)
ax2.set_xticklabels(["CPU (no PIM)", "PIM"])
ax2.set_ylabel("Memory Transactions")
ax2.set_title("Memory Traffic Breakdown\n(Reads vs Writes)")
ax2.legend()
ax2.set_ylim(0, max(reads) * 1.3)

for bar in b1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
             f"{int(bar.get_height()):,}", ha='center', va='bottom', fontsize=8)
for bar in b2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
             f"{int(bar.get_height()):,}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("pim_vs_cpu_cycles.png", dpi=150, bbox_inches='tight')
print("Saved: pim_vs_cpu_cycles.png")
plt.show()
