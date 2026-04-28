#!/usr/bin/env python3
"""Generate performance comparison charts for the 3D vision book.
Usage: python assets/diagrams/python/performance_charts.py
Output: assets/diagrams/python/*.svg
"""
import matplotlib
matplotlib.use("SVG")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "figure.dpi": 150,
    "svg.fonttype": "none",
})

# --- Chart 1: Monocular Depth - Performance over time ---
fig, ax = plt.subplots(figsize=(8, 4))
years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
delta1 = [0.60, 0.63, 0.67, 0.71, 0.75, 0.78, 0.81, 0.84, 0.87, 0.89, 0.92]
methods = ["Eigen", "", "", "MonoDepth", "", "", "MiDaS v2", "DPT", "MiDaS v3", "ZoeDepth", "Depth Anything"]

ax.plot(years, delta1, "o-", color="#2196F3", linewidth=2, markersize=8)
for i, m in enumerate(methods):
    if m:
        ax.annotate(m, (years[i], delta1[i]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color="#333")

ax.set_xlabel("Year")
ax.set_ylabel("delta < 1.25 (KITTI)")
ax.set_title("Monocular Depth Estimation Progress (2014-2024)")
ax.set_ylim(0.5, 1.0)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "mono_depth_progress.svg")
plt.close(fig)

# --- Chart 2: Stereo Matching - Speed vs Accuracy ---
fig, ax = plt.subplots(figsize=(8, 5))
methods_s = ["PSMNet", "GANet", "LEAStereo", "CREStereo", "RAFT-Stereo",
             "IGEV-Stereo", "Selective-IGEV", "DLNR", "FoundationStereo"]
speed = [0.41, 1.6, 0.30, 0.41, 0.21, 0.33, 0.20, 0.60, 0.15]  # seconds
error = [1.09, 0.84, 0.78, 0.70, 0.63, 0.55, 0.56, 0.67, 0.33]  # AvgError px

colors = ["#F44336"]*4 + ["#FF9800"]*4 + ["#4CAF50"]
ax.scatter(speed, error, c=colors, s=100, edgecolors="k", linewidth=0.5, zorder=3)
for i, m in enumerate(methods_s):
    offset = (8, 8) if i != 8 else (8, -12)
    ax.annotate(m, (speed[i], error[i]), textcoords="offset points",
                xytext=offset, fontsize=8)

ax.set_xlabel("Inference Time (s)", fontsize=12)
ax.set_ylabel("AvgError (px)", fontsize=12)
ax.set_title("Stereo Matching: Speed vs Accuracy (Middlebury)", fontsize=14)
ax.grid(True, alpha=0.3)
# legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#F44336", label="2018-2022 (CNN)"),
    Patch(facecolor="#FF9800", label="2022-2024 (Iterative)"),
    Patch(facecolor="#4CAF50", label="2025 (Foundation)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(OUT / "stereo_speed_vs_accuracy.svg")
plt.close(fig)

print(f"Charts saved to {OUT}")
for f in sorted(OUT.glob("*.svg")):
    print(f"  {f.name} ({f.stat().st_size:,} bytes)")
