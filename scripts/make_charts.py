"""Build README charts from results/sweeps-2026-11-23.csv. Run: python scripts/make_charts.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "results" / "sweeps-2026-11-23.csv")

# One color per config, same in every chart (Okabe-Ito, colorblind safe).
CONFIGS = {
    "1.5b-fp16-1gpu": ("1.5B fp16, 1 T4", "#0072B2"),
    "7b-fp16-tp2": ("7B fp16, TP=2 on 2 T4", "#D55E00"),
    "7b-awq-1gpu": ("7B AWQ, 1 T4", "#009E73"),
}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e5e5", "axes.axisbelow": True,
})
DPI = 200


def line_chart(ax, y, annotate):
    for key, (label, color) in CONFIGS.items():
        d = df[df.config == key].sort_values("concurrency")
        x = d.output_tok_s if annotate else d.concurrency
        ax.plot(x, d[y], marker="o", color=color, label=label, linewidth=2, markersize=6)
        if annotate:
            for xi, yi, c in zip(x, d[y], d.concurrency):
                ax.annotate(f"c={c}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color="#444")


# 1. latency vs throughput
fig, ax = plt.subplots(figsize=(7.5, 4.8))
line_chart(ax, "tpot_median_ms", True)
ax.set_xscale("log")
ax.set_xticks([30, 50, 100, 200, 400, 650])
ax.set_xticklabels(["30", "50", "100", "200", "400", "650"])
ax.minorticks_off()
ax.set_xlabel("Output throughput (tokens/s, log scale)")
ax.set_ylabel("TPOT, median (ms)")
ax.set_title("Latency against throughput (c = concurrency)", loc="left")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "charts" / "latency-throughput.png", dpi=DPI)

# 2. TTFT p99
fig, ax = plt.subplots(figsize=(7.5, 4.8))
line_chart(ax, "ttft_p99_ms", False)
ax.set_xscale("log", base=2)
ax.set_xticks([1, 4, 16, 64])
ax.set_xticklabels(["1", "4", "16", "64"])
ax.minorticks_off()
ax.set_yscale("log")
ax.set_xlabel("Concurrent requests")
ax.set_ylabel("TTFT, p99 (ms, log scale)")
ax.set_title("Tail time to first token against load", loc="left")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "charts" / "ttft-p99.png", dpi=DPI)

# 3. memory per GPU
mem = {
    "1.5B fp16, 1 T4": (2.98, 8.97, 1.15, "#0072B2"),
    "7B fp16, TP=2 (per GPU)": (7.16, 4.27, 1.67, "#D55E00"),
    "7B AWQ, 1 T4": (5.29, 6.02, 1.79, "#009E73"),
}
fig, ax = plt.subplots(figsize=(7.5, 3.6))
names = list(mem)[::-1]
shades = [1.0, 0.6, 0.3]  # weights dark, KV mid, other light
parts = ["Weights", "KV cache", "Other"]
for i, name in enumerate(names):
    w, kv, o, color = mem[name]
    left = 0
    for v, a, p in zip((w, kv, o), shades, parts):
        ax.barh(i, v, left=left, color=color, alpha=a, edgecolor="white", linewidth=1.5, height=0.55)
        ax.text(left + v / 2, i, f"{p}\n{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if a == 1.0 else "#222")
        left += v
ax.axvline(13.10, color="#222", linestyle="--", linewidth=1)
ax.text(13.10, 2.45, "budget 13.10 GiB", ha="right", va="bottom", fontsize=9)
ax.set_yticks(range(3))
ax.set_yticklabels(names)
ax.set_xlim(0, 14)
ax.set_ylim(-0.5, 2.8)
ax.set_xlabel("GPU memory (GiB)")
ax.set_title("Memory split per GPU", loc="left")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(ROOT / "charts" / "memory-per-gpu.png", dpi=DPI)
