"""Generate the leaderboard figures from the recorded benchmark."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(result, output):
    rows = [row for row in result["models"] if row["ranked"]]
    plt.rcParams.update({"font.size": 10, "text.color": "#172838",
                         "axes.labelcolor": "#253b4b", "xtick.color": "#253b4b",
                         "ytick.color": "#253b4b"})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.55), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1]})
    labels = [r["model"] for r in rows]
    colors = ["#116575" if r["kind"] == "Text" else "#3d7c49" for r in rows]
    for ax, key, label in [(axes[0], "f1", "Mean BERTScore F1"),
                           (axes[1], "latency_ms", "Mean generation latency (ms)")]:
        vals = [r[key] for r in rows]
        bars = ax.barh(labels, vals, color=colors, height=.54)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_xlim(0, 1.12 if key == "f1" else max(vals)*1.23)
        ax.bar_label(bars, fmt="%.3f" if key == "f1" else "%.2f", padding=4)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for extension in ("svg", "png"):
        fig.savefig(output/f"benchmark.{extension}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    plot(json.loads((directory/"results.json").read_text(encoding="utf-8")), directory)
