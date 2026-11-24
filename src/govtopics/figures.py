"""Print figures for the Journal of Information Policy.

Journal rules applied here: separate image files, TIFF or JPG, 300 dpi,
at least 2.25 in wide (we use 6.5 in, the full text width), no caption
inside the image (captions are supplied in figure_captions.txt).
Colours: sequential single-hue blue for magnitude; two fixed categorical
slots (blue, orange) for the EU and US benchmarks; India in neutral ink.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
HIGHLIGHT = {"India", "European Union", "United States"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK, "savefig.facecolor": "white",
})


def _save(fig, out: Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("tif", "jpg"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {"pil_kwargs": {"quality": 95}}
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight", **kw)
    plt.close(fig)


def similarity_heatmap(cos: pd.DataFrame, order: list[str], out: Path, stem="Figure1_similarity_heatmap"):
    M = cos.loc[order, order]
    n = len(order)
    size = 6.5
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(M.values, cmap=SEQ, vmin=max(0, np.nanmin(M.values)), vmax=1, interpolation="nearest")
    fs = max(4, min(8, 300 / max(n, 1)))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=90, fontsize=fs); ax.set_yticklabels(order, fontsize=fs)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        if lab.get_text() in HIGHLIGHT:
            lab.set_fontweight("bold"); lab.set_color(INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Cosine similarity of topic profiles", color=INK2)
    cb.outline.set_visible(False)
    _save(fig, out, stem)


def dendro(Z, labels: list[str], out: Path, stem="Figure2_dendrogram") -> list[str]:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.13 * n)))
    d = dendrogram(Z, labels=labels, orientation="left", color_threshold=0, above_threshold_color=INK2,
                   leaf_font_size=max(4, min(8, 300 / max(n, 1))), ax=ax)
    for lab in ax.get_yticklabels():
        if lab.get_text() in HIGHLIGHT:
            lab.set_fontweight("bold"); lab.set_color(INK)
    ax.set_xlabel("Jensen-Shannon distance (average linkage)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    _save(fig, out, stem)
    return d["ivl"][::-1]


GAP_SERIES = (("European Union", BLUE, "o", "vs European Union"),
              ("United States", ORANGE, "s", "vs United States"),
              ("NON_OECD_MEAN", AQUA, "D", "vs non-OECD mean"))


def gap_plot(gaps: pd.DataFrame, labels: dict[int, str], out: Path, stem="Figure3_india_gaps"):
    names = [x[0] for x in GAP_SERIES]
    g = gaps[gaps["benchmark"].isin(names) & gaps["eligible"]]
    if g.empty:
        return
    order = (g.groupby("topic")["gap"].mean().sort_values()).index.tolist()
    y = {t: i for i, t in enumerate(order)}
    fig, ax = plt.subplots(figsize=(6.5, max(2.6, 0.38 * len(order) + 0.9)))
    ax.axvline(0, color=BASE, linewidth=1)
    offs = np.linspace(-0.22, 0.22, len(GAP_SERIES))
    for (bench, color, mk, lab), off in zip(GAP_SERIES, offs):
        s = g[g["benchmark"] == bench]
        if s.empty:
            continue
        yy = [y[t] + off for t in s["topic"]]
        ax.hlines(yy, s["lo"], s["hi"], color=color, linewidth=1.6)
        ax.scatter(s["gap"], yy, s=24, color=color, marker=mk, edgecolor="white", linewidth=1.2, zorder=3, label=lab)
        for yi, x, sig in zip(yy, s["hi"], s["sig_fdr"]):
            if sig:
                ax.text(x + 0.004, yi, "*", va="center", color=INK, fontsize=8)
    clean = lambda t: str(labels.get(t, f"Topic {t}")).split(": ", 1)[-1] if str(labels.get(t, "")).startswith("T") else labels.get(t, f"Topic {t}")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([clean(t) for t in order])
    ax.set_xlabel("Benchmark share minus India share (positive = India gives the theme less weight)")
    ax.grid(axis="x", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    _save(fig, out, stem)


def instrument_plot(shares: pd.DataFrame, out: Path, stem="Figure4_instrument_mix"):
    tools = [c for c in ["authority", "treasure", "nodality", "organization", "unclassified"] if c in shares.columns]
    series = [("India", INK2, "India"), ("European Union", BLUE, "European Union"),
              ("United States", ORANGE, "United States"), ("NON_OECD_POOLED", AQUA, "Non-OECD pooled")]
    series = [x for x in series if x[0] in shares.index]
    if not series or not tools:
        return
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    w = 0.8 / len(series)
    for i, (r, color, lab) in enumerate(series):
        xs = np.arange(len(tools)) + (i - (len(series) - 1) / 2) * w
        vals = shares.loc[r, tools].values.astype(float)
        ax.bar(xs, vals, width=w - 0.02, color=color, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=5.5, color=INK2)
    ax.set_xticks(range(len(tools))); ax.set_xticklabels([t.capitalize() for t in tools])
    ax.set_ylabel("Share of unit's initiatives")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=7)
    _save(fig, out, stem)
