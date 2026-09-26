#!/usr/bin/env python3
"""Draw the images on the repository's front page.

    python docs/make_images.py [results/v4_stats.json]

Writes docs/img/:
  overview.png    Figure 1 of the paper, its two panels side by side
  design.png      Figure 2 of the paper
  effects.png     Table 2 drawn as a chart
  regrouping.png  the aggregation control of Section 3.4
  pages.png       the five pages of the paper

The two figures and the pages are rasterized from the committed PDFs, so they
show exactly what the paper shows.  The two charts are drawn from the
statistics bundle; without an argument it is rebuilt from the logs in
results/ into a temporary directory (the bootstrap is seeded, so the charts
come out the same every time).

Needs pdftocairo (poppler) on PATH, matplotlib and Pillow.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
PAPER = ROOT / "manuscript"

DATASETS = [("BCI IV 2a", "v4_2a"), ("BNCI2014-002", "v4_002"),
            ("PhysionetMI", "v4_physionet")]

# Every image is drawn for a 7-inch line at this density.  GitHub shows the
# README about 860 px wide, so this is twice the screen size and stays sharp
# on high-density displays.
DPI = 250
PAD = 0.14                     # inches of white around each image
RADIUS = 0.10                  # inches; rounded corners read as a card on
                               # GitHub's dark theme and vanish on the light one

# The face and colours of Figures 1 and 2, so the charts sit beside them.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "Nimbus Roman",
                   "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "savefig.facecolor": "white",
})
INK, SUB, GRID = "#1c2730", "#65737e", "#dde3e7"
SIG = "#2a8c82"               # alignment: signals read from C_s   (Figure 2)
LAB = "#b04a3a"               # supervision: labels read from C_s  (Figure 2)
NEUT = "#7b8590"              # source-only reads nothing          (Figure 2)
BOTH = "#41484f"


# ------------------------------------------------------------------ helpers

def rasterize(pdf, dpi):
    """Every page of a PDF -> list of RGB images, through pdftocairo."""
    exe = shutil.which("pdftocairo")
    if exe is None:
        raise SystemExit("pdftocairo not found; install poppler")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([exe, "-png", "-r", str(dpi), str(pdf),
                        str(Path(tmp) / "p")], check=True)
        return [Image.open(p).convert("RGB")
                for p in sorted(Path(tmp).glob("p*.png"))]


def card(img, pad=PAD, radius=RADIUS, fill=(255, 255, 255)):
    """Pad with a fill colour and round the corners (transparent outside)."""
    p, r = round(pad * DPI), round(radius * DPI)
    w, h = img.size[0] + 2 * p, img.size[1] + 2 * p
    out = Image.new("RGBA", (w, h), fill + (255,))
    out.paste(img, (p, p), img if img.mode == "RGBA" else None)
    # Draw the mask at 4x and scale down so the corners are antialiased.
    mask = Image.new("L", (4 * w, 4 * h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 4 * w - 1, 4 * h - 1),
                                           radius=4 * r, fill=255)
    out.putalpha(mask.resize((w, h), Image.LANCZOS))
    return out


def save(img, name):
    path = OUT / name
    img.save(path, optimize=True)
    print("wrote %s  %dx%d  %d kB" % (path.relative_to(ROOT), img.size[0],
                                     img.size[1], path.stat().st_size // 1024))


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def trim(img, margin=0):
    """Crop to the non-white content, keeping `margin` pixels around it."""
    a = np.asarray(img.convert("L")) < 250
    rows, cols = np.where(a.any(1))[0], np.where(a.any(0))[0]
    return img.crop((max(cols[0] - margin, 0), max(rows[0] - margin, 0),
                     min(cols[-1] + 1 + margin, img.size[0]),
                     min(rows[-1] + 1 + margin, img.size[1])))


# ------------------------------------------------------------- the figures

def overview():
    """Figure 1 is a single column, (a) over (b).  A screen is wider than it
    is tall, so the two panels are cut apart at the blank band between them
    and set side by side."""
    img = rasterize(PAPER / "figures" / "figure1.pdf", DPI)[0]
    ink = (np.asarray(img.convert("L")) < 250).any(1)
    h = len(ink)
    # the longest blank run in the middle third is the gap between panels
    best, start = (0, 0), None
    for y in range(h // 3, 2 * h // 3):
        if not ink[y]:
            start = y if start is None else start
            if y - start > best[1] - best[0]:
                best = (start, y)
        else:
            start = None
    cut = (best[0] + best[1]) // 2
    a, b = trim(img.crop((0, 0, img.size[0], cut))), \
        trim(img.crop((0, cut, img.size[0], h)))
    gap = round(0.32 * DPI)
    out = Image.new("RGB", (a.size[0] + gap + b.size[0],
                            max(a.size[1], b.size[1])), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.size[0] + gap, 0))
    save(card(out), "overview.png")


def design():
    img = rasterize(PAPER / "figures" / "figure2.pdf", DPI)[0]
    save(card(trim(img)), "design.png")


def pages():
    """The paper at a glance: every page as a sheet on a grey ground."""
    sheets = rasterize(PAPER / "main.pdf", 44)
    gap, pad = 26, 30
    w, h = sheets[0].size
    ground = (246, 248, 250)          # GitHub's own code-block grey
    out = Image.new("RGB", (len(sheets) * (w + gap) - gap + 2 * pad,
                            h + 2 * pad), ground)
    shadow = Image.new("L", out.size, 0)
    for i, _ in enumerate(sheets):
        x = pad + i * (w + gap)
        ImageDraw.Draw(shadow).rectangle((x + 2, pad + 4, x + w + 2, pad + h + 4),
                                         fill=70)
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    out = Image.composite(Image.new("RGB", out.size, (120, 130, 140)), out, shadow)
    for i, s in enumerate(sheets):
        x = pad + i * (w + gap)
        out.paste(s, (x, pad))
        ImageDraw.Draw(out).rectangle((x - 1, pad - 1, x + w, pad + h),
                                      outline=(208, 215, 222))
    # Text on a sheet this small is antialiased grey; a 256-colour palette
    # keeps it intact at under half the size.
    save(card(out, pad=0, fill=ground).quantize(256, Image.FASTOCTREE,
                                                dither=Image.NONE), "pages.png")


# --------------------------------------------------------------- the charts

def rows_of(res):
    return {r["condition"]: r for r in res["deltas"]["rows"]}


def effects(results):
    """Table 2: each procedure's paired effect against source-only, with its
    percentile interval; hollow where the Holm-corrected p does not resolve."""
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.84, bottom=0.2)
    arms = [("EA", SIG), ("SUP", LAB), ("EASUP", BOTH)]
    step = 0.25
    for i, res in enumerate(results):
        rows = rows_of(res)
        for k, (cond, colour) in enumerate(arms):
            r = rows[cond]
            d = r["delta"]
            x = i + (k - 1) * step
            y, lo, hi = (100 * d[key] for key in ("point", "lo", "hi"))
            ax.plot([x, x], [lo, hi], color=colour, lw=2.2,
                    solid_capstyle="round", zorder=2)
            style = dict(marker="o", ms=9, mew=1.6, zorder=3, mec=colour)
            if r["p_holm"] >= 0.05:
                style.update(mfc="white")
            elif cond == "EASUP":
                # half teal, half red: the badge of Figure 2's EA+SUP cell
                style.update(fillstyle="left", mfc=SIG, mfcalt=LAB)
            else:
                style.update(mfc=colour)
            ax.plot([x], [y], **style)
            ax.annotate("%+.2f" % y, (x, hi), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, color=INK)
        ax.text(i, -0.05, res["dataset"], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11, fontweight="bold", color=INK)
        ax.text(i, -0.16, "source-only acc %.3f  ·  %d subjects"
                % (res["deltas"]["reference_mean"], res["n_subjects"]),
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9.5, color=SUB)

    ax.axhline(0, color=SUB, lw=0.9, zorder=1)
    ax.set_xlim(-0.55, len(results) - 0.45)
    ax.set_ylim(-0.6, 20.5)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20]))
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_ylabel("$\\Delta$ acc vs. source-only (pp)", fontsize=10, color=INK)
    ax.tick_params(axis="y", labelsize=9.5, colors=INK, length=0, pad=4)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    for sep in (0.5, 1.5):
        ax.axvline(sep, color=GRID, lw=0.7, ls=(0, (3, 3)), zorder=0)

    handles = [
        Line2D([], [], color=SIG, lw=2.2, marker="o", ms=8, mfc=SIG, label="EA"),
        Line2D([], [], color=LAB, lw=2.2, marker="o", ms=8, mfc=LAB, label="SUP"),
        Line2D([], [], color=BOTH, lw=2.2, marker="o", ms=8, fillstyle="left",
               mfc=SIG, mfcalt=LAB, mec=BOTH, label="EA+SUP"),
        Line2D([], [], color=SUB, lw=0, marker="o", ms=8, mfc="white", mew=1.6,
               label="Holm $p \\geq 0.05$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=10, handlelength=1.6, columnspacing=2.2,
               bbox_to_anchor=(0.54, 1.0))
    save(card(trim(fig_to_image(fig))), "effects.png")


def regrouping(results):
    """Section 3.4: the between-subject SD of accuracy against the SD the same
    predictions give when their trials are dealt into random groups of the same
    sizes.  Averaged over the three runs."""
    fig, ax = plt.subplots(figsize=(7.0, 2.75))
    fig.subplots_adjust(left=0.25, right=0.97, top=0.82, bottom=0.2)
    arms = [("SRC", NEUT, "#d5dade"), ("SUP", LAB, "#ecc9c3")]
    for i, res in enumerate(results):
        sizes = sorted(set(res["regrouping"]["SRC"][0]["group_sizes"]))
        per_subject = "%d–%d" % (sizes[0], sizes[-1]) \
            if len(sizes) > 1 else "%d" % sizes[0]
        for k, (cond, colour, tint) in enumerate(arms):
            runs = res["regrouping"][cond]
            obs = np.mean([r["sd_by_true_subject"] for r in runs])
            null = np.mean([r["sd_by_random_groups_mean"] for r in runs])
            lo = np.mean([r["sd_by_random_groups_lo"] for r in runs])
            hi = np.mean([r["sd_by_random_groups_hi"] for r in runs])
            y = -i + (0.17 if k == 0 else -0.17)
            ax.plot([lo, hi], [y, y], color=tint, lw=8.5,
                    solid_capstyle="round", zorder=1)
            ax.plot([null, obs], [y, y], color=colour, lw=1.1, zorder=2)
            ax.plot([null], [y], marker="|", ms=11, mew=1.8, color=colour,
                    zorder=3)
            ax.plot([obs], [y], marker="o", ms=8.5, color=colour, zorder=3)
            ax.annotate("×%.1f" % (obs / null), (obs, y), xytext=(8, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=10, color=colour, fontweight="bold")
        ax.text(-0.02, -i + 0.1, res["dataset"], transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=11, fontweight="bold", color=INK)
        ax.text(-0.02, -i + 0.06, "%s test trials per subject" % per_subject,
                transform=ax.get_yaxis_transform(), ha="right", va="top",
                fontsize=9.5, color=SUB)

    ax.set_xscale("log")
    ax.set_xlim(0.0125, 0.3)
    ax.set_ylim(-len(results) + 0.5, 0.5)
    ax.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _p: "%g" % v))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_xlabel("between-group SD of accuracy (log scale)", fontsize=10,
                  color=INK)
    ax.tick_params(axis="x", labelsize=9.5, colors=INK, length=0, pad=5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(SUB)

    handles = [
        Line2D([], [], color="#c4cbd1", lw=8.5, solid_capstyle="round",
               label="same predictions, random groups (95% range)"),
        Line2D([], [], color=INK, lw=0, marker="o", ms=8,
               label="grouped by true subject"),
        Line2D([], [], color=NEUT, lw=5, label="SRC"),
        Line2D([], [], color=LAB, lw=5, label="SUP"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=10, handlelength=1.5, columnspacing=1.6,
               bbox_to_anchor=(0.5, 1.0))
    save(card(trim(fig_to_image(fig))), "regrouping.png")


# --------------------------------------------------------------------- main

def load_stats(argv):
    if len(argv) > 1:
        return json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "v4_stats.json"
        specs = ["%s=%s" % (name, (Path("results") / d / "*.json").as_posix())
                 for name, d in DATASETS]
        print("rebuilding the statistics bundle from results/ ...")
        subprocess.run([sys.executable, "analysis/v4_stats.py", *specs,
                        "--out", str(out)], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        return json.loads(out.read_text(encoding="utf-8"))


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    results = load_stats(argv)
    overview()
    design()
    effects(results)
    regrouping(results)
    pages()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
