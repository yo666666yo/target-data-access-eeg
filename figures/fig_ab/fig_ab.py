#!/usr/bin/env python3
"""Figure 1 of the paper: two stacked panels, one column wide.

a) (top)    the protocol confound: N subjects -> one trial matrix; the
            same trials go to both protocols, which feed the same decoder
            and differ only in their rules, and the two console dumps
            disagree -- mismatch.
b) (bottom) the controlled pipeline: held-out subject -> C_s / T_s ->
            decoder -> scorer on T_s; N-1 source subjects -> 80 / 20 ->
            training and checkpoint selection.

Placement contract (ICASSP 2027 kit)
------------------------------------
The figure is authored at exactly the width it is placed at -- one column,
3.39 in -- so every point size here is the point size on the page, and none
is under 9 pt.  Text is set in Times, as the paper is; the console dumps in
Courier.  `check()` fails the build if, within a panel, any label is under
9 pt, overlaps another label or a drawn component, or leaves the canvas.
An earlier layout of this figure was 4.9 in wide with 3.9-8.6 pt labels; no
placement of it could reach 9 pt, which is why the layout below is a rewrite
rather than a rescale.

Every visual element is an imported vector component under components/; the
head is a raster render of a CC BY 3.0 scan (see assets/mesh/NOTICE.md).

Run:  python figures/fig_ab/fig_ab.py
      -> manuscript/figures/figure1.pdf, figures/figure1_preview.png
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, PathPatch, FancyBboxPatch, \
    Circle
from matplotlib.path import Path as MPath
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from components import palette as P
from components import head3d, trial_heat, mlp_planes, eeg_panel

# The same font stack as figures/gen_fig_design.py, so Figure 1 and Figure 2
# are set in one face.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "Nimbus Roman",
                   "Liberation Serif", "DejaVu Serif"],
    "font.monospace": ["Courier New", "Liberation Mono", "DejaVu Sans Mono"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

COL_W = 3.39                  # one spconf column; placed at \columnwidth
FLOOR = 9.0                   # the kit's minimum, enforced by check()
FS = 9.0                      # every plain label
FS_B = 9.5                    # bold subject labels, MISMATCH!
FS_MATH = 10.5                # C_s, T_s
FS_TAG = 10.0                 # a) b)

ARROW = P.ARROW               # every connector: one blue-grey, faded
ALPHA = 0.70
INK, SUB = P.INK, P.SUB
RED, BLUE = "#b0453a", "#2f6fb5"     # the two protocols
FRAME, FRAME_BG = "#c86a5e", "#f7e7e4"
N_TRIALS = 20
HR = 0.15                     # one head radius everywhere

_AX, _TX, _KEEP = None, None, None


def _bind(ax):
    """Everything drawn from here on belongs to this panel's checks."""
    global _AX, _TX, _KEEP
    _AX, _TX, _KEEP = ax, [], []


def keep_out(x, y, w, h, name, inside=()):
    """Register a drawn component that no label may overlap, except the
    labels it is meant to contain."""
    _KEEP.append((name, (x, y, x + w, y + h), set(inside)))


def t(x, y, s, size=FS, c=INK, w="normal", ha="center", va="center", z=10):
    a = _AX.text(x, y, s, fontsize=size, color=c, fontweight=w,
                 ha=ha, va=va, zorder=z)
    _TX.append(a)
    return a


def mono(x, y, s, size=FS, c=INK, z=11):
    a = _AX.text(x, y, s, fontsize=size, color=c, family="monospace",
                 ha="left", va="center", zorder=z)
    _TX.append(a)
    return a


def arr(p, q, c=ARROW, lw=1.0, ms=6.5, z=6):
    _AX.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                  color=c, lw=lw, shrinkA=0, shrinkB=0,
                                  alpha=ALPHA, zorder=z, joinstyle="miter",
                                  capstyle="butt"))


def _rounded(pts, radius):
    """Poly-line as a Path, every interior corner softened by an arc."""
    verts, codes = [pts[0]], [MPath.MOVETO]
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        d0 = math.hypot(x0 - x1, y0 - y1)
        d1 = math.hypot(x2 - x1, y2 - y1)
        r = min(radius, 0.5 * d0, 0.5 * d1)
        verts += [(x1 + (x0 - x1) / d0 * r, y1 + (y0 - y1) / d0 * r),
                  (x1, y1),
                  (x1 + (x2 - x1) / d1 * r, y1 + (y2 - y1) / d1 * r)]
        codes += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
    verts.append(pts[-1])
    codes.append(MPath.LINETO)
    return MPath(verts, codes)


def route(points, c=ARROW, lw=1.0, radius=0.05, ms=6.5, head=True, z=4):
    """Rounded poly-line of horizontal/vertical legs; the head rides the
    last segment."""
    pts = [(float(a), float(b)) for a, b in points]
    _AX.add_patch(PathPatch(_rounded(pts, radius), fill=False, ec=c, lw=lw,
                            alpha=ALPHA, capstyle="butt", joinstyle="round",
                            zorder=z))
    if head:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        d = math.hypot(x1 - x0, y1 - y0)
        k = min(0.07, 0.6 * d) / d
        arr((x1 - (x1 - x0) * k, y1 - (y1 - y0) * k), (x1, y1), c, lw, ms,
            z=z + 1)
    return pts


def rbox(x, y, w, h, fc="white", ec=INK, lw=1.0, r=0.03, z=6, ls="-"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0,rounding_size=%g" % r,
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    _AX.add_patch(p)
    return p


def head(cx, cy, tone):
    head3d.draw(_AX, cx - HR, cy - HR, 2 * HR, 2 * HR, tone=tone)
    keep_out(cx - HR, cy - HR, 2 * HR, 2 * HR, "head")


def heads_pair(cx, cy, tone, gap=0.36):
    """Two heads with an ellipsis between them: 'N subjects'."""
    head(cx - gap / 2, cy, tone)
    head(cx + gap / 2, cy, tone)
    t(cx, cy, r"$\cdots$", FS, SUB)


def eeg(x, y, w, h, kind):
    eeg_panel.draw(_AX, x, y, w, h, kind=kind)
    keep_out(x, y, w, h, "eeg")


def decoder(x, y, w, h, seed):
    net = mlp_planes.draw(_AX, x, y, w, h, seed=seed)
    pts = [p for layer in net["layers"] for p in layer]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    keep_out(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), "decoder")
    return min(xs), max(xs), min(ys), max(ys)


def text_size_in(artist):
    """Rendered width and height of a text artist, in inches (= data units)."""
    fig = artist.figure
    fig.canvas.draw()
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    return bb.width / fig.dpi, bb.height / fig.dpi


def check(name):
    """Fail the build rather than ship a figure the kit would reject."""
    fig = _AX.figure
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    boxes = [(a, a.get_window_extent(ren)) for a in _TX if a.get_visible()]

    for a, _ in boxes:
        if a.get_fontsize() < FLOOR:
            raise AssertionError("[%s] %r is %.1f pt, under the %.0f pt floor"
                                 % (name, a.get_text(), a.get_fontsize(), FLOOR))

    def ov(a, b, pad=0.5):
        return not (a.x1 <= b.x0 + pad or b.x1 <= a.x0 + pad or
                    a.y1 <= b.y0 + pad or b.y1 <= a.y0 + pad)

    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            if ov(ba, bb):
                raise AssertionError("[%s] text collision: %r / %r"
                                     % (name, a.get_text(), b.get_text()))

    to_px = _AX.transData
    for label, (x0, y0, x1, y1), inside in _KEEP:
        (px0, py0), (px1, py1) = to_px.transform([(x0, y0), (x1, y1)])
        kb = Bbox([[px0, py0], [px1, py1]])
        for a, ba in boxes:
            if a.get_text() == r"$\cdots$" or a.get_text() in inside:
                continue          # the ellipsis sits between heads by design
            if ov(ba, kb, pad=1.0):
                raise AssertionError("[%s] %r overlaps the %s"
                                     % (name, a.get_text(), label))

    cv = _AX.get_window_extent(ren)
    for a, bb in boxes:
        if bb.x0 < cv.x0 - 1 or bb.x1 > cv.x1 + 1 or \
           bb.y0 < cv.y0 - 1 or bb.y1 > cv.y1 + 1:
            raise AssertionError("[%s] text outside canvas: %r"
                                 % (name, a.get_text()))
    print("[%s] checks passed: %d labels, all >= %.0f pt, %d components clear"
          % (name, len(boxes), FLOOR, len(_KEEP)))


# ================================================================== panel a
HA = 2.12


def panel_confound(ax):
    _bind(ax)
    W = COL_W
    ax.set_xlim(0, W)
    ax.set_ylim(0, HA)
    ax.axis("off")

    # ---------------------------------------------------- the console dumps
    # Sized from the rendered text, so the boxes fit whatever Courier the
    # machine has.  Compact JSON (no space after the brace) keeps the widest
    # line to 20 characters; two of them side by side is what the column
    # holds at 9 pt.
    DUMPS = (['{"reads": "Cs",', ' "ckpt": "src_val",', ' "split": "v1"}'],
             ['{"reads": "Cs + Ts",', ' "ckpt": "Cs",', ' "split": "redrawn"}'])
    PADX, PITCH, PADY = 0.045, 0.148, 0.032
    probe = [mono(0, 0, max(d, key=len)) for d in DUMPS]
    widths = [text_size_in(p)[0] for p in probe]
    for p in probe:
        p.remove()
        _TX.remove(p)
    BW = [w_ + 2 * PADX for w_ in widths]
    BH = 3 * PITCH + 2 * PADY
    GAP, FP = 0.07, 0.055                      # between boxes; frame padding
    span = BW[0] + GAP + BW[1] + 2 * FP
    FX0 = (W - span) / 2.0
    FX1 = FX0 + span
    BY0 = 0.245
    BOX = [(FX0 + FP, BY0, BW[0], BH),
           (FX0 + FP + BW[0] + GAP, BY0, BW[1], BH)]
    FY0, FY1 = BY0 - FP, BY0 + BH + FP

    rbox(FX0, FY0, FX1 - FX0, FY1 - FY0, FRAME_BG, FRAME, lw=1.0, r=0.07,
         z=5, ls=(0, (4, 2.4)))
    t(0.5 * (FX0 + FX1), FY0 - 0.10, "MISMATCH!", FS_B, FRAME, "bold")
    for (bx, by, bw, bh), lines in zip(BOX, DUMPS):
        rbox(bx, by, bw, bh, "#e9eef3", (0.31, 0.42, 0.49, ALPHA), lw=0.9,
             r=0.045, z=7)
        for k, s_ in enumerate(lines):
            mono(bx + PADX, by + bh - PADY - (k + 0.5) * PITCH, s_)

    # ---------------------------------------- one decoder above each dump
    DW, DH, DY0 = 0.46, 0.32, 0.86
    DCX = [bx + bw / 2.0 for bx, _, bw, _ in BOX]
    dec = [decoder(cx - DW / 2, DY0, DW, DH, seed)
           for cx, seed in zip(DCX, (20260923, 20260924))]
    for (l, r, b, tp), (bx, by, bw, bh) in zip(dec, BOX):
        route([(0.5 * (l + r), b - 0.01), (0.5 * (l + r), by + bh + 0.012)],
              z=8)
    DMID = 0.5 * (dec[0][2] + dec[0][3])
    a = t(0.5 * (DCX[0] + DCX[1]), DMID, "Decoder", FS, ARROW, "bold")
    a.set_alpha(ALPHA)

    # --------------------------------------------------- the trial matrix
    MS, MY0 = 0.58, 1.50
    MX0 = 0.5 * (DCX[0] + DCX[1]) - MS / 2.0 + 0.10
    trial_heat.draw(ax, MX0, MY0, MS, MS, n=10, rows=10, tone="amber",
                    fade_left=0.0, fade_right=0.0)
    keep_out(MX0, MY0, MS, MS, "matrix")

    # Both protocols take the same trials: one grey trunk leaves the
    # matrix and only then forks.  Had they read different slices, the two
    # outputs could differ because their inputs did, and the panel would
    # no longer isolate the protocol.  What each protocol does with the
    # trials is in its console dump.
    RUN = MY0 - 0.11
    TOP = max(d[3] for d in dec) + 0.012
    FORK = MX0 + MS / 2.0
    route([(FORK, MY0 - 0.012), (FORK, RUN)], head=False)
    route([(FORK, RUN), (DCX[0], RUN), (DCX[0], TOP)], c=RED)
    route([(FORK, RUN), (DCX[1], RUN), (DCX[1], TOP)], c=BLUE)
    # the trunk's colour as it reads on white, opaque, so the dot covers
    # the three line ends without darkening where they overlap
    ax.add_patch(Circle((FORK, RUN), 0.022, fc="#8497a4", ec="none",
                        zorder=9))
    t(DCX[0] - 0.05, 0.5 * (RUN + TOP), "protocol 1", FS, RED, "bold",
      ha="right")
    t(DCX[1] + 0.05, 0.5 * (RUN + TOP), "protocol 2", FS, BLUE, "bold",
      ha="left")

    # ------------------------------------------- subjects -> EEG -> matrix
    HCX, HCY = 0.42, MY0 + MS / 2.0 + 0.02
    heads_pair(HCX, HCY, "white")
    t(HCX, HCY - HR - 0.12, r"$N$ subjects", FS_B, INK, "bold")
    AX0, AX1 = HCX + 0.18 + HR + 0.04, MX0 - 0.015
    arr((AX0, HCY), (AX1, HCY))
    eeg(AX0 + 0.04, HCY + 0.04, AX1 - AX0 - 0.10, 0.17, "target")
    t(0.5 * (AX0 + AX1) - 0.02, HCY - 0.11, "EEG signal", FS, SUB)

    ax.text(0.02, HA - 0.08, "a)", fontsize=FS_TAG, color=INK,
            fontweight="bold", ha="left", va="center")


# ================================================================== panel b
HB = 1.94


def panel_pipeline(ax):
    _bind(ax)
    W = COL_W
    ax.set_xlim(0, W)
    ax.set_ylim(0, HB)
    ax.axis("off")

    TX0, TX1, TAPE_H = 1.38, W - 0.03, 0.28
    YT, YS = 1.52, 0.42                      # target row, source row centres

    # ------------------------------------------------ target row: C_s | T_s
    tape = trial_heat.draw(ax, TX0, YT - TAPE_H / 2, TX1 - TX0, TAPE_H,
                           n=N_TRIALS, split=0.5, rows=5, tone="cool")
    keep_out(TX0, YT - TAPE_H / 2, TX1 - TX0, TAPE_H, "target tape")
    CUT = tape["cut_x"]
    CS_X, TS_X = 0.5 * (TX0 + CUT), 0.5 * (CUT + TX1)
    t(CS_X, YT + TAPE_H / 2 + 0.12, r"available $C_s$", FS, P.BLUE, "bold")
    t(TS_X, YT + TAPE_H / 2 + 0.12, r"reserved $T_s$", FS, P.ORANGE, "bold")

    head(0.36, YT, "target")
    t(0.03, YT - HR - 0.12, r"held-out subject $s$", FS_B, INK, "bold",
      ha="left")
    AX0, AX1 = 0.36 + HR + 0.04, TX0 - 0.015
    arr((AX0, YT), (AX1, YT))
    eeg(AX0 + 0.04, YT + 0.04, AX1 - AX0 - 0.10, 0.17, "target")
    t(0.5 * (AX0 + AX1), YT - 0.11, "EEG signal", FS, SUB)

    # ------------------------------------ decoder under C_s, scorer under T_s
    DW, DH, DCY = 0.46, 0.34, 1.04
    l, r, b, tp = decoder(CS_X - DW / 2, DCY - DH / 2, DW, DH, 20260923)
    route([(CS_X, YT - TAPE_H / 2 - 0.012), (CS_X, tp + 0.012)])
    t(CS_X - 0.05, 0.5 * (YT - TAPE_H / 2 + tp), "Input", FS, INK,
      ha="right")

    SW, SH = 0.70, 0.40
    SX = TS_X - SW / 2
    rbox(SX, DCY - SH / 2, SW, SH, "#e9eef3", (0.31, 0.42, 0.49, ALPHA),
         lw=1.0, r=0.05, z=6)
    keep_out(SX, DCY - SH / 2, SW, SH, "scorer",
             inside=("score on", r"$T_s$"))
    t(TS_X, DCY + 0.085, "score on", FS, INK, "bold")
    t(TS_X, DCY - 0.095, r"$T_s$", FS_MATH, INK, "bold")
    route([(TS_X, YT - TAPE_H / 2 - 0.012), (TS_X, DCY + SH / 2 + 0.012)])

    route([(r + 0.012, DCY), (SX - 0.012, DCY)])
    t(0.5 * (r + SX), DCY + 0.10, "Decode", FS, INK)

    # ------------------------------------------ source row: training | sel.
    stape = trial_heat.draw(ax, TX0, YS - TAPE_H / 2, TX1 - TX0, TAPE_H,
                            n=N_TRIALS, split=0.8, rows=5, tone="warm")
    keep_out(TX0, YS - TAPE_H / 2, TX1 - TX0, TAPE_H, "source tape")
    SCUT = stape["cut_x"]
    t(0.5 * (TX0 + SCUT) - 0.12, YS - TAPE_H / 2 - 0.11, "training  80%", FS,
      P.BLUE, "bold")
    t(TX1, YS - TAPE_H / 2 - 0.11, "selection  20%", FS, P.ORANGE, "bold",
      ha="right")

    heads_pair(0.36, YS, "source", gap=0.34)
    t(0.03, YS - HR - 0.12, r"$N{-}1$ source subjects", FS_B, SUB, "bold",
      ha="left")
    AX0 = 0.36 + 0.17 + HR + 0.04
    arr((AX0, YS), (TX0 - 0.015, YS))
    eeg(AX0 + 0.04, YS + 0.04, TX0 - AX0 - 0.10, 0.17, "source")
    t(0.5 * (AX0 + TX0), YS - 0.11, "EEG signal", FS, SUB)

    # training feeds the decoder from below; selection joins it, routed under
    # the scorer so the two never cross
    TRAIN_X = CS_X - 0.10
    SEL_X = 0.5 * (SCUT + TX1)
    JOIN_Y = b - 0.08
    arr((TRAIN_X, YS + TAPE_H / 2 + 0.012), (TRAIN_X, b - 0.012))
    route([(SEL_X, YS + TAPE_H / 2 + 0.012), (SEL_X, JOIN_Y),
           (CS_X + 0.10, JOIN_Y), (CS_X + 0.10, b - 0.012)])
    t(TRAIN_X + 0.06, 0.5 * (YS + TAPE_H / 2 + JOIN_Y), "Train & Select", FS,
      INK, ha="left")

    ax.text(0.02, HB - 0.08, "b)", fontsize=FS_TAG, color=INK,
            fontweight="bold", ha="left", va="center")


# --------------------------------------------------------------------- main
def out_paths():
    """Inside the repository the figure goes where the paper reads it;
    anywhere else it stays next to this script."""
    repo = ROOT.parents[1]
    if (repo / "manuscript" / "figures").is_dir():
        return (repo / "manuscript" / "figures" / "figure1.pdf",
                repo / "figures" / "figure1_preview.png")
    return ROOT / "figure1.pdf", ROOT / "figure1_preview.png"


def main():
    GAP = 0.06
    H = HA + GAP + HB
    fig = plt.figure(figsize=(COL_W, H))
    axa = fig.add_axes([0, (HB + GAP) / H, 1, HA / H])
    axb = fig.add_axes([0, 0, 1, HB / H])

    # each panel is drawn and then checked while it is still the bound one
    panel_confound(axa)
    check("a")
    panel_pipeline(axb)
    check("b")

    pdf, png = out_paths()
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, dpi=300)
    print("figure: %.2f x %.2f in" % (COL_W, H))
    print("wrote", pdf)
    print("wrote", png)


if __name__ == "__main__":
    main()
