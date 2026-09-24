#!/usr/bin/env python3
"""Figures 1 and 2: the target-data design.

Figure 1, one column wide: the two divisions of the data, drawn as banded
zones.  The held-out subject is cut once into an available half and a reserved
test half; the source pool is cut once into training and checkpoint-selection
trials.  Both cuts are made before any condition runs and are identical in all
four conditions.

Figure 2, the full text width: the design itself, as a 2x2 over the two
procedures.  Each factor is named on the axis state that switches it on, and
each cell states what that condition reads from $C_s$ -- which is what makes
the caveat visible, since supervision reads the same signals alignment does,
so the grid crosses procedures rather than kinds of access.  All four are
scored on the same reserved half.

Each figure is authored at exactly the width it is placed at, so nothing is
scaled and the point sizes here are also the point sizes on the page; none is
below 9 pt.  That floor is what splits this into two figures: two access pills
side by side do not fit an 86 mm column at 9 pt, so the grid cannot be a
single-column figure and the two panels cannot share one.

Icons are true vector graphics converted from the source SVG path data (arcs
approximated by cubic Beziers), so the PDFs are infinitely zoomable.
Self-contained: only matplotlib is required.

Run:  python figures/gen_fig_design.py
      -> manuscript/figures/figure2.pdf
      (Figure 1 comes from figures/fig_ab/fig_ab.py)
"""
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.transforms import Affine2D

# The paper is set in Times (spconf pulls in Nimbus Roman, the URW clone), and
# the ICASSP kit asks for Times or Computer Modern throughout, so the figures
# follow the body rather than standing apart in a sans face.  STIX is the
# Times-matched maths font, which keeps $C_s$ in a figure looking like $C_s$ in
# a sentence.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman No9 L",
                              "Nimbus Roman", "Liberation Serif",
                              "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42

# >>> ICON_SVG_DATA
ICON_SVG = {
    "preprocess": {"w": 1024.0, "h": 1024.0, "d": ["M192.333 145.826c0-44.99 35.837-81.43 79.993-81.43H752.27c44.183 0 79.994 36.44 79.994 81.43v87.936c0 59.448-25.494 115.929-69.831 154.572L620.445 512.249l141.988 123.89a205.125 205.125 0 0 1 69.831 154.572v87.959c0 44.967-35.838 81.435-79.994 81.435H272.326a79.275 79.275 0 0 1-56.573-23.855 82.164 82.164 0 0 1-23.42-57.58v-87.933c0-59.421 25.521-115.876 69.83-154.571l141.985-123.917L262.163 388.36a205.126 205.126 0 0 1-69.83-154.598v-87.936z m559.937-1.999H272.326v89.936a123.067 123.067 0 0 0 41.929 92.743l141.958 123.89a82.05 82.05 0 0 1 27.955 61.829c0 23.775-10.217 46.347-27.955 61.828L314.255 697.994a123.07 123.07 0 0 0-41.929 92.743v88.933H752.27v-88.933a123.08 123.08 0 0 0-41.927-92.743L568.38 574.051a82.036 82.036 0 0 1-27.951-61.828c0-23.776 10.212-46.347 27.951-61.829L710.342 326.53a123.07 123.07 0 0 0 41.927-92.769v-89.934z m0 1.999", "M469.347 615.039a41.241 41.241 0 0 1 60.308 0l85.336 89.591c12.185 12.826 15.845 32.103 9.241 48.817-6.604 16.737-22.141 27.643-39.419 27.643H414.157c-17.251 0-32.788-10.906-39.419-27.643a46.432 46.432 0 0 1 9.268-48.817l85.341-89.591z m0 0"]},
    "align": {"w": 1462.0, "h": 1024.0, "d": ["M258.20947288 353.06942282h741.55596063c14.61966409 0 27.099348 5.19104004 37.40373846 15.52368213 10.41739334 10.40326789 15.58018248 22.83351381 15.58018247 37.47436499 0 14.61966409-5.16278913 27.04990928-15.58018247 37.46730263-10.30439118 10.325579-22.78407509 15.52368141-37.40373846 15.52368139H258.20947288c-14.60553864 0-27.0710971-5.1981024-37.48849044-15.52368139C210.42365363 433.11031686 205.21142577 420.68007094 205.21142577 406.06040757c0-14.64085191 5.21222786-27.0710971 15.50955667-37.47436572 10.41739334-10.33264136 22.88295181-15.52368141 37.48849044-15.52368139m0 211.85799704h953.39983222c14.60553864 0 27.0710971 5.13453896 37.48849045 15.5519323 10.2973281 10.31145354 15.50955667 22.81232527 15.50955666 37.41786463 0 14.64085191-5.21222786 27.13466127-15.50955666 37.45317718-10.41739334 10.38208008-22.88295181 15.56605776-37.48849045 15.56605703H258.20947288c-14.60553864 0-27.0710971-5.17691458-37.48849044-15.56605703C210.42365363 645.03894043 205.21142577 632.53806869 205.21142577 617.8972168c0-14.60553864 5.21222786-27.099348 15.50955667-37.41786391 10.41739334-10.41739334 22.88295181-15.5519323 37.48849044-15.55193231m0 211.88624794h741.55596063c14.61966409 0 27.099348 5.15572678 37.40373845 15.53780685 10.41739334 10.33264136 15.58018248 22.82645072 15.58018248 37.46730264 0 14.61966409-5.16278913 27.099348-15.58018248 37.40373845-10.30439118 10.41739334-22.78407509 15.56605776-37.40373845 15.56605776H258.20947288c-14.60553864 0-27.0710971-5.15572678-37.48849044-15.56605776C210.42365363 856.91812601 205.21142577 844.43844138 205.21142577 829.818778c0-14.64085191 5.21222786-27.13466127 15.50955667-37.46730263 10.41739334-10.38208008 22.88295181-15.53780686 37.48849044-15.53780685M258.20947288 141.21142578h953.39983222c14.60553864 0 27.0710971 5.16278913 37.48849045 15.5519323C1259.39512436 167.07481163 1264.60735221 179.58980881 1264.60735221 194.20947289c0 14.61966409-5.21222786 27.10641037-15.50955667 37.43198937-10.41739334 10.41033098-22.88295181 15.55899466-37.48849044 15.55899467H258.20947288c-14.60553864 0-27.0710971-5.15572678-37.48849044-15.5519323C210.42365363 221.31588327 205.21142577 208.8220739 205.21142577 194.21653526c0-14.62672646 5.21222786-27.13466127 15.50955667-37.45317718C231.13131271 146.38127801 243.60393424 141.21142578 258.20947288 141.21142578"]},
    "normalize": {"w": 1024.0, "h": 1024.0, "d": ["M731.52 885.12h219.52V138.88h-219.52V64h219.52c40.32 0 72.96 33.28 72.96 74.88v746.88c0 40.96-32.64 74.88-72.96 74.88h-219.52v-75.52zM292.48 138.88H72.96v746.88h219.52V960H72.96C32.64 960 0 926.72 0 885.12V138.88C0 97.28 32.64 64 72.96 64h219.52v74.88z m0 224c20.48 0 36.48 16.64 36.48 37.12v373.12c0 20.48-16.64 37.12-36.48 37.12a36.224 36.224 0 0 1-36.48-37.12V400c0-20.48 16.64-37.12 36.48-37.12zM512 213.12c20.48 0 36.48 16.64 36.48 37.12v522.88c0 20.48-16.64 37.12-36.48 37.12s-36.48-16.64-36.48-37.12V250.88c0-20.48 16-37.76 36.48-37.76z m219.52 224c20.48 0 36.48 16.64 36.48 37.12v298.88c0 20.48-16.64 37.12-36.48 37.12-20.48 0-36.48-16.64-36.48-37.12V474.88c0-21.12 16-37.76 36.48-37.76z"]},
    "split": {"w": 1024.0, "h": 1024.0, "d": ["M765.056 465.024h-73.024a10.432 10.432 0 0 0-10.432 10.432V548.48c0 5.76 4.672 10.432 10.432 10.432h73.024c5.76 0 10.432-4.672 10.432-10.432V475.456a10.496 10.496 0 0 0-10.432-10.432zM765.056 898.112h-73.024a10.432 10.432 0 0 0-10.432 10.432v73.024c0 5.76 4.672 10.432 10.432 10.432h73.024c5.76 0 10.432-4.672 10.432-10.432v-73.024a10.496 10.496 0 0 0-10.432-10.432zM765.056 681.536h-73.024a10.432 10.432 0 0 0-10.432 10.432v73.024c0 5.76 4.672 10.432 10.432 10.432h73.024c5.76 0 10.432-4.672 10.432-10.432v-73.024a10.496 10.496 0 0 0-10.432-10.432zM765.056 248.512h-73.024a10.432 10.432 0 0 0-10.432 10.432v73.024c0 5.76 4.672 10.432 10.432 10.432h73.024c5.76 0 10.432-4.672 10.432-10.432V258.944a10.496 10.496 0 0 0-10.432-10.432zM765.056 32h-73.024a10.432 10.432 0 0 0-10.432 10.432v73.024c0 5.76 4.672 10.432 10.432 10.432h73.024c5.76 0 10.432-4.672 10.432-10.432V42.432A10.496 10.496 0 0 0 765.056 32zM115.456 32H42.432A10.432 10.432 0 0 0 32 42.432v939.136c0 5.76 4.672 10.432 10.432 10.432h73.024c5.76 0 10.432-4.672 10.432-10.432V42.432A10.432 10.432 0 0 0 115.456 32zM332.032 32H258.944a10.432 10.432 0 0 0-10.432 10.432v939.072c0 5.76 4.672 10.432 10.432 10.432h73.024c5.76 0 10.432-4.672 10.432-10.432V42.432A10.368 10.368 0 0 0 332.032 32zM548.544 32H475.456a10.432 10.432 0 0 0-10.432 10.432v939.136c0 5.76 4.672 10.432 10.432 10.432H548.48c5.76 0 10.432-4.672 10.432-10.432V42.432A10.368 10.368 0 0 0 548.544 32zM981.568 32h-73.024a10.432 10.432 0 0 0-10.432 10.432v939.072c0 5.76 4.672 10.432 10.432 10.432h73.024c5.76 0 10.432-4.672 10.432-10.432V42.432A10.432 10.432 0 0 0 981.568 32z"]},
    "train": {"w": 1024.0, "h": 1024.0, "d": ["M830.2 638.7c-26.9 0-51.7 9.2-71.3 24.5L648.6 544.7l108.2-115.5c20 16.3 45.5 26.1 73.4 26.1 64.1 0 116-51.9 116-116s-51.9-116-116-116c-45.6 0-85.1 26.3-104 64.6L313.4 149.5c3.2-10.6 4.9-21.9 4.9-33.5 0-64.1-51.9-116-116-116s-116 51.9-116 116 51.9 116 116 116c27.6 0 52.9-9.6 72.8-25.7l208.6 224.3L313 487.9c-14.8-47.1-58.7-81.2-110.7-81.2-64.1 0-116 51.9-116 116s51.9 116 116 116c39.8 0 75-20.1 95.9-50.7l190.9 64.1-186 198.5c-20-35-57.6-58.6-100.8-58.6-64.1 0-116 51.9-116 116s51.9 116 116 116c55.7 0 102.2-39.3 113.4-91.6L721.8 796c16.7 43.6 58.9 74.7 108.4 74.7 64.1 0 116-51.9 116-116s-51.9-116-116-116z m-110.7 45.4l-153.3-51.4 52.9-56.5 100.4 107.9z m9.5-288L619.3 513.2l-58.6-63L727.8 394l1.2 2.1zM327.8 199.7l386.9 129.8c-0.3 3.3-0.4 6.6-0.4 9.9 0 4.6 0.3 9.1 0.8 13.6l-186.5 62.6-200.8-215.9z m-12.4 348.8c1.2-5.5 2.1-11.1 2.5-16.9l198-66.5 73.9 79.5-68.4 73-206-69.1z m401.7 180.1c-1.8 7.9-2.8 16.2-2.9 24.6L333.4 881.1l200.5-213.9 183.2 61.4z"]},
    "select": {"w": 1024.0, "h": 1024.0, "d": ["M458.9 421.7l129 7.3c22.6 0.1 41 18.5 41 41L465.8 633.2c-22.6-0.1-41-18.5-41-41L418 462.6c-0.2-22.5 18.3-41.1 40.9-40.9z", "M835.7 841.3c-19.5 19.5-51.2 19.5-70.7 0L452.5 528.8c-19.5-19.5-19.5-51.2 0-70.7s51.2-19.5 70.7 0l312.5 312.5c19.5 19.5 19.5 51.2 0 70.7z", "M902 119.2c17.7 17.7 27.5 41.4 27.5 66.7v164c0 19.3-15.7 35-35 35s-35-15.7-35-35V185.8c0-6.5-2.5-12.7-7-17.2-4.6-4.5-10.8-7.1-17.2-7.1H188.9c-6.5 0-12.6 2.5-17.1 7-4.7 4.7-7.3 11-7.3 17.4v652.3c0 13.4 10.9 24.3 24.4 24.3h164.6c19.3 0 35 15.7 35 35s-15.7 35-35 35H188.9c-52.1 0-94.4-42.3-94.4-94.3V185.9c0-25.1 9.8-48.7 27.5-66.6 17.8-17.9 41.5-27.8 66.9-27.8h646.4c24.8 0 49.1 10.1 66.7 27.7z"]},
}
# <<< ICON_SVG_DATA

# ---------------------------------------------------------------- SVG parsing
_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
_TOKEN = re.compile(r"([MmLlHhVvCcSsAaZz])|(" + _NUM + r")")


def _tokenize(d):
    out = []
    for m in _TOKEN.finditer(d):
        out.append(m.group(1) if m.group(1) else float(m.group(2)))
    return out


def _arc_cubics(x0, y0, rx, ry, phi_deg, laf, sf, x1, y1):
    """SVG F.6.5 endpoint->centre arc conversion, cubic Bezier approximation."""
    if (x0, y0) == (x1, y1):
        return []
    if rx == 0 or ry == 0:
        return [(x0, y0, x1, y1, x1, y1)]
    phi = math.radians(phi_deg % 360.0)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x0 - x1) / 2.0, (y0 - y1) / 2.0
    x1p = cp * dx + sp * dy
    y1p = -sp * dx + cp * dy
    rx, ry = abs(rx), abs(ry)
    lam = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry)
    if lam > 1.0:
        s = math.sqrt(lam)
        rx *= s
        ry *= s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    co = math.sqrt(max(0.0, num / den)) if den else 0.0
    if bool(laf) == bool(sf):
        co = -co
    cxp = co * rx * y1p / ry
    cyp = -co * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x0 + x1) / 2.0
    cy = sp * cxp + cp * cyp + (y0 + y1) / 2.0

    def _ang(ux, uy, vx, vy):
        dd = math.hypot(ux, uy) * math.hypot(vx, vy)
        cc = max(-1.0, min(1.0, (ux * vx + uy * vy) / dd))
        a = math.acos(cc)
        return -a if ux * vy - uy * vx < 0 else a

    th1 = _ang(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = _ang((x1p - cxp) / rx, (y1p - cyp) / ry,
               (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sf and dth > 0:
        dth -= 2.0 * math.pi
    elif sf and dth < 0:
        dth += 2.0 * math.pi
    n = max(1, int(math.ceil(abs(dth) / (math.pi / 2.0))))
    seg = dth / n
    alpha = 4.0 / 3.0 * math.tan(seg / 4.0)

    def _pt(t):
        ct, st = math.cos(t), math.sin(t)
        return (cx + rx * cp * ct - ry * sp * st,
                cy + rx * sp * ct + ry * cp * st)

    def _dpt(t):
        ct, st = math.cos(t), math.sin(t)
        return (-rx * cp * st - ry * sp * ct,
                -rx * sp * st + ry * cp * ct)

    out = []
    t0 = th1
    for _ in range(n):
        t1 = t0 + seg
        e0, e1 = _pt(t0), _pt(t1)
        d0, d1 = _dpt(t0), _dpt(t1)
        out.append((e0[0] + alpha * d0[0], e0[1] + alpha * d0[1],
                    e1[0] - alpha * d1[0], e1[1] - alpha * d1[1],
                    e1[0], e1[1]))
        t0 = t1
    return out


def _parse_d(d, vb_h):
    """Parse one SVG path string; return (verts, codes), y flipped to y-up."""
    toks = _tokenize(d)
    verts, codes = [], []
    i = 0
    cx = cy = 0.0
    sx0 = sy0 = 0.0
    last_c2 = None
    cmd = None

    def _num():
        nonlocal i
        v = toks[i]
        i += 1
        return v

    while i < len(toks):
        if isinstance(toks[i], str):
            cmd = toks[i]
            i += 1
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            x, y = _num(), _num()
            if rel:
                x += cx
                y += cy
            cx, cy = x, y
            sx0, sy0 = cx, cy
            verts.append((cx, vb_h - cy))
            codes.append(Path.MOVETO)
            cmd = "l" if rel else "L"  # implicit repeats become lineto
        elif c in ("L", "H", "V"):
            if c == "L":
                x, y = _num(), _num()
                if rel:
                    x += cx
                    y += cy
                cx, cy = x, y
            elif c == "H":
                x = _num()
                cx = cx + x if rel else x
            else:
                y = _num()
                cy = cy + y if rel else y
            verts.append((cx, vb_h - cy))
            codes.append(Path.LINETO)
        elif c in ("C", "S"):
            if c == "C":
                x1, y1, x2, y2, x, y = (_num() for _ in range(6))
                if rel:
                    x1 += cx
                    y1 += cy
                    x2 += cx
                    y2 += cy
                    x += cx
                    y += cy
            else:
                x2, y2, x, y = (_num() for _ in range(4))
                if rel:
                    x2 += cx
                    y2 += cy
                    x += cx
                    y += cy
                if last_c2 is not None:
                    x1, y1 = 2 * cx - last_c2[0], 2 * cy - last_c2[1]
                else:
                    x1, y1 = cx, cy
            verts += [(x1, vb_h - y1), (x2, vb_h - y2), (x, vb_h - y)]
            codes += [Path.CURVE4] * 3
            cx, cy = x, y
            continue
        elif c == "A":
            rx, ry, phi, laf, sf, x, y = (_num() for _ in range(7))
            if rel:
                x += cx
                y += cy
            for c1x, c1y, c2x, c2y, ex, ey in _arc_cubics(
                    cx, cy, rx, ry, phi, laf, sf, x, y):
                verts += [(c1x, vb_h - c1y), (c2x, vb_h - c2y), (ex, vb_h - ey)]
                codes += [Path.CURVE4] * 3
            cx, cy = x, y
        elif c == "Z":
            verts.append((sx0, vb_h - sy0))
            codes.append(Path.CLOSEPOLY)
            cx, cy = sx0, sy0
        else:
            raise ValueError(f"unsupported SVG command: {cmd}")
        last_c2 = None
    return verts, codes


def _icon_path(entry):
    verts, codes = [], []
    for d in entry["d"]:
        v, c = _parse_d(d, entry["h"])
        verts += v
        codes += c
    return Path(verts, codes)


ICON_PATHS = {n: (e["w"], e["h"], _icon_path(e)) for n, e in ICON_SVG.items()}


import pathlib                                       # noqa: E402
from matplotlib.colors import to_rgb                 # noqa: E402


# ------------------------------------------------------------------- drawing

INK, SUB = "#1a1a1a", "#4d4d4d"
FLOW, GRAY, BORDER = "#3d3d3d", "#68727e", "#9aa0a6"
SIG = "#2a8c82"                 # signals read from C_s, no labels
LAB = "#b04a3a"                 # labels read from C_s
NEUT = "#7b8590"                # nothing read from C_s
GREEN = "#3a7d44"
YEL_BG, YEL_DK, YEL_EC = "#fbf6ed", "#8f6a10", "#e6dabe"
GRN_BG, GRN_DK, GRN_EC = "#eef5ee", "#2f6a38", "#d3e3d4"


def tint(colour, a):
    """The colour laid over white at fraction ``a`` -- a printable pale fill."""
    r, g, b = to_rgb(colour)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def rbox(x, y, w, h, fc, ec, lw=1.0, rs=0.045, ls="-", z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
                       fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def text(x, y, s, size, color=INK, weight="normal", ha="center", va="center",
         z=6):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z)


def text_width(s, size, weight="normal"):
    """Rendered width of a string in data units (1 unit = 1 inch)."""
    t = ax.text(0, -100, s, fontsize=size, fontweight=weight)
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi


def icon_v(name, cx, cy, half_h, color, z=8):
    """Draw a vector icon centred at (cx, cy); all icons fit one height."""
    vbw, vbh, path = ICON_PATHS[name]
    s = 2 * half_h / vbh
    hw = half_h * vbw / vbh
    t = Affine2D().scale(s).translate(cx - hw, cy - half_h) + ax.transData
    ax.add_patch(PathPatch(path, transform=t, fc=color, ec="none", zorder=z))


def icon_group_width(name, label, fs, ih, gap=0.05):
    vbw, vbh, _ = ICON_PATHS[name]
    return 2 * ih * vbw / vbh + gap + text_width(label, fs, "bold")


def icon_title(x, y, name, label, colour, fs=6.0, ih=0.058, gap=0.05):
    """Icon-plus-bold-title group, laid out left to right from x."""
    vbw, vbh, _ = ICON_PATHS[name]
    ihw = ih * vbw / vbh
    icon_v(name, x + ihw, y, ih, colour)
    text(x + 2 * ihw + gap, y, label, fs, color=colour, weight="bold",
         ha="left")


def flow_arrow(x1, x2, y, color=FLOW, ms=8, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0.5, shrinkB=0.5, zorder=4))


def pill(x, y, label, colour, fs=5.2, pad=0.055, h=0.132):
    """A small outlined chip; the unit in which access to C_s is stated."""
    w = text_width(label, fs) + 2 * pad
    rbox(x, y - h / 2, w, h, tint(colour, 0.14), colour, lw=0.7, rs=0.042, z=5)
    text(x + w / 2, y, label, fs, color=colour, z=7)
    return w


# ---------------- canvases ----------------
# Two figures, authored at their exact placed width so that no scaling happens
# and the sizes below are also the sizes on the page: the divisions at one
# ICASSP column (86 mm), the design grid at the full text width (178 mm).  The
# kit allows nothing under 9 pt, and that is what forces the split -- two pills
# side by side do not fit an 86 mm column at 9 pt, so the grid cannot be a
# single-column figure and the two panels cannot share one.
COL_W, FULL_W = 3.39, 7.00

FS_TITLE, FS_SUB = 9.5, 9.0        # zone heading / its one line of detail
FS_BAR, FS_BARSM = 9.5, 9.0        # division-bar labels
FS_HEAD, FS_TICK = 10.0, 9.5       # grid factor headers / axis states
FS_BADGE, FS_PILL = 10.0, 9.0      # condition name / what it reads
FS_BOX, FS_BOXSM = 10.5, 9.0       # scoring box
FS_NOTE = 9.0                      # the caveat under the grid


def new_canvas(w, h):
    """Start a fresh figure; the drawing helpers read fig/ax as globals."""
    global fig, ax
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")


def save(name):
    root = pathlib.Path(__file__).resolve().parents[1]
    out = root / "manuscript" / "figures" / (name + ".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    png = root / "figures" / (name + "_preview.png")
    fig.savefig(png, dpi=230)
    print("[saved]", out)
    print("[saved]", png)


# ================= figure 1 =================
# Figure 1 is drawn by figures/fig_ab/fig_ab.py (two stacked panels, one
# column). The two-division diagram that used to be here is retired; do
# not reinstate a save("figure1") in this script, or it will overwrite
# that figure.

# ================= figure 2: the 2x2 over the two procedures =================
new_canvas(FULL_W, 1.95)

LEFT = 0.05
TICKR = 1.06                        # row labels are right-aligned to here
TICKY = 1.78                        # the column labels
CW, CH = 1.85, 0.56
CX = [1.18, 3.09]                   # not aligned / aligned
CY = [1.03, 0.33]                   # not supervised / supervised
CXM = [x + CW / 2 for x in CX]
CYM = [y + CH / 2 for y in CY]
BANDPAD = 0.05
GXR = CX[1] + CW + BANDPAD
EX0, EX1 = 5.14, 6.95

# Each factor's "on" half is a tinted band running the length of its row or
# column.  The fills are translucent rather than opaque so that the square both
# bands cross -- EA+SUP -- shows both tints instead of whichever was drawn
# second.
rbox(CX[1] - BANDPAD, CY[1] - BANDPAD, GXR - CX[1] + BANDPAD,
     1.90 - CY[1] + BANDPAD, (*to_rgb(SIG), 0.085), (*to_rgb(SIG), 0.36),
     lw=0.8, rs=0.06, z=0)
rbox(LEFT, CY[1] - BANDPAD, GXR - LEFT, CH + 2 * BANDPAD,
     (*to_rgb(LAB), 0.085), (*to_rgb(LAB), 0.36), lw=0.8, rs=0.06, z=0)

# Each factor is named once, on the axis state that switches it on, carrying
# its icon: a separate header block and a bare "aligned"/"supervised" tick
# said the same thing twice.  The off states stay plain grey text, so which
# half of each factor a cell sits in is still readable at a glance.
text(CXM[0], TICKY, "not aligned", FS_TICK, color=GRAY, z=5)
_hw = icon_group_width("align", "aligned on $C_s$", FS_TICK, 0.072)
icon_title(CXM[1] - _hw / 2, TICKY, "align", "aligned on $C_s$", SIG,
           fs=FS_TICK, ih=0.072)

text(TICKR, CYM[0], "not supervised", FS_TICK, color=GRAY, ha="right", z=5)
# Two lines: at 9.5 pt "supervised on $C_s$" plus its icon is half again as
# wide as the label gutter, and widening the gutter would cost the cells the
# room the "reads:" pills need.
_sw = icon_group_width("train", "supervised", FS_TICK, 0.072)
icon_title(TICKR - _sw, CYM[1] + 0.085, "train", "supervised", LAB,
           fs=FS_TICK, ih=0.072)
text(TICKR, CYM[1] - 0.085, "on $C_s$", FS_TICK, color=LAB, weight="bold",
     ha="right")

# (row, col) -> name, badge colours, what it reads from C_s
CELLS = {
    (0, 0): ("SRC", (NEUT, None), [("none", NEUT)]),
    (0, 1): ("EA", (SIG, None), [("signals", SIG)]),
    (1, 0): ("SUP", (LAB, None), [("signals", SIG), ("labels", LAB)]),
    (1, 1): ("EA+SUP", (SIG, LAB), [("signals", SIG), ("labels", LAB)]),
}

for (i, j), (name, (c1, c2), reads) in CELLS.items():
    x0, y0 = CX[j], CY[i]
    rbox(x0, y0, CW, CH, "white", BORDER, lw=0.9, rs=0.04, z=2)
    ix = x0 + 0.10

    bw = text_width(name, FS_BADGE, "bold") + 0.18
    badge = rbox(ix, y0 + 0.31, bw, 0.20, c1, "none", rs=0.035, z=5)
    if c2 is not None:
        # EA+SUP is one badge in two colours: the condition is both procedures,
        # and a single flat colour would invent a fifth category for it.
        half = Rectangle((ix + bw / 2, y0 + 0.31), bw / 2, 0.20, fc=c2,
                         ec="none", zorder=6)
        ax.add_patch(half)
        half.set_clip_path(badge)
    text(ix + bw / 2, y0 + 0.41, name, FS_BADGE, color="white", weight="bold",
         z=7)

    px = ix
    text(px, y0 + 0.15, "reads:", FS_PILL, color=GRAY, ha="left")
    px += text_width("reads:", FS_PILL) + 0.05
    for lbl, colour in reads:
        px += pill(px, y0 + 0.15, lbl, colour, fs=FS_PILL, pad=0.06,
                   h=0.17) + 0.04

text((LEFT + GXR) / 2, 0.10,
     "supervision uses the same $C_s$ signals as alignment", FS_NOTE,
     color=SUB)

# ---------------- the single scoring rule ----------------
rbox(EX0, CY[1], EX1 - EX0, CY[0] + CH - CY[1], "white", GREEN, lw=1.3,
     rs=0.05, z=2)
ecx = (EX0 + EX1) / 2
bcy = 1.28                           # centre of the badge, and of its tick
rbox(EX0 + 0.33, bcy - 0.115, 1.15, 0.23, GREEN, "none", rs=0.03, z=5)
ax.plot([EX0 + 0.455, EX0 + 0.490], [bcy + 0.005, bcy - 0.030], color="white",
        lw=1.1, zorder=7, solid_capstyle="round")
ax.plot([EX0 + 0.490, EX0 + 0.575], [bcy - 0.030, bcy + 0.055], color="white",
        lw=1.1, zorder=7, solid_capstyle="round")
text(EX0 + 0.615, bcy, "IDENTICAL", FS_BOXSM, color="white", weight="bold",
     ha="left", z=7)
text(ecx, 1.02, "Scored on $T_s$", FS_BOX, weight="bold")
text(ecx, 0.82, "the same reserved trials,", FS_BOXSM, color=SUB)
text(ecx, 0.68, "in the same order,", FS_BOXSM, color=SUB)
text(ecx, 0.54, "in all four conditions", FS_BOXSM, color=SUB)
flow_arrow(GXR, EX0, CYM[0], ms=9)
flow_arrow(GXR, EX0, CYM[1], ms=9)

save("figure2")
