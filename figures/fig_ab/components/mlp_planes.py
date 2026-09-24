"""Three decoder layers drawn as three square temperature planes.

Every cell in the figure is the same size, so the layer sizes follow their
cell counts: the 4x4 input plane on the left is twice the edge of the 2x2
output plane on the right, with 3x3 in between.  They are fanned out to the
right, each turned so its left edge is the near edge, and each leans a little
further out of the page.

The colour is a **continuous temperature field**, not one colour per cell:
each plane carries a smooth analytic field (a few broad blobs) that is
evaluated on a fine grid and drawn as filled iso-temperature bands — the way a
global temperature map is drawn.  The 4x4 / 3x3 / 2x2 cell lines are then
drawn *on top* as thin reference lines, so they read as a grid laid over a
continuous map rather than as a mosaic of coloured tiles.

Occlusion is strict: planes are painted far-to-near with opaque fills, so a
nearer plane simply covers what is behind it.
"""
import math

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

try:                                    # imported as a package
    from .palette import *              # noqa: F401,F403
except ImportError:                     # run as a script
    from palette import *               # noqa: F401,F403

# ---------------------------------------------------------------- geometry
CELL = 0.13             # edge of one cell -- identical in every layer
Z0, DZ = 4.00, 0.14     # far camera; each next plane leans slightly nearer
OX, OY = 0.155, 0.022   # stack offset per layer: a tight deck of cards
YAW_DEG, ROLL_DEG = -42.0, -2.0
FOCAL = 1.0
LAYER_N = (8, 6, 4)     # cells per side, left (input) -> right (output)

# One-hue temperature ramp: blue -> pale blue, the whole map faded.
# The pale end is a mid blue, not near-white, so that after the fade the
# lightest band still reads as a colour instead of dissolving into the paper.
DEEP_B = "#0a3f7d"      # deep end
MID_B = "#4a90c8"       # ... middle ...
PALE_B = "#7fb0d8"      # pale end (kept off white on purpose)
FIELD_FADE = 0.50       # how far the *background* of the map sits back


def _rgba(c, a):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)) + (a,)


def _mix_white(c, f):
    """Fade towards white without using alpha.

    Alpha would let a nearer plane show through the one in front of it and
    destroy the strict occlusion, so the fade is baked into the colours.
    """
    c = c.lstrip("#")
    v = np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    v = v * (1.0 - f) + f
    return "#%02x%02x%02x" % tuple(int(round(x * 255)) for x in v)


# The whole map is faded, top to bottom: every plane is meant to sit back.
CMAP = LinearSegmentedColormap.from_list(
    "temp", [_mix_white(c, FIELD_FADE) for c in (DEEP_B, MID_B, PALE_B)])

FIELD_N = 52            # resolution the temperature field is sampled on
BANDS = 10              # iso-temperature bands: few and legible
CELL_LINE = "#ffffff"   # the reference grid, white
OUTLINE = "#4f6b7d"     # plane outline: same colour as the wires
OUTLINE_ALPHA = 0.70    # ... and the same fade
SHOW_CELLS = False      # the white reference grid (off for now)


def _edge(i):
    """Plane edge: one cell size per grid step, so 4x4 > 3x3 > 2x2."""
    return LAYER_N[i] * CELL


def _depth(i):
    """Layer i sits this far from the camera; smaller is nearer."""
    return Z0 - i * DZ


def _rotation():
    """Turned about the vertical, then rolled a touch in the picture plane.

    The yaw is negative so that each plane's -u (left) edge comes towards the
    camera: the plane leans inwards, and its left edge projects longest.
    """
    y, r = math.radians(YAW_DEG), math.radians(ROLL_DEG)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry


def _project(u, v, cx, cy, cz, rmat):
    """Local plane coords -> screen (x, y) and camera depth."""
    world = np.stack([u, v, np.zeros_like(u)], axis=-1) @ rmat.T
    world[..., 0] += cx
    world[..., 1] += cy
    world[..., 2] += cz
    depth = world[..., 2]
    return FOCAL * world[..., 0] / depth, FOCAL * world[..., 1] / depth, depth


def _as_poly(xy):
    """(xs, ys) -> list of (x, y) corners."""
    return [(float(a), float(b)) for a, b in zip(*xy)]


def _layer_quad(i, rmat):
    h = _edge(i) / 2.0
    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    x, y, _ = _project(corners[:, 0], corners[:, 1],
                       i * OX, i * OY, _depth(i), rmat)
    return [(float(a), float(b)) for a, b in zip(x, y)]


def _field(rng, k=2):
    """A smooth continuous temperature field on the unit square.

    A handful of broad blobs — their width is a good fraction of the plane, so
    the field has no cell-scale structure and the iso-band contours come out
    as long smooth curves, the way a temperature map looks.
    """
    blobs = [(rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98),
              rng.uniform(-1.0, 1.0),
              rng.uniform(0.26, 0.72) * rng.choice([0.55, 1.0]))
             for _ in range(k)]

    def f(u, v):
        u, v = np.broadcast_arrays(u, v)
        z = np.zeros(u.shape)
        for cu, cv, w, sg in blobs:
            z += w * np.exp(-(((u - cu) ** 2 + (v - cv) ** 2)
                              / (2.0 * sg ** 2)))
        return z
    return f


def draw(ax, x, y, w, h, seed=SEED, z=6, **kw):
    rng = np.random.default_rng(seed)
    rmat = _rotation()
    fields = [_field(rng, k=3) for _ in LAYER_N]

    edges = [_edge(i) for i in range(len(LAYER_N))]
    quads = [_layer_quad(i, rmat) for i in range(len(LAYER_N))]

    # fit the whole stack into the requested rectangle, aspect preserved
    allpts = np.array([p for q in quads for p in q])
    x0p, y0p = allpts.min(axis=0)
    x1p, y1p = allpts.max(axis=0)
    sx, sy = x1p - x0p, y1p - y0p
    scale = min(w / sx, h / sy)
    ox = x + (w - sx * scale) / 2.0 - x0p * scale
    oy = y + (h - sy * scale) / 2.0 - y0p * scale

    def to_data(px, py):
        return px * scale + ox, py * scale + oy

    levels = np.linspace(0.0, 1.0, BANDS + 1)
    for i in range(len(LAYER_N)):                 # 0 = farthest, painted first
        hh = edges[i] / 2.0
        u = np.linspace(-hh, hh, FIELD_N)
        v = np.linspace(-hh, hh, FIELD_N)
        uu, vv = np.meshgrid(u, v, indexing="ij")

        xs, ys, _ = _project(uu.ravel(), vv.ravel(),
                             i * OX, i * OY, _depth(i), rmat)
        gx, gy = to_data(xs.reshape(uu.shape), ys.reshape(uu.shape))

        # evaluate the continuous field, then squash it into [0, 1]
        t = fields[i]((uu + hh) / (2.0 * hh), (vv + hh) / (2.0 * hh))
        t = (t - t.min()) / (np.ptp(t) + 1e-9)
        t = 0.004 + 0.992 * t

        zc = z + 0.010 * i
        # continuous temperature map: filled iso-bands, no per-cell colouring
        ax.contourf(gx, gy, t, levels=levels, cmap=CMAP,
                    antialiased=False, zorder=zc)

        if SHOW_CELLS:
            # the reference grid, laid over the map
            n = LAYER_N[i]
            step = edges[i] / n
            for k in range(n + 1):
                tpos = -hh + k * step
                for us, vs in ((np.array([tpos, tpos]), np.array([-hh, hh])),
                               (np.array([-hh, hh]), np.array([tpos, tpos]))):
                    px, py, _ = _project(us, vs, i * OX, i * OY, _depth(i), rmat)
                    ax.plot(*to_data(px, py), color=CELL_LINE, lw=0.45,
                            zorder=zc + 0.002, solid_capstyle="butt")

        ax.add_patch(Polygon([to_data(px, py) for px, py in quads[i]],
                             closed=True, fill=False,
                             edgecolor=_rgba(OUTLINE, OUTLINE_ALPHA),
                             lw=0.9, zorder=zc + 0.004))

    return {"layers": [[to_data(px, py) for px, py in q] for q in quads]}


# ---------------------------------------------------------------- preview
if __name__ == "__main__":
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    out = Path(__file__).resolve().parent.parent / "assets"

    big = plt.figure(figsize=(4.0, 4.0))
    bx = big.add_axes([0, 0, 1, 1])
    bx.set_xlim(0, 4.0); bx.set_ylim(0, 4.0); bx.axis("off")
    draw(bx, 0.3, 0.3, 3.4, 3.4)
    big.savefig(out / "mlp_planes_preview.png", dpi=200)
    big.savefig(out / "mlp_planes_preview.svg")

    small = plt.figure(figsize=(2.4, 1.3))
    sx = small.add_axes([0, 0, 1, 1])
    sx.set_xlim(0, 2.4); sx.set_ylim(0, 1.3); sx.axis("off")
    draw(sx, 0.10, 0.10, 1.05, 0.82)
    sx.text(1.30, 0.95, "final size", fontsize=6, ha="left", va="center",
            color="#65737e")
    small.savefig(out / "mlp_planes_preview_small.png", dpi=400)
    print("wrote mlp_planes previews to", out)
