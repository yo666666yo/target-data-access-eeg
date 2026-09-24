"""Fig. 1 component: 3D human head seen from above (dorsal, slightly tilted)
with the 10-20 EEG electrode set drawn on the scalp.

The geometry is a real open-source mesh, not a hand-built approximation: the
Lee Perry-Smith head scan shipped with three.js (CC-BY 3.0, ir-ltd.net),
stored as ``assets/mesh/LeePerrySmith.glb`` (9,279 vertices / 17,684
triangles) and parsed here with the stdlib + numpy only.

It is rendered as a genuine surface: every triangle is z-buffer rasterised at
high resolution (3x supersampled), the interpolated vertex normals are shaded
with a Lambert + rim term, and the result is embedded as an RGBA image.  A
depth-field approximation of the surface was tried first and rejected -- it
stair-steps the silhouette and reads as a lump, not a head.

The 10-20 electrodes and their connecting traces are then drawn as *vector*
artists on top, ray-cast against the full-resolution mesh so each disc really
sits on the scalp.

    draw(ax, x, y, w, h, tone="target", show_eeg=True, seed=SEED) -> dict
"""

from __future__ import annotations

import json
import math
import os
import struct

import numpy as np
from matplotlib.patches import Circle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

try:  # package import
    from .palette import *  # noqa: F401,F403
    from .palette import SEED, BLUE, SUB
except ImportError:  # script import
    from palette import *  # noqa: F401,F403
    from palette import SEED, BLUE, SUB

# --------------------------------------------------------------------------- #
#  tunables
# --------------------------------------------------------------------------- #
MESH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir,
    "assets", "mesh", "LeePerrySmith.glb")

Y_CUT = 1.00           # cut plane: keeps the cranium, drops neck and face
TILT_DEG = 30.0        # 0 = straight down (dorsal), 90 = frontal
YAW_DEG = 0.0          # no head turn: the 10-20 circle is a dorsal view
IMAGE_PX = 900         # rendered resolution along the long axis
SUPERSAMPLE = 3        # antialiasing factor
AMBIENT = 0.26         # Lambert ambient floor
LIGHT = np.array([-0.44, 0.52, 0.73])       # camera space: upper-left, frontal
RIM = 0.20             # extra shading at grazing angles
THETA_MAX_DEG = 78.0   # radius 1 of the 10-20 circle -> this polar angle
OCCL_TOL = 0.06        # depth tolerance when testing electrode occlusion
SEAM_HALF = 22         # half-width of the hair-parting fill, in final px

# 10-20 layout in the classic normalised head circle: front = +y, subject left
# = -x, the unit circle is the head outline.
ELECTRODES = {
    "Fp1": (-0.28, 0.95), "Fp2": (0.28, 0.95),
    "F7":  (-0.72, 0.70), "F3":  (-0.38, 0.60), "Fz": (0.0, 0.56),
    "F4":  (0.38, 0.60),  "F8":  (0.72, 0.70),
    "T3":  (-0.98, 0.00), "C3":  (-0.45, 0.00), "Cz": (0.0, 0.00),
    "C4":  (0.45, 0.00),  "T4":  (0.98, 0.00),
    "T5":  (-0.72, -0.70), "P3": (-0.38, -0.60), "Pz": (0.0, -0.56),
    "P4":  (0.38, -0.60), "T6":  (0.72, -0.70),
    "O1":  (-0.28, -0.95), "O2": (0.28, -0.95),
}

_RAMP = {
    "target": ("#7d9cb7", "#c3d7e7", "#fdfeff"),
    "source": ("#8d98a0", "#ccd5da", "#fbfdfd"),
    # neutral white: for figures where the head is only a marker, not a
    # coloured object
    "white": ("#a7b4bc", "#dbe2e6", "#ffffff"),
}
_DOT = {"target": BLUE, "source": SUB, "white": SUB}
_TRACE = {"target": "#a9c9e2", "source": "#d5dee3"}


# --------------------------------------------------------------------------- #
#  tiny colour helpers
# --------------------------------------------------------------------------- #
def _rgb(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _shade_arr(ramp, t):
    """Vectorised ramp lookup: (dark, mid, light) -> RGB for t in [0, 1]."""
    lo, mid, hi = (_rgb(c) for c in ramp)
    t = np.clip(t, 0.0, 1.0)[..., None]
    low = lo + (mid - lo) * np.clip(t * 2.0, 0.0, 1.0)
    high = mid + (hi - mid) * np.clip((t - 0.5) * 2.0, 0.0, 1.0)
    return np.where(t < 0.5, low, high)


# --------------------------------------------------------------------------- #
#  glTF binary loader (stdlib + numpy)
# --------------------------------------------------------------------------- #
_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
          5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _load_glb(path):
    """Return (vertices Nx3 float64, triangles Mx3 int64) from a .glb file."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:4] != b"glTF":
        raise ValueError("not a glb file: %s" % path)
    off, js, buf = 12, None, None
    while off + 8 <= len(blob):
        clen, ctype = struct.unpack_from("<II", blob, off)
        chunk = blob[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:          # 'JSON'
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:        # 'BIN\0'
            buf = chunk
        off += 8 + clen
    if js is None or buf is None:
        raise ValueError("malformed glb")

    def accessor(idx):
        a = js["accessors"][idx]
        bv = js["bufferViews"][a["bufferView"]]
        dt = _DTYPE[a["componentType"]]
        n = _NCOMP[a["type"]]
        itemsize = np.dtype(dt).itemsize
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride") or itemsize * n
        if stride == itemsize * n:
            return np.frombuffer(buf, dtype=dt, count=a["count"] * n,
                                 offset=start).reshape(a["count"], n)
        raw = np.frombuffer(buf, dtype=np.uint8)
        idxs = (start + np.arange(a["count"])[:, None] * stride
                + np.arange(n)[None, :] * itemsize)
        return raw[idxs].copy().view(dt).reshape(a["count"], n)

    prim = js["meshes"][0]["primitives"][0]
    verts = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
    faces = accessor(prim["indices"]).astype(np.int64).reshape(-1, 3)
    return verts, faces


_MESH_CACHE = {}


def _head_mesh():
    """Head-only mesh (shoulders cropped), cached across calls."""
    if "head" in _MESH_CACHE:
        return _MESH_CACHE["head"]
    try:
        verts, faces = _load_glb(MESH_PATH)
    except Exception:
        verts, faces = _fallback_mesh()
    faces = faces[np.all(verts[faces][:, :, 1] > Y_CUT, axis=1)]
    used = np.unique(faces)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts, faces = verts[used], remap[faces]
    verts, faces = _cap_open_boundary(verts, faces)
    _MESH_CACHE["head"] = (verts, faces)
    return verts, faces


def _cap_open_boundary(verts, faces):
    """Close the rim left by the crop with a flat fan so nothing shows through.

    The scan is an open shell; slicing it leaves the interior visible from any
    viewpoint that can see under the cut.  A fan across each boundary loop makes
    the cut a solid face instead of a hole.
    """
    from collections import defaultdict

    edge_count = defaultdict(int)
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_count[(min(a, b), max(a, b))] += 1
    boundary = [e for e, n in edge_count.items() if n == 1]
    if not boundary:
        return verts, faces

    adj = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)

    loops, seen = [], set()
    for a, _ in boundary:
        if a in seen:
            continue
        loop, cur, prev = [a], a, None
        seen.add(a)
        while True:
            nxt = [x for x in adj[cur] if x != prev]
            if not nxt:
                break
            step = nxt[0]
            if step == loop[0]:
                break
            loop.append(step)
            seen.add(step)
            prev, cur = cur, step
        if len(loop) >= 3:
            loops.append(loop)
    if not loops:
        return verts, faces

    verts = verts.copy()
    extra = []
    for loop in loops:
        centre = verts[loop].mean(axis=0)
        ci = len(verts)
        verts = np.vstack([verts, centre[None, :]])
        for k in range(len(loop)):
            extra.append([loop[k], loop[(k + 1) % len(loop)], ci])
    return verts, np.vstack([faces, np.asarray(extra, dtype=faces.dtype)])


def _fallback_mesh():
    """Procedural superellipsoid bust -- only if the glb cannot be read."""
    u = np.linspace(0.0, np.pi, 46)
    v = np.linspace(0.0, 2.0 * np.pi, 92)
    U, V = np.meshgrid(u, v, indexing="ij")
    e = 0.72
    a_ = np.sin(U) ** e
    b_ = np.abs(np.cos(U)) ** e * np.sign(np.cos(U))
    verts = np.stack([(2.85 * a_ * np.sin(V) ** e).ravel(),
                      (3.05 * b_ ** 0.72).ravel(),
                      (3.60 * a_ * np.cos(V) ** e).ravel()], 1)
    idx = np.arange(len(u) * len(v)).reshape(len(u), len(v))
    a0, a1 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    a2, a3 = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    return verts, np.concatenate([np.stack([a0, a1, a2], 1),
                                  np.stack([a0, a2, a3], 1)])


# --------------------------------------------------------------------------- #
#  view
# --------------------------------------------------------------------------- #
def _view_basis(tilt_deg, yaw_deg):
    """Camera basis (right, up, towards viewer) in head coordinates.

    Head frame: +y = superior, +z = anterior (nose), x = lateral.
    """
    t = math.radians(tilt_deg)
    # the scan faces -z, so tilting towards the face means going negative in z
    zv = np.array([0.0, math.cos(t), -math.sin(t)])      # towards the viewer
    yv = np.array([0.0, math.sin(t), -math.cos(t)])      # nose -> panel top
    xv = np.cross(yv, zv)
    xv /= np.linalg.norm(xv)
    y = math.radians(yaw_deg)
    cy, sy = math.cos(y), math.sin(y)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return np.stack([xv, yv, zv]) @ ry


def _vertex_normals(verts, faces):
    """Area-weighted vertex normals."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    nrm = np.zeros_like(verts)
    for k in range(3):
        np.add.at(nrm, faces[:, k], fn)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    return nrm / np.maximum(ln, 1e-12)


# --------------------------------------------------------------------------- #
#  z-buffer surface render
# --------------------------------------------------------------------------- #
def _rasterise(scr, zbuf_in, nrm, size, ss):
    """Rasterise triangles into a depth + interpolated-normal buffer.

    ``scr`` is (M, 3, 2) pixel-space vertices, ``nrm`` (M, 3, 3) their normals.
    Returns (depth, normal, coverage) buffers at ``size`` (pre-downsample).
    """
    n = size * ss
    depth = np.full((n, n), -np.inf)
    nbuf = np.zeros((n, n, 3))
    cover = np.zeros((n, n), dtype=np.float32)

    for t in range(len(scr)):
        px, py = scr[t, :, 0], scr[t, :, 1]
        x0 = int(max(0, np.floor(px.min())))
        x1 = int(min(n - 1, np.ceil(px.max())))
        y0 = int(max(0, np.floor(py.min())))
        y1 = int(min(n - 1, np.ceil(py.max())))
        if x1 < x0 or y1 < y0:
            continue

        det = ((py[1] - py[2]) * (px[0] - px[2])
               + (px[2] - px[1]) * (py[0] - py[2]))
        if abs(det) < 1e-12:
            continue

        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        XX, YY = np.meshgrid(xs, ys)

        w0 = ((py[1] - py[2]) * (XX - px[2])
              + (px[2] - px[1]) * (YY - py[2])) / det
        w1 = ((py[2] - py[0]) * (XX - px[2])
              + (px[0] - px[2]) * (YY - py[2])) / det
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z = w0 * zbuf_in[t, 0] + w1 * zbuf_in[t, 1] + w2 * zbuf_in[t, 2]
        sub = depth[y0:y1 + 1, x0:x1 + 1]
        upd = inside & (z > sub)
        if not upd.any():
            continue

        nn = (w0[..., None] * nrm[t, 0] + w1[..., None] * nrm[t, 1]
              + w2[..., None] * nrm[t, 2])
        nsub = nbuf[y0:y1 + 1, x0:x1 + 1]
        sub[upd] = z[upd]
        nsub[upd] = nn[upd]
        cover[y0:y1 + 1, x0:x1 + 1][upd] = 1.0

    return depth, nbuf, cover


def _block_mean(a, k):
    """Downsample the leading two axes by an integer factor k."""
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    a = a[:h, :w]
    if a.ndim == 2:
        return a.reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    return a.reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))


_RENDER_CACHE = {}


def _render(tone):
    """RGBA uint8 image of the shaded head, cached per tone."""
    if tone in _RENDER_CACHE:
        return _RENDER_CACHE[tone]

    verts, faces = _head_mesh()
    origin = verts[verts[:, 1] > np.percentile(verts[:, 1], 55.0)].mean(axis=0)
    basis = _view_basis(TILT_DEG, YAW_DEG)
    vn = _vertex_normals(verts, faces)

    cam = (verts - origin) @ basis.T
    camn = vn @ basis.T
    # normals point away from the surface; the visible sheet faces the camera
    flip = np.sign(camn[:, 2])[:, None] * 0 + 1.0
    camn = camn * flip

    lo = cam[:, :2].min(axis=0)
    hi = cam[:, :2].max(axis=0)
    span = (hi - lo).max()
    pad = 0.02 * span
    lo, hi = lo - pad, hi + pad
    span = (hi - lo).max()

    n = IMAGE_PX * SUPERSAMPLE
    scale = (n - 1) / span

    def to_px(pts):
        out = np.empty((pts.shape[0], 2))
        out[:, 0] = (pts[:, 0] - lo[0]) * scale
        out[:, 1] = (n - 1) - (pts[:, 1] - lo[1]) * scale     # image y is down
        return out

    xy = cam[faces][:, :, :2]                      # (M, 3, 2)
    scr_all = to_px(xy.reshape(-1, 2)).reshape(-1, 3, 2)
    zn_all = cam[:, 2][faces]
    nrm_all = camn[faces]

    # back-face culling: on a closed surface the hidden half can never win the
    # depth test, and skipping it halves the rasteriser's work
    e1 = cam[faces[:, 1]] - cam[faces[:, 0]]
    e2 = cam[faces[:, 2]] - cam[faces[:, 0]]
    fnz = np.cross(e1, e2)[:, 2]
    keep = fnz > 0
    if not (0.2 < keep.mean() < 0.8):       # winding is inconsistent; keep all
        keep = np.ones(len(keep), dtype=bool)
    scr, zn, nrm = scr_all[keep], zn_all[keep], nrm_all[keep]

    depth, nbuf, cover = _rasterise(scr, zn, nrm, IMAGE_PX, SUPERSAMPLE)

    # view-space normal shading
    mask = depth > -np.inf
    nn = nbuf.copy()
    ln = np.linalg.norm(nn, axis=-1, keepdims=True)
    nn = nn / np.maximum(ln, 1e-12)
    lt = LIGHT / np.linalg.norm(LIGHT)
    lam = np.clip(np.einsum("ijk,k->ij", nn, lt), 0.0, None)
    rim = RIM * np.clip(1.0 - np.abs(nn[..., 2]), 0.0, 1.0)
    shade = AMBIENT + (1.0 - AMBIENT) * lam + rim

    # a faint depth cue stops the crown reading as a flat decal
    dz = depth[mask]
    if dz.size:
        shade = np.where(mask, shade * (0.94 + 0.06 * (
            (depth - dz.min()) / max(dz.max() - dz.min(), 1e-9))), 0.0)

    rgb = _shade_arr(_RAMP[tone], shade)
    # darken the outer silhouette so the form stays crisp when small
    edge = mask & ~_erode(mask, 1)
    rgb = np.where(edge[..., None], rgb * 0.78, rgb)

    img = np.zeros((n, n, 4), dtype=np.float32)
    img[..., :3] = np.clip(rgb, 0, 1)
    img[..., 3] = cover

    small = _block_mean(img, SUPERSAMPLE)

    # fill the hair parting, which renders as one dark vertical line
    inside = small[..., 3] > 0.5
    if inside.any():
        filled = _close_seam_x(small[..., :3], SEAM_HALF)
        small[..., :3] = np.where(inside[..., None], filled, small[..., :3])

    _RENDER_CACHE[tone] = small

    # remember the camera mapping so electrodes can be placed on top
    _RENDER_CACHE[tone + "_map"] = (origin, basis, lo, span)
    return small


def _window_max_x(a, half):
    """Windowed maximum along x only (used to close a vertical dark seam)."""
    out = a
    for d in range(-half, half + 1):
        if d:
            out = np.maximum(out, np.roll(a, d, axis=1))
    return out


def _window_min_x(a, half):
    out = a
    for d in range(-half, half + 1):
        if d:
            out = np.minimum(out, np.roll(a, d, axis=1))
    return out


def _close_seam_x(rgb, half):
    """Morphological closing along x: fills the narrow vertical hair parting.

    The parting is a concave groove in the mesh, so it renders as one dark
    vertical line.  Closing horizontally (grow, then shrink, along x only)
    fills a gap of that width while leaving horizontal detail alone.
    """
    return _window_min_x(_window_max_x(rgb, half), half)


def _erode(mask, k):
    out = mask.copy()
    for _ in range(k):
        m = out
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        out = e
    return out


# --------------------------------------------------------------------------- #
#  electrodes on the scalp (ray cast against the full-resolution mesh)
# --------------------------------------------------------------------------- #
def _electrode_points(verts, faces, origin, theta_max_deg):
    names = list(ELECTRODES)
    dirs = np.zeros((len(names), 3))
    for k, nm in enumerate(names):
        u, v = ELECTRODES[nm]
        r = min(np.hypot(u, v), 1.0)
        phi = np.arctan2(u, v)
        th = math.radians(theta_max_deg) * r
        dirs[k] = (np.sin(th) * np.sin(phi), np.cos(th), np.sin(th) * np.cos(phi))

    v0 = verts[faces[:, 0]]
    e1 = verts[faces[:, 1]] - v0
    e2 = verts[faces[:, 2]] - v0
    d = dirs[:, None, :]
    h = np.cross(d, e2[None, :, :])
    a = np.einsum("kfi,fi->kf", h, e1)
    ok = np.abs(a) > 1e-12
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = origin[None, None, :] - v0[None, :, :]
    uu = f * np.einsum("kfi,kfi->kf", s, h)
    q = np.cross(s, e1[None, :, :])
    vv = f * np.einsum("kfi,kfi->kf", d, q)
    tt = f * np.einsum("kfi,fi->kf", q, e2)
    good = (ok & (uu >= -1e-9) & (vv >= -1e-9) & (uu + vv <= 1.0 + 1e-9)
            & (tt > 1e-6))
    tmax = np.where(good, tt, -np.inf).max(axis=1)
    tmax[~np.isfinite(tmax)] = 0.0
    pts = origin[None, :] + dirs * tmax[:, None]
    return {nm: pts[k] for k, nm in enumerate(names)}


# --------------------------------------------------------------------------- #
#  entry point
# --------------------------------------------------------------------------- #
def draw(ax, x, y, w, h, **kwargs):
    tone = kwargs.get("tone", "target")
    show_eeg = bool(kwargs.get("show_eeg", True))
    seed = int(kwargs.get("seed", SEED))
    _ = np.random.default_rng(seed)          # kept seeded / reproducible
    if tone not in _RAMP:
        tone = "target"

    img = _render(tone)
    origin, basis, lo, span = _RENDER_CACHE[tone + "_map"]

    cx, cy = x + w / 2.0, y + h / 2.0
    rad = 0.5 * min(w, h) * 0.96
    # square image -> square on screen: no head is ever stretched
    half = 0.5 * span * rad / (0.5 * span)
    ax.imshow(img, extent=(cx - rad, cx + rad, cy - rad, cy + rad),
              origin="upper", interpolation="bilinear", zorder=2,
              aspect="auto")

    def proj(p3):
        c = (np.asarray(p3, dtype=float) - origin) @ basis.T
        px = (c[0] - lo[0]) / span
        py = (c[1] - lo[1]) / span
        return np.array([cx + (px - 0.5) * 2 * rad, cy + (py - 0.5) * 2 * rad]), c[2]

    n_dot = 0
    if show_eeg:
        verts, faces = _head_mesh()
        pts3 = _electrode_points(verts, faces, origin, THETA_MAX_DEG)
        proj_e = {nm: proj(p) for nm, p in pts3.items()}

        for nm, (xy, z) in proj_e.items():
            rr = 0.040 * rad
            ax.add_patch(Circle(xy, rr, facecolor=_DOT[tone], edgecolor="white",
                                linewidth=0.3, zorder=6))
            n_dot += 1

    return {"cx": cx, "cy": cy, "r": rad, "n_electrodes": n_dot}


# --------------------------------------------------------------------------- #
#  self-check preview
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, os.pardir, "assets"))

    t0 = time.time()
    _render("target")
    _render("source")
    print("render time: %.1fs" % (time.time() - t0))

    fig = plt.figure(figsize=(3.2, 1.9), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 1.9)
    ax.axis("off")
    info = draw(ax, 0.16, 0.10, 1.70, 1.70, tone="target")
    draw(ax, 1.95, 0.30, 1.10, 1.10, tone="source")
    fig.savefig(os.path.join(out, "head3d_preview.png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(0.80, 0.80), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 0.80)
    ax.set_ylim(0, 0.80)
    ax.axis("off")
    draw(ax, 0.10, 0.10, 0.60, 0.60, tone="target")
    fig.savefig(os.path.join(out, "head3d_preview_small.png"), dpi=300)
    plt.close(fig)

    print("head3d ok:", info)
    np.save(os.path.join(out, "head3d_target.npy"), _render("target"))
    print("image px:", _render("target").shape)
