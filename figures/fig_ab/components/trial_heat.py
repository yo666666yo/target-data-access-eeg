"""试次色块矩阵：``rows x n`` 个方格，**一格 = 一个样本，一色 = 一个类别**。

* 矩阵是 ``3 x 10``（三行、十个样本 / 列），格子尺寸完全一致 ——
  大小不编码任何量；
* 方格的**颜色 = 该样本的类别**，取自一条离散色带；
* 整条 stripe 只用一条色带，由 ``tone`` 选定：
  **上侧 stripe（held-out subject）用「蓝 -> 绿」**，
  **下侧 stripe（N-1 source subjects）用「红 -> 橙」**，两条一眼可分；
* **左侧样本整体淡化 70 %、右侧淡化 30 %**（即左端 alpha 0.30、右端 0.70），
  用透明度而不是再加一个颜色通道来区分数据划分
  （available ``C_s`` / reserved ``T_s``，或 training 80 % / selection 20 %）。

每条的类别序列都是**分层**的：切分点两侧的类别计数一致，所以
class-stratified 这件事在图上直接看得见。

    draw(ax, x, y, w, h, n=10, split=0.5, tone="cool") -> {"cut_x", ...}
"""

import numpy as np
from matplotlib.patches import Rectangle

try:                                    # 作为包导入
    from .palette import *              # noqa: F401,F403
    from .palette import SEED
    from .trial_raster import stratified_classes
except ImportError:                     # 作为脚本导入
    from palette import *               # noqa: F401,F403
    from palette import SEED
    from trial_raster import stratified_classes

RAMP = {"cool": ("#2c7bb6", "#5aae61"),     # 上侧条带：只有 蓝 -> 绿
        "warm": ("#b2182b", "#e6550d"),     # 下侧条带：只有 红 -> 橙
        "amber": ("#f4c95d", "#e07b39")}    # 黄 -> 橙
N_CLASSES = 3                               # 每条色带上取几个离散类别色
CELL_W = 1.00                               # 方格宽 / 列宽 —— 热力图不设缝隙
CELL_H = 1.00                               # 方格高 / 行高
FADE_LEFT, FADE_RIGHT = 0.60, 0.20          # 左侧 / 右侧的淡化量
ROWS = 5                                    # 一个 stripe = ROWS x n 个方格


def _mix_white(c, f):
    """Fade a colour towards white by fraction f (no alpha, so overlays stack)."""
    if isinstance(c, (tuple, list)):
        v = np.asarray(c[:3], dtype=float)
        v = v * (1.0 - f) + f
        return tuple(v)
    c = c.lstrip("#")
    v = np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    v = v * (1.0 - f) + f
    return tuple(v)


def _hex(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _ramp(c0, c1, k):
    """在 c0..c1 之间取 k 个离散类别色。"""
    a, b = _hex(c0), _hex(c1)
    if k < 2:
        return [tuple(a)]
    return [tuple(a + (b - a) * (i / (k - 1.0))) for i in range(k)]


def draw(ax, x, y, w, h, **kwargs):
    """``overlays`` marks regions of the matrix as touched by a protocol.

    Each entry is (kind, a0, a1, colour) where kind is "rows" or "cols".
    Every cell inside an overlay first fades to 20 % of its own colour, then
    the overlay colour is laid over it at 40 % -- so the touched cells read as
    washed-out and tinted, visibly different from the untouched ones.
    """
    overlays = kwargs.get("overlays")

    def _hit(r, i):
        for kind, a0, a1, _c in (overlays or []):
            if kind == "rows" and a0 <= r < a1:
                return True
            if kind == "cols" and a0 <= i < a1:
                return True
        return False

    n = int(kwargs.get("n", 10))
    split = float(kwargs.get("split", 0.5))
    tone = kwargs.get("tone", "cool")
    fade_left = float(kwargs.get("fade_left", FADE_LEFT))
    fade_right = float(kwargs.get("fade_right", FADE_RIGHT))
    seed = int(kwargs.get("seed", SEED))
    z = float(kwargs.get("zorder", 3.0))

    rows = int(kwargs.get("rows", ROWS))
    pal = _ramp(*RAMP.get(tone, RAMP["cool"]), N_CLASSES)
    seqs = [stratified_classes(n, split, N_CLASSES, seed=seed + 17 * r)
            for r in range(rows)]

    cw, rh = w / n, h / rows
    bw, bh = cw * CELL_W, rh * CELL_H
    kcut = int(round(n * split))

    for r in range(rows):
        cy = y + h - (r + 0.5) * rh             # 第 0 行在最上面
        for i in range(n):
            cx = x + (i + 0.5) * cw
            base = pal[seqs[r][i] % N_CLASSES]
            fc = _mix_white(base, 0.80) if _hit(r, i) else base
            ax.add_patch(Rectangle(
                (cx - bw / 2.0, cy - bh / 2.0), bw, bh,
                facecolor=fc, edgecolor="none",
                alpha=1.0 - (fade_left if i < kcut else fade_right), zorder=z))

    for kind, a0, a1, colour in (overlays or []):
        if kind == "rows":
            ry = y + h - a1 * (h / rows)
            ax.add_patch(Rectangle((x, ry), w, (a1 - a0) * (h / rows),
                                   fc=colour, ec="none", alpha=0.40,
                                   zorder=z + 2))
        else:
            ax.add_patch(Rectangle((x + a0 * (w / n), y),
                                   (a1 - a0) * (w / n), h,
                                   fc=colour, ec="none", alpha=0.40,
                                   zorder=z + 2))

    return {"cut_x": x + split * w, "x0": x, "x1": x + w,
            "cx0": x, "cx1": x + w, "n": n, "rows": rows,
            "classes": [s_.tolist() for s_ in seqs]}


# ------------------------------------------------------------------ self-check
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3.4, 1.20), dpi=340)
    a = fig.add_axes([0, 0, 1, 1])
    a.set_xlim(0, 3.4); a.set_ylim(0, 1.20); a.axis("off")
    for yy, sp, tn in ((0.62, 0.5, "cool"), (0.10, 0.8, "warm")):
        info = draw(a, 0.30, yy, 2.60, 0.44, split=sp, tone=tn)
        a.plot([info["cut_x"]] * 2, [yy - 0.04, yy + 0.48], color="#9aa8b1",
               lw=0.9, ls=(0, (2.2, 1.6)))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "assets", "trial_heat_preview.png")
    fig.savefig(out, dpi=340)
    print("wrote", out)
