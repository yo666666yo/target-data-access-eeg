"""试次栅格 (categorical raster) —— stripe 的学术化画法。

不画"胶囊条"，而是把它画成论文里标准的 categorical raster /
state-sequence 图：

* **一列 = 一个试次**，列宽等分，方块是矩形（无圆角、无内嵌小方块）；
* **上排方块的颜色 = 该试次的刺激类别**（4 类，``palette.CLASS_COLORS``）；
* **下排方块的颜色 = 该试次所属的数据划分**
  （available ``C_s`` / reserved ``T_s``，或 training 80 % / selection 20 %）；
* 两排共享同一组列，所以"只切一次"这件事只需要一条竖线就能同时切断两个编码。

mark 的**大小不编码任何量**：旧版从 ``trial_strip_available_reserved.svg``
继承的柱高来自 ``rng.normal``，是随机数，读图的人会看到一个并不存在的量，
这是它显得"儿戏"的根源。这里所有方块尺寸完全一致。

默认的类别序列是**分层**的：切分点两侧的类别计数相同（``split`` 处
class-stratified），所以"available 和 reserved 的类别组成一致"这件事在图上
直接看得见，而不是只写在正文里。

    draw(ax, x, y, w, h, n=10, split=0.5, ...) -> {"cut_x", "x0", "x1", ...}
"""

import numpy as np
from matplotlib.patches import Rectangle

try:                                    # 作为包导入
    from .palette import *              # noqa: F401,F403
    from .palette import SEED, BLUE, ORANGE, DARK, LIGHT, CLASS_COLORS
except ImportError:                     # 作为脚本导入
    from palette import *               # noqa: F401,F403
    from palette import SEED, BLUE, ORANGE, DARK, LIGHT, CLASS_COLORS

BAR_FRAC = 0.84         # 方块宽度 / 列宽
GAP_FRAC = 0.16         # 两排之间的空隙 / 总高
CLASS_SHARE = 0.40      # 上排（类别）在可用高度中的占比


def stratified_classes(n, split, n_cls=4, seed=SEED):
    """类别序列，切分点两侧的类别计数完全一致。"""
    rng = np.random.default_rng(seed)

    def counts(m):
        c = np.full(n_cls, m // n_cls, dtype=int)
        c[: m - c.sum()] += 1
        return c

    k = int(round(n * split))
    parts = []
    for m in (k, n - k):
        seq = np.repeat(np.arange(n_cls), counts(m))
        rng.shuffle(seq)
        parts.append(seq)
    return np.concatenate(parts).astype(int)


def draw(ax, x, y, w, h, **kwargs):
    """把 n 个试次画进矩形 ``(x, y, w, h)``（调用方的数据坐标）。"""
    n = int(kwargs.get("n", 10))
    split = float(kwargs.get("split", 0.5))
    left_color = kwargs.get("left_color", BLUE)
    right_color = kwargs.get("right_color", ORANGE)
    seed = int(kwargs.get("seed", SEED))
    z = float(kwargs.get("zorder", 3.0))
    classes = kwargs.get("classes")
    if classes is None:
        classes = stratified_classes(n, split, seed=seed)
    classes = np.asarray(classes, dtype=int)

    cw = w / n
    bar_w = cw * BAR_FRAC
    usable = h * (1.0 - GAP_FRAC)
    h_cls = usable * CLASS_SHARE
    h_role = usable * (1.0 - CLASS_SHARE)
    y_role = y
    y_cls = y + h_role + h * GAP_FRAC
    kcut = int(round(n * split))

    for i in range(n):
        cx = x + (i + 0.5) * cw
        ax.add_patch(Rectangle(
            (cx - bar_w / 2.0, y_cls), bar_w, h_cls,
            facecolor=CLASS_COLORS[classes[i] % len(CLASS_COLORS)],
            edgecolor="none", zorder=z + 1))
        ax.add_patch(Rectangle(
            (cx - bar_w / 2.0, y_role), bar_w, h_role,
            facecolor=(left_color if i < kcut else right_color),
            edgecolor="none", zorder=z))

    return {"cut_x": x + split * w, "x0": x, "x1": x + w,
            "cx0": x, "cx1": x + w, "n": n, "classes": classes.tolist()}


# ------------------------------------------------------------------ self-check
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3.4, 1.10), dpi=300)
    bw = fig.add_axes([0, 0, 1, 1])
    bw.set_xlim(0, 3.4)
    bw.set_ylim(0, 1.10)
    bw.axis("off")
    bw.plot([1.05, 1.05], [0.78, 1.02], color="#9aa8b1", lw=0.9,
            ls=(0, (2.2, 1.6)))
    draw(bw, 0.30, 0.74, 2.60, 0.30, split=0.5)
    bw.plot([2.35, 2.35], [0.14, 0.38], color="#9aa8b1", lw=0.9,
            ls=(0, (2.2, 1.6)))
    draw(bw, 0.30, 0.12, 2.60, 0.26, split=0.8,
         left_color=DARK, right_color=LIGHT)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "assets", "trial_raster_preview.png")
    fig.savefig(out, dpi=300)
    print("wrote", out)
