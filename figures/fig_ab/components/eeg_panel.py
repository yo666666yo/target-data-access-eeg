"""Simulated multi-channel EEG traces (Fig. 1 inset).

No background, no frame, no axes: the panel is *only* the traces, so it can sit
directly next to the subject icon without drawing a box around the signals.

The signal is synthetic but built the way real EEG is built, per channel:

* a 1/f background made by shaping white noise in the frequency domain
  (amplitude ~ f^-beta/2, band-limited to 0.6-42 Hz) and transforming back.
  This is the part a sum-of-sinusoids hack gets wrong: it gives the smooth,
  irregular, aperiodic look of a real trace instead of either fuzzy hair
  (white noise) or a visibly periodic wiggle (a handful of tones);
* an occipital-style alpha rhythm (9-12 Hz) under a slow amplitude envelope,
  so it comes and goes in bursts the way alpha actually does;
* a slow drift;
* a biphasic ERP - an early negativity and a later, broader positivity - on a
  contiguous block of channels, as a scalp region would show.

Entry point::

    draw(ax, x, y, w, h, kind="target", n_ch=None, seed=palette.SEED) -> dict
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

try:                                    # package import
    from .palette import *              # noqa: F401,F403
except ImportError:                     # run as a plain script
    from palette import *               # noqa: F401,F403

_KIND = {
    "target": dict(n_ch=10, color=INK, seed_off=0, erp=1.00, lw=0.42),
    "source": dict(n_ch=7, color=DARK, seed_off=911, erp=0.72, lw=0.42),
}

_T_EPOCH = 0.80         # epoch length in seconds
_N_SAMP = 1100          # samples per epoch
_FS = _N_SAMP / _T_EPOCH
_AMP_FRAC = 0.46        # trace amplitude as a fraction of channel pitch


def _one_over_f(rng, n, beta, f_lo=0.6, f_hi=32.0):
    """Band-limited 1/f^beta noise, unit standard deviation."""
    f = np.fft.rfftfreq(n, d=1.0 / _FS)
    amp = np.zeros_like(f)
    band = (f >= f_lo) & (f <= f_hi)
    amp[band] = (f[band] / f_lo) ** (-beta / 2.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, f.size)
    spec = amp * np.exp(1j * phase)
    spec[0] = 0.0
    sig = np.fft.irfft(spec, n)
    return sig / (sig.std() + 1e-12)


def _epoch(rng, n_ch, erp_scale):
    """(n_ch, n_samp) synthetic EEG, globally normalised to unit peak."""
    t = np.linspace(0.0, _T_EPOCH, _N_SAMP)
    tn = t / _T_EPOCH
    out = np.zeros((n_ch, _N_SAMP))

    # the ERP sits over one scalp region: a contiguous block of channels
    n_erp = max(2, int(round(0.6 * n_ch)))
    start = int(rng.randint(0, n_ch - n_erp + 1))
    erp_ch = set(range(start, start + n_erp))

    early = np.exp(-((tn - 0.26) / 0.045) ** 2)      # sharp early negativity
    late = np.exp(-((tn - 0.54) / 0.105) ** 2)       # broader late positivity

    for c in range(n_ch):
        # aperiodic background, kept low: it is the floor alpha rides on,
        # not the thing that should dominate the trace
        beta = 1.55 + 0.60 * rng.rand()
        sig = 0.62 * _one_over_f(rng, _N_SAMP, beta)

        # alpha: the visible rhythm, gated by a slow envelope so it comes and
        # goes in bursts the way occipital alpha actually does
        f_a = 8.8 + 3.2 * rng.rand()
        env = np.clip(np.sin(2.0 * np.pi * (0.6 + 0.7 * rng.rand()) * t
                             + rng.rand() * 2.0 * np.pi), 0.0, None)
        a_amp = (1.45 + 1.30 * rng.rand()) * (0.30 + 0.70 * env)
        ph = rng.rand() * 2.0 * np.pi
        sig += a_amp * np.sin(2.0 * np.pi * f_a * t + ph)
        sig += 0.16 * a_amp * np.sin(2.0 * np.pi * 2.0 * f_a * t + 1.7 * ph)

        # a light band of fast activity: the fine texture every real trace has,
        # kept low enough that it never smears into grey at final size
        sig += 0.16 * _one_over_f(rng, _N_SAMP, 1.05, f_lo=13.0, f_hi=46.0)

        # slow drift
        sig += 0.32 * np.sin(2.0 * np.pi * (0.35 + 0.5 * rng.rand()) * t
                             + rng.rand() * 2.0 * np.pi)

        # sharp transient events: a real trace is not a smooth undulation, it
        # carries narrow high-amplitude spikes.  x*exp(-x^2) is biphasic and
        # dies away in a few samples, exactly the shape a spike has.
        for _ in range(int(rng.randint(1, 4))):
            t0 = rng.uniform(0.06, 0.94) * _T_EPOCH
            w = 0.0038 + 0.0028 * rng.rand()
            z = (t - t0) / w
            sig += (3.4 + 2.8 * rng.rand()) * z * np.exp(-(z ** 2))

        if c in erp_ch:
            g = erp_scale * (0.60 + 1.00 * rng.rand())
            sig += g * (-1.30 * early + 1.05 * late)

        sig *= 0.80 + 0.40 * rng.rand()             # channels differ in gain
        out[c] = sig

    peak = np.abs(out).max()
    if peak > 0:
        out /= peak
    return out, tn


def draw(ax, x, y, w, h, **kwargs):
    """Draw the EEG traces into the rect ``(x, y, w, h)`` in data coordinates.

    Nothing else is drawn - no fill and no border - so the caller controls the
    surrounding whitespace.
    """
    kind = kwargs.get("kind", "target")
    cfg = _KIND.get(kind, _KIND["target"])
    seed = int(kwargs.get("seed", SEED)) + cfg["seed_off"]
    n_ch = int(kwargs.get("n_ch", cfg["n_ch"]))
    color = kwargs.get("color", cfg["color"])
    lw = float(kwargs.get("lw", cfg["lw"]))
    amp_frac = float(kwargs.get("amp", _AMP_FRAC))
    rng = np.random.RandomState(seed)

    clip_rect = Rectangle((x, y), w, h, transform=ax.transData)

    def emit(line):
        # add first, then clip: add_line resets the clip to the axes patch
        ax.add_line(line)
        line.set_clip_path(clip_rect)
        return line

    data, t = _epoch(rng, n_ch, cfg["erp"])
    marg = 0.06
    pitch = (1.0 - 2.0 * marg) * h / max(n_ch - 1, 1)
    amp = amp_frac * pitch

    xs = x + t * w
    for c in range(n_ch):
        base = y + h * (1.0 - marg) - c * pitch     # channel 0 on top
        emit(Line2D(xs, base + amp * data[c], color=color, linewidth=lw,
                    alpha=1.0, solid_capstyle="round",
                    solid_joinstyle="round", zorder=3))

    return {"kind": kind, "n_ch": n_ch}


# ------------------------------------------------------------------ self-check
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PW, PH = 0.92, 0.56          # final on-figure size of one panel (inches)
    fig = plt.figure(figsize=(2.5, 0.78), dpi=300)
    fig.patch.set_facecolor(PAPER)

    for i, kind in enumerate(("target", "source")):
        left = 0.12 + i * 1.15
        ax = fig.add_axes([left / 2.5, 0.14 / 0.78, PW / 2.5, PH / 0.78])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        draw(ax, 0.0, 0.0, 1.0, 1.0, kind=kind)

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "assets", "eeg_panel_preview.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, facecolor=PAPER)
    print("wrote", out)
