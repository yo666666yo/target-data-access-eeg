"""Shared matplotlib style for all paper figures.

Monochrome + colorblind-safe (Wong 2011 / IBM-like palette) for print-friendliness.
"""
import matplotlib
import matplotlib.pyplot as plt

FONT_SIZE = 10
DPI = 300
FORMAT = 'pdf'
FIG_DIR = 'figures'

matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,   # editable text in PDFs
    'ps.fonttype': 42,
})

# Wong 2011 colorblind-safe palette (Nature Methods)
COLORS = {
    'black':  '#000000',
    'orange': '#E69F00',
    'skyblue':'#56B4E9',
    'green':  '#009E73',
    'yellow': '#F0E442',
    'blue':   '#0072B2',
    'vermillion':'#D55E00',
    'purple': '#CC79A7',
}
CB_CYCLE = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple'],
            COLORS['vermillion'], COLORS['skyblue'], COLORS['yellow'], COLORS['black']]


def save_fig(fig, name, fmt=FORMAT, output_dir=FIG_DIR):
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(path)
    print(f'[saved] {path}')
    return path
