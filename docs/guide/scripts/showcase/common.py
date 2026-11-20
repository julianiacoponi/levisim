"""
Shared pieces of the README showcase scripts in this folder

Each showcase script defines ``compute(quick=False)``, which returns a dict of arrays,
``static_figure(data)``, which draws the README figure with matplotlib in levisim's xmgrace-based
style, and ``interactive_figure(data)``, which draws the same data with Plotly for the documentation
site (docs/_ext/showcase_plotly.py runs it at build time). Run a script directly to rewrite its PNG:

    python docs/guide/scripts/showcase/prr5_mode_orientations.py
"""
from pathlib import Path

import numpy as np

from levisim.core.logger import set_console_level
from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style

FIGURES = Path(__file__).resolve().parents[2] / 'figures' / 'showcase'
DOCS_URL = 'https://julianiacoponi.github.io/levisim/showcase/'

# a qualitative palette with enough saturation to read at README width, used for named curves
COLOURS = {
    'blue': '#1f4fb4',
    'purple': '#7b2d8e',
    'red': '#c0392b',
    'orange': '#e67e22',
    'teal': '#138d75',
    'grey': '#606060',
}


def quiet_console():
    """Console logging at ERROR: the sweeps build many systems on purpose, some unstable, and their WARNINGs are expected"""
    set_console_level('ERROR')


def setup_matplotlib():
    """levisim's xmgrace-based rcParams and a non-interactive backend"""
    import matplotlib
    matplotlib.use('Agg')
    set_plt_rcParams_to_xmgrace_style()


def save(fig, name, dpi=180):
    """
    Saves a README figure into docs/guide/figures/showcase/<name>.png and returns the path

    The PNG is reduced to a 256-colour palette (with dithering), which keeps the colour maps smooth
    to the eye at a third of the size of the full-colour file.
    """
    import io
    from PIL import Image
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f'{name}.png'
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=dpi, format='png')
    buffer.seek(0)
    image = Image.open(buffer).convert('RGB')
    image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(path, optimize=True)
    return path


def rgb_hex(rgba):
    """'#rrggbb' of a matplotlib RGBA tuple (for passing colormap colours to Plotly)"""
    return '#' + ''.join(f'{int(round(255 * c)):02x}' for c in rgba[:3])


def colormap_to_plotly(name, n=11):
    """A Plotly colorscale sampled from a matplotlib colormap, so both figures use the same colours"""
    import matplotlib
    cmap = matplotlib.colormaps[name]
    return [[float(x), rgb_hex(cmap(x))] for x in np.linspace(0, 1, n)]


def plotly_axes_style():
    """Plotly axis settings that echo the xmgrace style: a closed black frame, ticks inside"""
    return dict(
        showline=True, linecolor='black', linewidth=1.5, mirror='ticks',
        ticks='inside', ticklen=7, tickwidth=1.2, tickcolor='black',
        showgrid=False, zeroline=False,
    )


def plotly_layout(fig, height=480):
    """Common Plotly layout: white background, Arial, a compact legend"""
    fig.update_layout(
        template='simple_white', height=height, font=dict(family='Arial', size=14, color='black'),
        margin=dict(l=70, r=20, t=40, b=60), hovermode='closest',
        legend=dict(bgcolor='rgba(255,255,255,0.8)', borderwidth=0),
    )
    fig.update_xaxes(**plotly_axes_style())
    fig.update_yaxes(**plotly_axes_style())
    return fig
