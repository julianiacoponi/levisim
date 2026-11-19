"""
Sphinx extension: interactive (Plotly) versions of the README showcase figures

At the start of an HTML build, every script in docs/guide/scripts/showcase/ listed in SHOWCASE is
imported and its ``compute()`` and ``interactive_figure(data)`` are run; the figure is written as an
HTML fragment to docs/_generated/showcase/<name>.html (git-ignored), which the pages under
docs/showcase/ embed with ``{raw} html`` and ``:file:``. Nothing generated is committed, so the
repository holds only the scripts and the static PNGs that the README shows.

plotly.js is loaded from its CDN (cdn.plot.ly), pinned to the version that the installed plotly
package was built against. Set LEVISIM_DOCS_SKIP_INTERACTIVE=1 to skip the computation (about half
a minute, most of it the stochastic simulations of the PRR 6 example) for a quick local build; the
pages then show a note and the static figure instead. If plotly is not installed the same happens,
with an INFO message rather than a warning, so that a -W build without plotly still passes.
"""
import importlib.util
import os
import time
from pathlib import Path

from sphinx.util import logging

logger = logging.getLogger(__name__)

DOCS = Path(__file__).resolve().parents[1]
SCRIPTS = DOCS / 'guide' / 'scripts' / 'showcase'
OUTPUT = DOCS / '_generated' / 'showcase'
SHOWCASE = ('prr5_mode_orientations', 'prr6_directional_force', 'prr8_mode_comb', 'stability_tweezer_phase')

PLACEHOLDER = (
    '<div class="admonition note"><p class="admonition-title">Interactive figure not built</p>'
    '<p>{reason} The static version is below; run <code>python docs/guide/scripts/showcase/{name}.py</code> '
    'to regenerate it.</p></div>\n'
    '<img src="../_images/{name}.png" alt="{name}" style="width:100%">\n'
)


def load_script(name):
    spec = importlib.util.spec_from_file_location(f'levisim_showcase_{name}', SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_placeholders(reason):
    for name in SHOWCASE:
        (OUTPUT / f'{name}.html').write_text(PLACEHOLDER.format(reason=reason, name=name), encoding='utf-8')


def build_interactive_figures(app):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if app.builder.format != 'html':
        write_placeholders('Interactive figures are built for the HTML pages only.')
        return
    if os.environ.get('LEVISIM_DOCS_SKIP_INTERACTIVE'):
        logger.info('showcase_plotly: LEVISIM_DOCS_SKIP_INTERACTIVE is set, skipping the interactive figures')
        write_placeholders('This build skipped them (LEVISIM_DOCS_SKIP_INTERACTIVE).')
        return
    try:
        import plotly  # noqa: F401
    except ImportError:
        logger.info('showcase_plotly: plotly is not installed, skipping the interactive figures')
        write_placeholders('plotly was not installed when these pages were built.')
        return
    import matplotlib
    matplotlib.use('Agg')
    for name in SHOWCASE:
        start = time.time()
        module = load_script(name)
        figure = module.interactive_figure(module.compute())
        html = figure.to_html(
            full_html=False, include_plotlyjs='cdn', default_width='100%',
            config=dict(responsive=True, displaylogo=False, toImageButtonOptions=dict(filename=name)),
        )
        (OUTPUT / f'{name}.html').write_text(html, encoding='utf-8')
        logger.info(f'showcase_plotly: {name} in {time.time() - start:.1f} s')


def setup(app):
    app.connect('builder-inited', build_interactive_figures)
    return dict(version='1.0', parallel_read_safe=True, parallel_write_safe=True)
