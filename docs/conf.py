"""
Sphinx configuration for the levisim documentation

Build from the repo root with:
    pip install -r docs/requirements.txt
    sphinx-build -b html docs docs/_build/html
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

# make the in-tree package importable for autodoc, whether or not it is pip-installed
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ----------------------------------------------------------------------------------------------
# project
# ----------------------------------------------------------------------------------------------
project = 'levisim'
author = 'Julian Iacoponi'
copyright = '2025-2026, Julian Iacoponi'

# ----------------------------------------------------------------------------------------------
# extensions
# ----------------------------------------------------------------------------------------------
extensions = [
    # MyST-NB parses both the MyST Markdown pages and the gallery notebooks; it includes and
    # configures myst_parser itself, so myst_parser must not also be listed here
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
]

# myst_nb parses .md and .ipynb ('myst-nb'). '.rst' must come first: autosummary writes its stub
# pages with the first suffix it takes to be reStructuredText, and it cannot tell that the
# 'myst-nb' parser is not (with '.md' first the API stubs came out as unparseable .md files)
source_suffix = {'.rst': 'restructuredtext', '.md': 'myst-nb', '.ipynb': 'myst-nb'}
root_doc = 'index'
templates_path = ['_templates']
html_static_path = ['_static']
exclude_patterns = [
    '_build',
    'requirements.txt',
    'check_sphinx_warnings.py',
    'Thumbs.db',
    '.DS_Store',
]

# MyST: $...$ / $$...$$ maths, \begin{align} environments, ::: fences, definition lists
myst_enable_extensions = ['dollarmath', 'amsmath', 'colon_fence', 'deflist']
# GitHub-style #heading-slug anchors, so the in-page tables of contents written for GitHub work
myst_heading_anchors = 4
# ```mermaid fences (which GitHub renders) become sphinxcontrib-mermaid directives, so the README's
# flowchart of the calculation methods renders on the site too
myst_fence_as_directive = ['mermaid']

# autodoc / autosummary: one page per public module, generated at build time into _autosummary/
autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
# optional dependencies used by a few helpers only; mocked so autodoc can import every module
autodoc_mock_imports = ['cv2', 'elasticsearch', 'plyer']
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Cross-references to targets Sphinx cannot resolve are NOT errors. Most of the API's docstrings
# are free-form physics notes rather than numpydoc, with no type annotations to resolve.
nitpicky = False

# ----------------------------------------------------------------------------------------------
# notebook gallery (MyST-NB; notebooks are rendered as committed and never executed)
# ----------------------------------------------------------------------------------------------
nb_execution_mode = 'off'
# progress bars and warnings that a notebook wrote to stderr are not part of the gallery
nb_output_stderr = 'remove'
nb_merge_streams = True

# The demo, showcase and paper-reproduction notebooks live in notebooks/, outside this source
# directory, so every build copies them into docs/_notebooks/ (git-ignored) and writes one index
# page per folder there; notebooks.md links those pages. A folder with no notebooks yet gets a
# page saying so, so the build stays warning-free while the gallery fills up.
GALLERY_FOLDERS = {
    'demos': ('Demos', 'Short, task-focused walk-throughs of the levisim API.'),
    'showcase': ('Showcase', 'A tour of what levisim can do, in one notebook.'),
    'papers': ('Paper reproductions', 'Figures from published papers, recomputed with levisim.'),
}
GALLERY_DIR = Path(__file__).resolve().parent / '_notebooks'


def build_notebook_gallery(gallery_dir=GALLERY_DIR, notebooks_dir=REPO_ROOT / 'notebooks'):
    """Copies notebooks/{demos,showcase,papers}/**/*.ipynb into `gallery_dir`, with an index page per folder"""
    shutil.rmtree(gallery_dir, ignore_errors=True)
    gallery_dir.mkdir(parents=True)
    for folder, (title, purpose) in GALLERY_FOLDERS.items():
        source = notebooks_dir / folder
        notebooks = sorted(
            path for path in source.rglob('*.ipynb') if '.ipynb_checkpoints' not in path.parts
        ) if source.is_dir() else []
        entries = []
        for notebook in notebooks:
            relative = notebook.relative_to(source)
            target = gallery_dir / folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            notebook_json = close_notebook_heading_gaps(notebook.read_text(encoding='utf-8'))
            target.write_text(rewrite_notebook_docs_links(notebook_json, notebook, target), encoding='utf-8')
            entries.append(f'{folder}/{relative.with_suffix("").as_posix()}')
        if entries:
            body = '```{toctree}\n:maxdepth: 1\n\n' + '\n'.join(entries) + '\n```\n'
        else:
            body = f'No notebooks in `notebooks/{folder}/` yet.\n'
        (gallery_dir / f'{folder}.md').write_text(f'# {title}\n\n{purpose}\n\n{body}', encoding='utf-8')


def close_notebook_heading_gaps(notebook_json):
    """
    Applies `close_heading_level_gaps` across all the markdown cells of a notebook, in order

    Jupyter renders a skipped heading level (a title cell '# ...' followed by '### ...') happily;
    MyST warns about it, which would fail the -W build. The notebook in notebooks/ is untouched.
    """
    nb = json.loads(notebook_json)
    cells = [cell for cell in nb.get('cells', []) if cell.get('cell_type') == 'markdown']
    if not cells:
        return notebook_json
    separator = '\x00levisim-cell-break\x00'
    sources = [''.join(cell['source']) if isinstance(cell['source'], list) else cell['source'] for cell in cells]
    fixed = close_heading_level_gaps(('\n' + separator + '\n').join(sources)).split('\n' + separator + '\n')
    for cell, source in zip(cells, fixed):
        cell['source'] = source
    return json.dumps(nb, ensure_ascii=False, indent=1)


# a Markdown link or image target: ](target) or ](target#anchor), relative paths only
_RELATIVE_LINK = re.compile(r'\]\((?![a-zA-Z][a-zA-Z0-9+.-]*:|#|/)([^)\s#]+)(#[^)\s]*)?\)')


def rewrite_notebook_docs_links(notebook_json, notebook, target, docs_dir=REPO_ROOT / 'docs'):
    """
    Re-points relative links from a notebook's markdown cells into docs/ so they work in its gallery copy

    A notebook in notebooks/demos/ links a documentation page as ../../docs/guide/page.md, which is
    right on GitHub; its copy in docs/_notebooks/demos/ needs ../../guide/page.md, or MyST cannot
    resolve the cross-reference (a warning, which fails the -W build). Links to anything outside
    docs/ (e.g. to a neighbouring gallery notebook) are left alone. The notebook in notebooks/ is
    untouched.
    """
    def rewrite(match):
        path, anchor = match.group(1), match.group(2) or ''
        resolved = (notebook.parent / path).resolve()
        if docs_dir not in resolved.parents:
            return match.group(0)
        return '](' + Path(os.path.relpath(resolved, target.parent)).as_posix() + anchor + ')'

    nb = json.loads(notebook_json)
    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'markdown':
            continue
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        new_source = _RELATIVE_LINK.sub(rewrite, source)
        if new_source != source:
            cell['source'] = new_source
            changed = True
    return json.dumps(nb, ensure_ascii=False, indent=1) if changed else notebook_json


# set LEVISIM_DOCS_OFFLINE=1 to build without fetching the intersphinx inventories
intersphinx_mapping = {} if os.environ.get('LEVISIM_DOCS_OFFLINE') else {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# ----------------------------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------------------------
html_theme = 'furo'
html_title = 'levisim'
html_theme_options = {
    'source_repository': 'https://github.com/julianiacoponi/levisim/',
    'source_branch': 'main',
    'source_directory': 'docs/',
}
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True

# ----------------------------------------------------------------------------------------------
# GitHub-flavoured maths -> MyST maths
# ----------------------------------------------------------------------------------------------
# The README and the two-tone study pages are written for GitHub, whose maths syntax is
# $`x`$ inline and ```math fenced blocks. MyST's dollarmath wants $x$ and $$...$$. Rather than
# keep two copies, the source text is converted in memory as Sphinx reads it (the files on disk
# are untouched, so they still render on GitHub).
_GITHUB_INLINE_MATH = re.compile(r'\$`(.+?)`\$')
# a ```math fence, possibly indented (e.g. inside a list item); the closing fence has the same indent
_GITHUB_BLOCK_MATH = re.compile(r'^([ \t]*)```math[ \t]*\n(.*?)\n\1```[ \t]*$', re.S | re.M)
_HEADING = re.compile(r'^(#{1,6})([ \t].*)$')

# pages written for GitHub, rendered as they are on disk apart from these in-memory conversions
GITHUB_PAGES = ('two_tone_entanglement/',)


def github_math_to_myst(text):
    """Converts GitHub maths syntax ($`x`$ and ```math fences) to MyST dollarmath ($x$ and $$)"""
    text = _GITHUB_BLOCK_MATH.sub(lambda m: f'{m.group(1)}$$\n{m.group(2)}\n{m.group(1)}$$', text)
    return _GITHUB_INLINE_MATH.sub(lambda m: '$' + m.group(1).strip() + '$', text)


def close_heading_level_gaps(text):
    """
    Renumbers Markdown headings so no level is skipped (e.g. H1 -> H3 becomes H1 -> H2)

    GitHub renders a skipped level happily; docutils cannot nest a section two levels down.
    Headings inside ``` fences are left alone.
    """
    lines, stack, in_fence = [], [], False
    for line in text.split('\n'):
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            level = len(match.group(1))
            while stack and stack[-1] >= level:
                stack.pop()
            stack.append(level)
            line = '#' * len(stack) + match.group(2)
        lines.append(line)
    return '\n'.join(lines)


def _convert_source(app, docname, source):
    source[0] = github_math_to_myst(source[0])
    if docname.startswith(GITHUB_PAGES):
        source[0] = close_heading_level_gaps(source[0])


def _convert_include(app, relative_path, parent_docname, content):
    content[0] = github_math_to_myst(content[0])


# ----------------------------------------------------------------------------------------------
# instances of third-party classes re-exported by levisim
# ----------------------------------------------------------------------------------------------
def _point_reexported_instances_to_their_docs(app, what, name, obj, options, lines):
    """
    Replaces the docstring of a module-level instance of a third-party class with a pointer to it

    autodoc shows such an instance (e.g. levisim.core.logger.logger, which is loguru's logger) with
    its class's docstring, written for that package's own documentation and full of substitutions
    that are undefined here. The levisim module's own docstring says what levisim does with it.
    """
    if what != 'data':
        return
    cls = type(obj)
    if cls.__module__.split('.')[0] in ('levisim', 'builtins', 'numpy'):
        return
    package = cls.__module__.split('.')[0]
    lines[:] = [f'An instance of ``{cls.__qualname__}`` from ``{package}``; see that package\'s documentation.']


# after close_heading_level_gaps is defined, which the gallery copies use
build_notebook_gallery()


def setup(app):
    app.connect('source-read', _convert_source)
    app.connect('include-read', _convert_include)
    app.connect('autodoc-process-docstring', _point_reexported_instances_to_their_docs)
