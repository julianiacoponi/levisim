# Contributing

## Code style

The convention is mild PEP 8, in service of human readability rather than rule compliance:

- No automatic formatting and no line-length enforcement. Aligned matrix rows and grouped
  constants are deliberate and should stay aligned.
- Unicode identifiers are welcome where they match the physics (`Ω`, `Γ`, `κ`, `Δ`, `ω_range`).
  Mechanical frequencies are capital `Ω`; `ω` is the spectral variable.
- Physics classes describe themselves through an `info_str` property and an `info` property that
  prints it.
- Every public function and class has a docstring: a one-line summary, then the explanation and
  the equations it implements, then a `[units]` line where relevant (e.g. `[2π·Hz]` for an
  angular frequency). The [API reference](api.md) is generated from these, through
  `sphinx.ext.napoleon`, so NumPy- or Google-style `Parameters` / `Returns` sections render as
  tables.
- Write conjugate susceptibilities out as `χ*(-ω)`, not with a bar.

## pre-commit and ruff

```sh
pip install pre-commit && pre-commit install
pre-commit run --all-files          # one-off over everything
```

`.pre-commit-config.yaml` runs `ruff` with a deliberately narrow rule set, and without `--fix`:

| Enabled | Catches |
|---|---|
| `F821` | undefined name |
| `F811` | redefinition of an unused name |
| `F541` | f-string without placeholders |
| `F632` | `is` comparison with a literal |
| `E711`, `E712`, `E714` | comparison to `None` / `True` / `False` with `==`, and `not x is y` |
| `E722` | bare `except` |

These are the rules that catch real bugs. Left out, on purpose:

- `F401` (unused import) - `core/constants.py` and the `__init__` files re-export on purpose,
  and auto-fixing would delete the package's physics symbols.
- `F403`/`F405` (`import *`) - the wildcard re-exports are the package's chosen structure; each
  module declares `__all__`, so the surface is bounded.
- `F841` (unused local) - almost every instance is unfinished work or self-documenting tuple
  unpacking.
- All other `E` and `W` (pycodestyle) rules - no line-length or whitespace enforcement.

The other hooks strip notebook outputs (`nbstripout`), fix trailing whitespace and file endings
(except in the fixed-width Fortran `.dat` files), check YAML, catch merge-conflict markers and
keep line endings LF. To run the same ruff check by hand:

```sh
ruff check --select F821,F811,F541,F632,E711,E712,E714,E722 levisim tests
```

## Commits and pull requests

- One commit per pull request, with a lowercase imperative subject in the style of the history
  ("fix ...", "add ...") and a body that says what changed, why, and how it was checked.
- Pull requests are stacked: each branch starts from the one before it and is merged in order.
- Changes written with an LLM say so in the commit message and the pull request description,
  pointing to the README's *Disclaimer regarding use of LLMs* (see [Provenance](provenance.md)).

## Building these docs

```sh
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

The site must build without warnings, from the pages and from the docstrings of `levisim` alike
(the API reference is generated from them). The docs workflow writes a warnings log and checks it
with `docs/check_sphinx_warnings.py`, which fails on any warning - the equivalent of
`sphinx-build -W --keep-going` - and lists the docstring warnings by object, so they are easy to
find:

```sh
sphinx-build -b html docs docs/_build/html -w docs/_build/warnings.log
python docs/check_sphinx_warnings.py docs/_build/warnings.log
```

The docstrings are mostly free-form physics notes rather than numpydoc, and they stay that way;
they only need to be valid reStructuredText. The usual fixes are markup only:

- a blank line before a list, and between a one-line summary and what follows it;
- an aligned derivation or matrix (where the layout is the point) as a literal block: end the
  line before it with `::` (`text::` renders as `text:`, and `text ::` as `text`), leave a blank
  line, and indent the block;
- an inline expression containing `|...|` (read as a substitution), `*` or a trailing `_` (read
  as emphasis or a link, e.g. `G_-`) in double backticks, e.g. ` ``|χ(ω)|²`` `, or as
  `` :math:`...` ``;
- a section underline at least as long as its title.

Every public module, class and function also needs at least a one-line summary docstring.

Set `LEVISIM_DOCS_OFFLINE=1` to skip fetching the Python, NumPy and SciPy inventories that link
type names to their documentation. The API pages are generated at build time into
`docs/_autosummary/` (git-ignored).

The pages are Markdown, parsed by MyST: `$...$` and `$$...$$` for maths, ` ```{directive} `
or `:::{directive}` fences for Sphinx directives. Pages written for GitHub - the README and the
two-tone study - keep GitHub's maths syntax (a dollar sign and a backtick on each side inline, and
fenced `math` blocks), and `docs/conf.py` converts it as each file is read, so the same file
renders on GitHub and on the site.

**Publishing.** `.github/workflows/docs.yml` builds the site on every pull request, and on a
push to `main` also deploys it to GitHub Pages at <https://julianiacoponi.github.io/levisim/>.
Deployment needs a one-time setting in the repository: **Settings → Pages → Build and
deployment → Source: GitHub Actions**. Until that is set, the build step still runs and the
deploy step fails. Note that GitHub Pages on a private repository needs a paid plan, and the site
of a private repository is public unless the plan allows restricting it.

```{include} ../README.md
:start-after: <!-- MIXINS-START -->
:end-before: <!-- MIXINS-END -->
:heading-offset: 1
```
