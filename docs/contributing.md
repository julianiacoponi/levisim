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

## Logging levels

levisim logs through loguru's `logger` (`from levisim.core.logger import logger`). Where the
records go (the console, a JSON-lines log file, optionally Elasticsearch) is configured in
`levisim/core/logger.py` and described in the README; by default the console shows WARNING and
above only, so notebooks stay quiet unless something needs attention. To see more, call
`set_console_level('INFO')` (or `'DEBUG'`, `'TRACE'`) after
`from levisim.core.logger import set_console_level`, or set `LEVISIM_LOG_LEVEL` before importing
levisim. Import the function by name: the attribute `levisim.core.logger` is the `logger` object
itself, because `levisim/core/__init__.py` re-exports it.

Choose the level by what the message tells a reader:

| Level | Use it for | Examples in levisim |
|---|---|---|
| `TRACE` | per-frequency or inner-loop detail | each Runge-Kutta step of a `FreeSpaceSimulation(rk_loop_debug=True)`; each temporary PNG deleted by the comparison helpers |
| `DEBUG` | intermediate values that help when debugging: parameters, the branch a method takes, term counts, run times | every `info_str` of a new `MultiTrap`; `Numerical H for MultiTrap(2×2): inverting the 4×4 derangement matrix D = I - R numerically at each ω; ...`; `⏱️ "MultiTrap.homodyne_spectrum" executed in 0.1125 s` |
| `INFO` | setup and progress milestones a user may want to follow | `Built MultiTrap(2×2): 4 dimensions (x1, y1, x2, y2) from toy parameters (Ω_model 'pairs'); ...`; `🧮 Calculating homodyne spectrum via LAM over 1600 ω points`; `Simulating 100000 time steps ...` and `Simulated 100000 samples ...`; loading or generating a power series |
| `SUCCESS` | a requested result produced, or a check passed | `✅ All relative difference checks passed`; `Test.notify_end` when the test passed |
| `WARNING` | suspicious input, ill-conditioning, fallbacks, deprecated usage: the code carries on, but a user should look | a drift matrix with an eigenvalue of positive real part (no steady state); a spectrum with NaN or infinite values; a linewidth estimate that fails and returns `None` |
| `ERROR` | a failure that is about to raise, or that is caught and swallowed | `TNT LAM only gives position power spectra ...` just before the `ValueError`; a temporary file that could not be deleted |

`CRITICAL` is not used. The conventions that go with the levels:

- **No `print()` for diagnostics.** `print` stays only where printing *is* the documented
  output: the `info` properties (which print their `info_str`), the `profiler` decorator's report
  and the console banners of `core.testing.Test`. `tests/test_log_levels.py` scans the package
  and fails on a `print` anywhere else.
- **Format lazily.** Pass the values as arguments to a constant message, not as an f-string:
  loguru only formats a record if some sink takes its level, so a call below every sink's level
  costs little more than the function call. For a message that is expensive to build, pass a
  callable through `logger.opt(lazy=True)`:

  ```python
  logger.debug('Δω = {:.3f} kHz over {} points', Δω / kHz, len(ω_range))
  logger.opt(lazy=True).debug('Parameters of {}:\n{}', lambda: self, lambda: self.info_str)
  ```

  Never pass a variable string (one holding user data or an `info_str`) as the message while
  also passing arguments, since its braces would be read as fields.
- **Keep DEBUG and INFO out of hot loops.** The log file records DEBUG and above in every run,
  so those calls are always formatted and written. Detail per frequency or per time step is
  `TRACE`, formatted lazily, and behind a flag if building its arguments costs anything (as with
  `rk_loop_debug`).
- **One record per event**, with the numbers and their units in it, rather than several lines.
  Values in levisim units print as `ω / kHz` (levisim's `kHz` is 2π·10³ rad/s, so this prints
  ordinary kHz).
- **Log before raising**: `logger.error(msg)` then `raise ValueError(msg)`, so the log file
  records the failure too.
- **Warnings must be cheap and must not change any result.** For example, the drift-matrix
  stability check is one `eigvals` of the (2N + 2)-square drift matrix per `MultiTrap`, and the
  spectrum methods check for non-finite values once per call, never per ω.
- `debug=True` arguments (`FreeSpaceSimulation(debug=True, rk_loop_debug=True)`,
  `SpectralDensityEstimate.fit_PSD(debug=True)`, `timer(..., debug=True)`) add DEBUG or TRACE
  records; they print nothing. Lower the console level to see them.

To check log output in a test, add a temporary sink and remove it afterwards, as
`tests/test_log_levels.py` does:

```python
records = []
handler_id = logger.add(lambda message: records.append(message.record), level='TRACE')
try:
    mtrap.homodyne_spectrum(method)
finally:
    logger.remove(handler_id)
assert not [r for r in records if r['level'].no >= logger.level('WARNING').no]
```

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
  pointing to `DISCLAIMER.md` (see [Provenance](provenance.md)).

## Releasing

Releases go to PyPI from GitHub Actions (`.github/workflows/release.yml`) by [trusted
publishing](https://docs.pypi.org/trusted-publishers/), so no API token is stored anywhere: PyPI
trusts uploads that come from that workflow file, in this repository, running in the GitHub
environment named `pypi`.

One-off setup, by the repository owner:

1. On pypi.org, *Your account → Publishing → Add a new pending publisher → GitHub*, fill in:
   PyPI project name `levisim`; owner `julianiacoponi`; repository name `levisim`; workflow name
   `release.yml` (the file name only, not its path); environment name `pypi`. The project is
   created on PyPI by the first successful upload.
2. In the repository's *Settings → Environments*, have an environment called `pypi`. Adding
   yourself as a required reviewer means every upload waits for your approval in the Actions run,
   and limiting its deployment branches and tags to `v*` tags means only a release can use it.
3. Optional, for rehearsals: the same on test.pypi.org (a separate account), with environment name
   `testpypi`, and a `testpypi` environment in the repository. A rehearsal runs from the branch
   `main` (not from a tag), so that environment must allow the branch `main`; with only a `v*` tag
   rule GitHub refuses the upload job ("Branch "main" is not allowed to deploy to testpypi").

Each release:

1. Bump `version` in `pyproject.toml` (`levisim.__version__` follows it once installed), add a
   dated `## [X.Y.Z] - YYYY-MM-DD` entry to `CHANGELOG.md`, and update `version` and
   `date-released` in `CITATION.cff`. `tests/test_packaging.py` checks that these agree on every
   push, and the release workflow refuses to upload if they do not.
2. Optionally rehearse: *Actions → release → Run workflow* on `main` builds, checks and uploads to
   TestPyPI; then `pip install -i https://test.pypi.org/simple/ --extra-index-url
   https://pypi.org/simple/ levisim`. TestPyPI, like PyPI, never accepts the same version twice.
3. Merge to `main`, then *Releases → Draft a new release*: create the tag `vX.Y.Z` on `main`, title
   it `levisim X.Y.Z`, and publish. Publishing runs the workflow, which builds the sdist and wheel,
   checks them (`twine check --strict`, a clean install, the version checks above) and, after your
   approval if the environment asks for it, uploads them to PyPI. The upload step also signs each
   file with a digital attestation (provenance tying it to this workflow and commit), which PyPI
   shows on the project page; the "Generating and uploading digital attestations" notice in the
   run is that, not a problem.
4. Check the release: `pip install --upgrade levisim` in a fresh environment, then
   `python -c "import levisim; print(levisim.__version__)"`.

The package ships the three smallest power-series pickles (`levisim/multiphysics/
power_series_data/`, about 270 KB) so `load_in_power_series=True` works from a pip install;
regenerate them with `python scripts/generate_power_series_pickles.py` (and confirm with
`--check`) if the series code changes. The `package` job in the tests workflow builds the wheel on
every push, installs it in a clean environment and loads a series from it.

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
python docs/check_rendered_maths.py docs/_build/html
```

The last line catches what Sphinx does not warn about: maths that MyST did not recognise as
maths, which is built into the page as prose typeset as maths and LaTeX printed as text. The two
causes so far are a `$$` block with no blank line before it (inside a list item it must also be
indented to the item's text), and a `|` inside maths in a table cell (write `\lvert x \rvert`).

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
fenced `math` blocks): `docs/conf.py` converts the inline maths as each file is read and has the
fenced blocks read as `math` directives, so the same file renders on GitHub and on the site.

**Publishing.** `.github/workflows/docs.yml` builds the site on every pull request, and on a
push to `main` also deploys it to GitHub Pages at <https://julianiacoponi.github.io/levisim/>.

```{include} ../README.md
:start-after: <!-- MIXINS-START -->
:end-before: <!-- MIXINS-END -->
:heading-offset: 1
```
