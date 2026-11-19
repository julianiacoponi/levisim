# Disclaimer regarding use of LLMs

The first paragraph below is the repository owner's own. The dated sections after it record which parts of the repository Claude (Anthropic) has since written at his request, and what those sessions found; each describes the repository as it was at the time.

While in early 2025 I dabbled with using ChatGPT and DeepSeek to give me starting points for certain more generic parts of this code (help with wrappers etc.) the only thing that remains of legitimate value that I gained from using those tools is the first attempt at the `docker-compose.yaml`, which I otherwise would have likely laboured a bit harder to do. Otherwise this code was proudly written without any LLM generated code :-)

## October 2026 update

That remains true of every commit on `main`. In one session I pointed Claude (Anthropic) at this repository, the untracked `notebooks/` folder and the `fortran/` folder, and asked it to propose fixes as pull requests. It opened a stack of them, and **this update and the two sections that then sat above it in the README** (the code pathways of the noise vectors, now on the documentation's *Calculation methods* page, and the planned `multitrap` mixins refactor) **are themselves part of that work** - so the claim above no longer covers the whole repository if any of those branches are merged. Each pull request description carries the same note.

For the record, the things it found that I had not:

- `pyproject.toml` shipped no subpackages, so `pip install levisim` produced a package that raised `ModuleNotFoundError: No module named 'levisim.core'` on import. Working from the repo root hid it completely.
- `import levisim` could not succeed on a fresh clone at all: `shelf.py` built 17 `MultiTrap` instances at import time, several of which read power-series pickles from the untracked `notebooks/`.
- Three latent `NameError`s - `Megasamples` for `megasamples`, a missing `colours_50` import, and a bare `k` where `self.k` was meant - each making its enclosing function raise on any call. None was reachable from anywhere in the repo, which is why they had survived.
- `core/logger.py` added only an Elasticsearch sink and then logged at import, so with no ELK stack running every `import levisim` printed a traceback and **no `logger` call has ever reached the console**. That is why there were 83 `print()` calls in here (the November 2026 log-level pass routed all but the 22 that are documented output through the logger).
- Nine `cmplx()` calls in the Fortran had no `kind` argument, so every complex susceptibility $`\chi`$ was computed in **single precision** and widened back to double - a median relative error of 2e-7 in the reference data.
- Three of the four build lines in the old `fortran/README.txt` did not compile, because the `num_traps` rename only reached three of the files.
- `physics/kick.py` defined `info` twice, so `Kick.info` returned the string instead of printing it.
- The comparison helpers default to `difficulty_level='very hard'`, a relative-difference threshold of 1e-12 that double-precision linear algebra cannot reach. This is why the test notebooks reported hundreds of failing method pairs that were not real disagreements - though the analytic-vs-rest disagreement on the **spectra** (mean 28-147%) is real, and is the thing the "BROKEN!" heading in `test_all_systems.ipynb` was correctly pointing at.

It also could not fix, and I should: the `kappa_nano (Hz) 1.216e-86` that `NParticle.f90` prints, and the two `NOT SURE: CHECK!!!` susceptibilities in `3DEQRom.f90`.

In a later session I asked it for a study of whether a two-tone drive can entangle the transverse motions. The four modules listed under *Steady-state entanglement of the transverse motions* in the README, their tests (`tests/test_gaussian_states.py`, `tests/test_two_tone.py`) and `docs/two_tone_entanglement/` are its work too, and the pull request that adds them says so.

A third session followed that up, on the tones' polarisation, mech-mech coupling, the node position and a second particle. Its work includes:
- `levisim.physics.focal_expansion` and `levisim.multiphysics.two_tone_traps`.
- The per-tone couplings, `QuadraticPotential` terms and stored monodromy in `multitone`.
- Their tests (`tests/test_focal_expansion.py`, `tests/test_two_tone_traps.py`).
- `docs/two_tone_entanglement/polarisation_and_two_particles.*`.

Its pull request says so too.

A fourth session, in November 2026, polished the repository as a whole. Its work includes:
- The general sections at the top of the README (overview, installation, quickstart, layout, tests, docs), the move of the calculation-method pathways from here into the documentation, the Sphinx documentation site under `docs/`, and the GitHub Actions workflows under `.github/`.
- The resolution of the remaining `TODO` comments, either by implementing them or by answering them in place.
- The generalisation of the analytic methods (6) and (7) to include the intra-particle couplings $`g_{jk}`$.
- The section of the README on stochastic simulations of a bead in a free-space tweezer, and its documentation page.

Each of its pull requests says so too.

A fifth session, also in November 2026, followed up the owner's review of the fourth. Its work includes:
- This file. Every passage about LLM use moved here from the README, which kept a one-line link, and the README lost the notes that no longer applied (the two `levisim` directories, the GitHub Pages and editable-install caveats).
- Logging: the always-on JSON-lines log file and the WARNING-level console sink in `levisim/core/logger.py`, the `levisim-logs` command (`levisim/core/elk.py`) that installs and runs Elasticsearch and Kibana without Docker, the rewritten `docker-compose.yaml` fallback, `docs/guide/logging.md`, and their tests (`tests/test_logger.py`, `tests/test_elk.py`).
- A pass over the log messages across the package, setting each to the level convention in `docs/guide/logging.md`.
- The flowchart of the seven calculation methods, and the edge-to-edge versions of the README's figures.
- The notebooks: tracked in git, updated to the current API or deleted where obsolete, with new demo, showcase and paper-reproduction notebooks, a CI smoke run and a gallery on the documentation site.
- An assessment of whether to publish `levisim` on PyPI, and the packaging work that follows from it.

Each of its pull requests says which of these it contains.
