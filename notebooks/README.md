# levisim notebooks

Jupyter notebooks that use levisim, from short demos to the research notebooks behind the thesis.
They import levisim from the repository, so install it first (`pip install -e .` from the repository
root) and run them from their own folder; any files they write go to an `output/` folder next to
them, which git ignores.

| Folder | What is in it | Outputs in git |
| --- | --- | --- |
| [`demos/`](demos/) | Short, task-focused walk-throughs of the levisim API, each a few minutes to read and run. Rendered in the docs [notebook gallery](../docs/notebooks.md). | kept (light) |
| [`showcase/`](showcase/) | One notebook touring what levisim can do. Rendered in the docs gallery. | kept (light) |
| [`papers/`](papers/) | Figures from published papers, recomputed with levisim. Rendered in the docs gallery. | kept (light) |
| [`multiphysics/`](multiphysics/) | Research notebooks on several levitated particles in one cavity: hybridisation and its power series, phase dependence, the mechanical mode comb. `multiphysics/power-series/pickle/` also holds the three power-series pickles that the test suite and `MultiTrap(load_in_power_series=True)` read (regenerate them with `scripts/generate_power_series_pickles.py`). | stripped |
| [`physics/`](physics/) | Single-particle physics: the focal field of a lens (`focussing_optics_plots`) and spectral analysis of a simulated trace (`simulation_analysis`). | stripped |
| [`archive/`](archive/) | Older research notebooks, kept as a record and updated to run on current levisim: `animations/` (spectral heatmaps and animations), `hybridisation/` (hybridisation functions of four-dimensional systems, a uniform-hybridisation approximation), `mechanical-mode-comb/` (supplementary-information figures). Each opens with a cell saying what changed in the update. | stripped |

## Conventions

- **Title cell.** Every notebook starts with a markdown cell saying what it shows and roughly how
  long it takes to run.
- **Outputs.** `nbstripout` (see `.pre-commit-config.yaml`) strips outputs on commit, except in
  notebooks whose top-level metadata has `"keep_output": true`, which the demo, showcase and paper
  notebooks set. Kept outputs stay light: figures at dpi ≤ 100, no embedded animations, ideally
  under 1.5 MB per notebook.
- **Animations** are written to `output/` (mp4 with ffmpeg, otherwise gif), never into the
  notebook. Notebooks that animate have a `MAKE_ANIMATIONS = False` flag at the top: `False`
  gives a quick static preview (a few frames, coarser grids), `True` computes every frame and
  writes the files.
- **Reproducible.** Random numbers are seeded; no absolute paths.
- **Figures** use levisim's style, `set_plt_rcParams_to_xmgrace_style()` from
  `levisim.core.plotting`, with curves running to the left and right edges of the axes.
  Frequencies are angular in levisim (`kHz` is 2π × 10³ rad/s), so divide by `kHz` to plot
  ordinary kHz.

## Smoke tests

[`smoke_list.txt`](smoke_list.txt) lists the notebooks that the `notebooks` GitHub workflow
(`.github/workflows/notebooks.yml`) executes on every pull request that touches `notebooks/` or
`levisim/`, each with a time limit. Add a notebook there if it runs in about a minute.

## Run times of the archive and physics notebooks

Measured end to end in an in-process IPython runner in Nov 2026, with the committed (quick) flag
settings. The machine was shared and heavily loaded during most of these runs, so the longer
times are upper bounds; the CPU time is a better guide where it was recorded.

| Notebook | Wall time |
| --- | --- |
| `archive/animations/homodyne_slice` | 4 s |
| `archive/animations/spectral_animations` | 13 s |
| `archive/animations/spectral_heatmap_single_frame` | 14 s |
| `archive/animations/3x2_truncated_heatmap` | 38 s |
| `archive/animations/spectral_heatmap_animations` | 45 s |
| `archive/hybridisation/uniform_hybridisation_approximation` | 111 s |
| `archive/hybridisation/2x2_system` | 306 s (about 60 s of CPU) |
| `archive/mechanical-mode-comb/supp_info_figures` | 781 s (about 200 s of CPU) |
| `archive/mechanical-mode-comb/unequal_couplings_supp_info` | 78 s (on a lightly loaded machine) |
| `physics/focussing_optics_plots` | 177 s |
| `physics/simulation_analysis` | 54 s |
