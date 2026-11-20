# levisim notebooks

Jupyter notebooks that use levisim, from short demos to the research notebooks behind the thesis.
They import levisim from the repository, so install it first (`pip install -e ".[notebooks]"` from the
repository root, an editable install with JupyterLab) and run them from their own folder. They follow
`main`; to run them against a release instead, check out its tag (e.g. `git checkout v0.1.0`) first; any files they write go to an `output/` folder next to
them, which git ignores.

| Folder | What is in it | Outputs in git |
| --- | --- | --- |
| [`demos/`](demos/) | Short, task-focused walk-throughs of the levisim API, each a few minutes to read and run. Rendered in the docs [notebook gallery](../docs/notebooks.md). | kept (light) |
| [`showcase/`](showcase/) | One notebook touring what levisim can do. Rendered in the docs gallery. | kept (light) |
| [`papers/`](papers/) | Figures from published papers, recomputed with levisim. Rendered in the docs gallery. | kept (light) |
| [`multiphysics/`](multiphysics/) | Research notebooks on several levitated particles in one cavity: hybridisation and its power series, phase dependence, the mechanical mode comb. The three power-series pickles that `MultiTrap(load_in_power_series=True)` reads ship with the package in `levisim/multiphysics/power_series_data/` (regenerate or extend them with `scripts/generate_power_series_pickles.py`). | stripped |
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
`levisim/`, each with a time limit: the demos, the paper reproductions, and every research or
archive notebook that runs in about half a minute or less (29 notebooks, about 4 minutes of
execution on a laptop). Add a notebook there if it runs that fast.

## Every notebook

Runtimes are the execution times measured in Nov 2026 with the committed (quick) flag settings, on
an 8-core laptop that was also running other work, rounded up; importing levisim adds a second or
two. ✓ marks the notebooks in the CI smoke run ([`smoke_list.txt`](smoke_list.txt)).

**[`demos/`](demos/)** (outputs kept)

| Notebook | What it shows | Runtime | CI |
| --- | --- | --- | --- |
| [`01_quickstart_cavity_spectra`](demos/01_quickstart_cavity_spectra.ipynb) | two particles in one cavity from physical parameters: spectra, normal modes, optical spring, and the steady-state check | ~5 s | ✓ |
| [`02_seven_methods_agree`](demos/02_seven_methods_agree.ipynb) | the seven quantum-linear-theory routes on three systems: agreement, cost, which to use | ~10 s | ✓ |
| [`03_stochastic_free_space`](demos/03_stochastic_free_space.ipynb) | seeded Langevin simulation of a bead in a free-space tweezer against the spectral theory | ~20 s | ✓ |
| [`04_two_tone_entanglement`](demos/04_two_tone_entanglement.ipynb) | $x$-$y$ entanglement of one particle under a two-tone drive | ~10 s | ✓ |
| [`05_derangements_and_hybridisation`](demos/05_derangements_and_hybridisation.ipynb) | hybridisation functions, the derangement power series and its cost, the rank-2 structure | ~5 s | ✓ |
| [`06_parameter_sweeps`](demos/06_parameter_sweeps.ipynb) | detuning and tweezer-phase sweeps, with the stability check | ~5 s | ✓ |

**[`showcase/`](showcase/)** (outputs kept)

| Notebook | What it shows | Runtime | CI |
| --- | --- | --- | --- |
| [`many_particles_one_cavity`](showcase/many_particles_one_cavity.ipynb) | a 100-particle comb in one cavity mode: superradiant and subradiant modes, cross-checked by independent routes | ~1.5 min | |

**[`papers/`](papers/)** (outputs kept; see [`papers/README.md`](papers/README.md))

| Notebook | What it shows | Runtime | CI |
| --- | --- | --- | --- |
| [`delic2020_ground_state_cooling`](papers/delic2020_ground_state_cooling.ipynb) | Delić et al., Science 367, 892 (2020): ground-state cooling by coherent scattering | ~5 s | ✓ |
| [`piotrowski2023_two_mode_ground_state`](papers/piotrowski2023_two_mode_ground_state.ipynb) | Piotrowski et al., Nat. Phys. 19, 1009 (2023): two modes in the ground state | ~5 s | ✓ |
| [`vijayan2024_cavity_mediated_interactions`](papers/vijayan2024_cavity_mediated_interactions.ipynb) | Vijayan et al., Nat. Phys. 20, 859 (2024): cavity-mediated coupling of two particles | ~5 s | ✓ |
| [`pontin2025_pair_dark_modes`](papers/pontin2025_pair_dark_modes.ipynb) | Pontin et al., PRL 135, 073602 (2025): strong coupling and dark modes of two particles in a bichromatic tweezer | ~60 s | ✓ |
| [`pontin2026_optical_squeezing`](papers/pontin2026_optical_squeezing.ipynb) | Pontin et al., Phys. Rev. A 114, 053704 (2026): sub-shot-noise light from two ground-state modes, squeezing maps against phase and frequency | ~5 s | ✓ |
| [`deplano2026_stationary_entanglement`](papers/deplano2026_stationary_entanglement.ipynb) | Deplano et al., Science 394, 113 (2026): entanglement of the motion with a propagating (filtered) output mode of a second cavity mode | ~10 s | ✓ |

**[`multiphysics/`](multiphysics/)** (outputs stripped)

| Notebook | What it shows | Runtime | CI |
| --- | --- | --- | --- |
| [`hybridisation/animate_hybridisation_solutions`](multiphysics/hybridisation/animate_hybridisation_solutions.ipynb) | det(**D**), adj(**D**) and the hybridisation solutions of a 2×2 system as arrows in the complex plane | ~20 s | ✓ |
| [`hybridisation/animate_hybridisation_terms_grid`](multiphysics/hybridisation/animate_hybridisation_terms_grid.ipynb) | every hybridisation function and power-series term of a 2×2 system, one panel each | ~5 s | ✓ |
| [`hybridisation/visualise_eigenvalues_of_hybridisation_matrix`](multiphysics/hybridisation/visualise_eigenvalues_of_hybridisation_matrix.ipynb) | eigenvalues and eigenvectors of the hybridisation matrix of a 3×2 system against frequency | ~10 s | ✓ |
| [`mechanical-mode-comb/plot_vertical_stack_varying_1_parameter`](multiphysics/mechanical-mode-comb/plot_vertical_stack_varying_1_parameter.ipynb) | stacked heterodyne spectra of particle arrays, one parameter varied per panel | ~45 s | |
| [`mechanical-mode-comb/reproduce_Figs_1_3_4_5`](multiphysics/mechanical-mode-comb/reproduce_Figs_1_3_4_5.ipynb) | the spectra of the mechanical-mode-comb paper (PRR 8, 2026; arXiv:2406.00366) | ~10 s | ✓ |
| [`phase-dependence/analytics_vs_numerics`](multiphysics/phase-dependence/analytics_vs_numerics.ipynb) | all seven QLT methods against each other on the standard systems | ~25 s | ✓ |
| [`phase-dependence/visualise_noise_vectors`](multiphysics/phase-dependence/visualise_noise_vectors.ipynb) | noise-vector components, numerical against analytic hybridisation | ~1.5 min | |
| [`phase-dependence/visualise_spectra`](multiphysics/phase-dependence/visualise_spectra.ipynb) | one spectrum by two QLT methods, overlaid, for unequal tweezer phases | ~10 s | ✓ |
| [`phase-dependence/zoo_of_phase_dependent_functions`](multiphysics/phase-dependence/zoo_of_phase_dependent_functions.ipynb) | the scalar functions behind the PRR 8 closed form, plotted and cross-checked | ~1 min | |
| [`power-series/animate_hybridisation_terms`](multiphysics/power-series/animate_hybridisation_terms.ipynb) | terms of the deranged determinant as arrows in the complex plane | ~5 s | ✓ |
| [`power-series/animate_spectra_vs_Rjk_power_series`](multiphysics/power-series/animate_spectra_vs_Rjk_power_series.ipynb) | spectra and truncated power series while following every hybridisation term | ~30 s | ✓ |
| [`power-series/hybridisation_number_theory`](multiphysics/power-series/hybridisation_number_theory.ipynb) | counting the power-series terms against closed forms (derangements, rencontres numbers) | ~5 s | ✓ |
| [`power-series/visualise_complex_functions`](multiphysics/power-series/visualise_complex_functions.ipynb) | the susceptibilities that make up the hybridisation functions, in the complex plane | ~5 s | ✓ |
| [`power-series/visualise_spectra_by_evaluation_order`](multiphysics/power-series/visualise_spectra_by_evaluation_order.ipynb) | spectra from the power series truncated at each order, against LAM | ~15 s | ✓ |

**[`physics/`](physics/)** (outputs stripped)

| Notebook | What it shows | Runtime | CI |
| --- | --- | --- | --- |
| [`focussing_optics_plots`](physics/focussing_optics_plots.ipynb) | the focal field of a lens (`FocussingOptics`): Bessel integrals, field maps, animations | ~1 min | |
| [`simulation_analysis`](physics/simulation_analysis.ipynb) | spectral analysis of a simulated free-space trace (PSDs, cross spectrum, fits) | ~35 s | ✓ |

**[`archive/`](archive/)** (outputs stripped)

| Notebook | What it shows | Runtime | CI |
| --- | --- | --- | --- |
| [`animations/3x2_truncated_heatmap`](archive/animations/3x2_truncated_heatmap.ipynb) | $x_1$ PSD of a 3×2 system against coupling, per truncation order of the power series | ~10 s | ✓ |
| [`animations/homodyne_slice`](archive/animations/homodyne_slice.ipynb) | homodyne spectrum of two beads at θhom = 0 and its minima in dB | ~5 s | ✓ |
| [`animations/spectral_animations`](archive/animations/spectral_animations.ipynb) | `animate_spectra` on a 3×1 system: slices and heatmaps | ~5 s | ✓ |
| [`animations/spectral_heatmap_animations`](archive/animations/spectral_heatmap_animations.ipynb) | spectral heatmaps of a 2×3 system over pairs of parameters | ~5 s | ✓ |
| [`animations/spectral_heatmap_single_frame`](archive/animations/spectral_heatmap_single_frame.ipynb) | heterodyne heatmap against detuning, with a bright-mode fit | ~10 s | ✓ |
| [`hybridisation/2x2_system`](archive/hybridisation/2x2_system.ipynb) | hybridisation functions of a four-dimensional system | ~15 s | ✓ |
| [`hybridisation/uniform_hybridisation_approximation`](archive/hybridisation/uniform_hybridisation_approximation.ipynb) | replacing every hybridisation function by one mean value | ~5 s | ✓ |
| [`mechanical-mode-comb/supp_info_figures`](archive/mechanical-mode-comb/supp_info_figures.ipynb) | supplementary-information heterodyne spectra of the mechanical-mode-comb work | ~1 min | |
| [`mechanical-mode-comb/unequal_couplings_supp_info`](archive/mechanical-mode-comb/unequal_couplings_supp_info.ipynb) | heterodyne spectra with unequal couplings | ~1.5 min | |
