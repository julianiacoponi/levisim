# Demo notebooks

Short tutorials, each self-contained and runnable in under a minute on a laptop. They build on the
documentation pages linked from each notebook, and they keep their (light) outputs in git, so they
can be read on GitHub without running them (notebook metadata `"keep_output": true`, which
`nbstripout` honours).

| notebook | what it shows | runtime |
|---|---|---|
| [`01_quickstart_cavity_spectra`](01_quickstart_cavity_spectra.ipynb) | build a two-particle `MultiTrap` from physical parameters; homodyne, heterodyne and position spectra; normal modes and the optical spring; checking that a steady state exists, and why the built-in tweezer phases have none | ~5 s |
| [`02_seven_methods_agree`](02_seven_methods_agree.ipynb) | the same spectra by all seven quantum-linear-theory routes, their agreement (≲ 10⁻¹³) and cost, and which to use when | ~10 s |
| [`03_stochastic_free_space`](03_stochastic_free_space.ipynb) | seeded Langevin simulation of a bead in a free-space tweezer against `SpectralDensityTheory`: equipartition, PSD fits, the directed-force cross spectrum, the detection floor | ~20 s |
| [`04_two_tone_entanglement`](04_two_tone_entanglement.ipynb) | $x$-$y$ entanglement of one particle under a two-tone drive at the Deplano et al. (2025) parameters: one tone never entangles, $E_N$ vs blue/red ratio, the threshold cooperativity | ~10 s |
| [`05_derangements_and_hybridisation`](05_derangements_and_hybridisation.ipynb) | hybridisation functions, the derangement power series and its factorial cost, truncation by order, and the rank-2 structure behind the closed form | ~5 s |
| [`06_parameter_sweeps`](06_parameter_sweeps.ipynb) | spectra and occupations against cavity detuning; spectra and stability against the tweezer phase of a second particle, with levisim's stability WARNING | ~5 s |

Runtimes are rough figures for a laptop (the stored outputs took 2-17 s each to compute, plus a second or two to
import levisim); on a busy machine they can be several times longer. Run the notebooks from a clone with
`pip install -e .`, or with `PYTHONPATH` set to the repository root.

The larger worked example is in [`../showcase`](../showcase/README.md).
