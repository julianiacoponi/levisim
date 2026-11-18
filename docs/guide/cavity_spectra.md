# Spectra of several particles in one cavity

{class}`~levisim.multiphysics.multitrap.MultiTrap` is the entry point for everything on this
page. Its notation and conventions are on the [Conventions](../concepts/conventions.md) page.

## Building a system

A `MultiTrap` is defined by the number of particles, the number of Cartesian axes kept per
particle (1 = $x$; 2 = $x, y$; 3 = $x, y, z$), the gas, the cavity, and the frequency grid:

```python
import numpy as np
from levisim.core.units import kHz
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs

mtrap = MultiTrap(
    num_traps=3,
    Cartesian_axes_per_trap=2,
    gas_kwargs=DEFAULT_gas_kwargs,
    cavity_kwargs=DEFAULT_cavity_kwargs,
    ω_range=np.arange(100, 200, 0.1) * kHz,
)
```

Each motion is labelled by axis and particle, `'x1'`, `'y1'`, `'x2'`, ..., and has a
{class}`~levisim.multiphysics.dimension.Dimension` in `mtrap.dimensions[label]` that holds its
frequency `Ω`, damping `Γ`, coupling and susceptibility. `mtrap.info`, `mtrap.info_per_trap` and
`mtrap.info_per_dimension` print summaries.

There are two ways to set the physical parameters:

- **From the physics** (the default). Each particle is a silica bead in its own tweezer, and the
  trap frequencies and the linearised coherent-scattering couplings follow from the tweezer waist
  and power, the bead and its position in the cavity standing wave. The tweezers and beads are
  set in `MultiTrap.setup_levitated_physics` (up to nine particles, all at the cavity node).
- **From toy parameters**, by passing `toy_params=dict(...)`: the frequencies follow an
  `Ω_model` (`'pairs'`, `'equal spacing'`, `'equal spacing: simple'`, ...), and the couplings and
  tweezer phases are given directly (`gjn_dict`, `φn_dict`, or `toy_gx`, `toy_gy`, `toy_gz`).
  {mod}`levisim.multiphysics.utils.shelf` holds the sets used by the tests and the Fortran
  comparison, e.g. `FORTRAN_toy_params`, `PHASE_EQUAL_toy_params`, `PHASE_DEPENDENT_toy_params`
  and `OFF_NODE_toy_params_pairs`.

**Choosing `ω_range`.** Spectra are evaluated point by point on this grid (angular frequency;
see [Units](../concepts/conventions.md#units)). The mechanical peaks are only as wide as their
optically damped linewidths - about 1 kHz at the defaults - so the step needs to be well below
that; the default grid is $-400$ to $400$ kHz in 0.5 kHz steps. Include negative frequencies
when the sideband asymmetry or an integral over the spectrum matters.

## Optical spectra

```python
from levisim.multiphysics.enums import QuantumLinearTheoryMethod

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
S_hh = mtrap.homodyne_spectrum(LAM)        # at the angle mtrap.cavity.θhom
S_het = mtrap.heterodyne_spectrum(LAM)
```

Both are one value per point of `mtrap.ω_range`, in shot-noise units.

## Mechanical spectra

```python
psd_array, psd_dict = mtrap.all_position_power_spectra(LAM)          # shape (len(ω_range), N)
S_x1x1 = mtrap.position_power_spectrum('x1', LAM)
S_x1x2 = mtrap.position_cross_spectrum('x1', 'x2', LAM)              # complex
```

All three return spectra of the dimensionless positions $\hat q_{j_n}$ by default; pass
`zpf_scaled=False` to multiply by $x_{\text{zpf}}^2$ (see
[Spectral densities](../concepts/conventions.md#spectral-densities) for the normalisation and
how to convert to a one-sided PSD). `all_position_power_spectra` is the efficient choice when
more than one motion is needed: the single-position functions rebuild the full solution for each
call.

## Cross-checking with other methods

Every spectrum function takes the method as its argument, so the same spectrum can be
recalculated by an independent route and compared. {mod}`levisim.multiphysics.utils.comparison`
has the comparator the tests use:

```python
from levisim.multiphysics.utils.comparison import assert_arrays_agree_within_threshold

S_hh_hyb = mtrap.homodyne_spectrum(QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION)
assert_arrays_agree_within_threshold(
    mtrap,
    {'LAM': S_hh, 'numerical hybridisation': S_hh_hyb},
    difficulty_level='easy',
)
```

It compares every pair in the dict and raises `RelativeDifferenceTooHigh` if any pair disagrees
beyond the threshold for that `difficulty_level`. The default, `'very hard'`, is a relative
difference of $10^{-12}$, which double-precision linear algebra does not always reach; pick the
level to suit the comparison.

Which methods support which spectra, and how closely they agree, is on the
[Calculation methods](../concepts/calculation_methods.md) page.

## Time-domain (covariance) description

The same drift and noise matrices give the steady-state covariance matrix by the Lyapunov
equation, in {func}`levisim.multiphysics.covariance.steady_state_covariance`. That is the starting
point of the two-tone entanglement studies, which extend `MultiTrap` with several tweezer tones
({mod}`levisim.multiphysics.multitone`, {mod}`levisim.multiphysics.two_tone`).
