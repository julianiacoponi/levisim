# Changelog

All notable changes to `levisim` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (0.x and pre-releases such as 1.0.0rc1: the API may still change;
from 1.0.0 the public API is stable within a major version).

## [Unreleased]

## [1.0.0rc1] - 2026-11-26

The first release candidate of levisim 1.0 ("round 5"). levisim now simulates levitated particles
beyond one bead in one optical tweezer: time-domain cavity simulations with output-light records,
interactions between particles, Paul and magnetic traps and their hybrids with tweezers,
non-spherical particles and materials, and rigid-body rotation. Install the pre-release with
`pip install --pre levisim` (add `[fast]` for the compiled numba kernels). The public API may still
change before 1.0.0; from 1.0.0 it follows semantic versioning.

### Changed defaults (results of 0.1.x change unless the opt-out is passed)
- `MultiTrap(static_cavity_shifts=True)` with physical parameters (opt-in in 0.1.1): the trap
  frequencies include the static stiffness of the cavity field. `static_cavity_shifts=False` gives
  the numbers of 0.1.x. `cavity_wavenumber` stays off by default.
- The static cavity field includes the tweezer phases e^{iφn} (`static_cavity_phases`), the
  dynamical coupling of the cavity dispersive term and the static cavity pull of the detuning
  (`cavity_dispersive_coupling`), the cavity mode envelope (`cavity_envelope`), and the
  radiation-reaction corrected Re α_eff in every cavity term (`cavity_effective_polarisability`;
  −5.4 × 10⁻⁵ on every cavity coupling, also at the node). All four keywords off together with
  `static_cavity_shifts=False` reproduce 0.1.1 bit for bit.
- Measured frequencies (`override_Ω_dict`) receive neither the static cavity shift nor the
  interaction self-stiffness (`interaction_frequency_shifts='all'` restores the latter).
- `MultiTrap` adds the intra-trap off-diagonal stiffness of combined force elements
  (`intra_trap_couplings`) and uses the exact Floquet frequencies for hybrid traps with one Paul
  trap (`paul_exact_frequencies`). Plain tweezer traps are unchanged.
- `Trap.Utw`, `U_depth` and `Tweezer.potential` are the cycle-averaged potential −(Re α/4)|E|², half
  the previous values (a factor-2 bug); `v_escape` and `p_escape` fall by √2. Forces, frequencies and
  spectra are unchanged.
- `Trap(fields=...)` defaults to directed force noise β = 0 (the tweezer shorthand `Trap()` keeps
  √2).
- `finite_size_torque=True` for non-spherical particles (spheres unchanged).
- `backend='auto'` (numba installed) compiles free-space runs from 2¹⁴ steps (was 2¹⁵) and the linear
  cavity model up to state dimension 384 (was 128), re-measured on a quiet machine
  (`benchmarks/auto_backend_thresholds.py`); it now also compiles Paul, magnetic, harmonic and
  electric force elements, optical binding and pair interactions. Seeded runs then differ from NumPy
  at round-off; `backend='numpy'` is the reference. An explicit `backend='numba'` for a configuration
  without a compiled kernel raises `UnsupportedBackendError` instead of running NumPy.
- `OpticalBinding` uses each tweezer's Jones polarisation (`polarisation='y'` for the old
  assumption).
- `magnetic_trap_report` takes the trap depth from a saddle-point search
  (`trap_depth_method='rays'` gives the previous ray estimate).
- `Sphere.α` of an anisotropic material is the scalar projection on the tweezer polarisation, as
  for every other shape; the tensor is `α_tensor`. Isotropic spheres are unchanged.
- `BISMUTH` has an electrical conductivity, so `magnetic_trap_report` includes its eddy-current
  damping (`BISMUTH.replace(conductivity=None)` for the previous report).
- `Simulation` is a deprecated alias of `FreeSpaceSimulation`, to be removed in 2.0.

### Added
- Simulations (`levisim.simulation`): `StochasticSimulation` base with integrator objects, seeding
  and chunked noise; `FreeSpaceSimulation`; optional numba backend (`levisim[fast]`) with a threading
  policy; per-run NumPy force dispatch (30-42 % faster per step).
- `CavitySimulation` (`MultiTrap.simulate`): time traces of the cavity output and the particle
  positions. `model='linear'` is exact (Van Loan discretisation); `model='nonlinear'` integrates the
  full optical potential and the cavity amplitude (split step, `NonlinearCavityModel` with Newton
  equilibrium and Jacobian). Records `Homodyne`, `Heterodyne`, `PhotonNumber`, `Position`
  (sample-averaged or `sampling='instantaneous'`), detection efficiency, LO phase noise and an
  opt-in static carrier, electronic floor, voltage conversion; Fabry-Perot ports
  (`Cavity(κ_ports=..., κ_loss=...)`); ensembles; theory spectra of every record;
  `SpectralDensityEstimate` with an ensemble axis.
- Force elements: a `Trap` is a sum of elements (`Tweezer`, `HarmonicPotential`, `Gravity`,
  `Feedback`, `PhotonRecoil`, `PaulTrap`, `UniformElectricField`, `MagneticTrap`,
  `EddyCurrentDamping`, `MeissnerImagePlane`), each with force, torque, potential, stiffness, noise
  and damping.
- Paul traps: ring/endcap and linear presets, Mathieu analysis (lowest, next and exact Floquet
  secular frequencies, stability, growth rates), trap depth, micromotion, electrode-noise heating;
  hybrid Paul-optical traps.
- Magnetic levitation: field sources (`CurrentLoop`, `CoilPair`, `CuboidMagnet`, `MultipoleField`,
  `LocalFieldExpansion`, `FieldSum`, `CallableField`), diamagnetic and magneto-gravitational traps
  with demagnetisation, eddy-current damping, saddle-point trap depth, and magnets above
  superconductors (`MeissnerImagePlane`).
- Interactions between particles (`levisim.physics.interactions`): `Coulomb`, `NewtonianGravity`,
  `Casimir` (Casimir-Polder and proximity-force limits) and `OpticalBinding` (full dipole Green
  tensor, Gouy phase, first-order and self-consistent fields); `MultiTrap(interactions=[...])` with
  general (also non-reciprocal) mechanical coupling matrices, `MultiParticleFreeSpaceSimulation`, and
  entanglement outputs (logarithmic negativity).
- Materials and particles: `Material` and a cited materials library; `Particle` with `Sphere` (and
  `Bead`, kept permanently), `Ellipsoid`, `Spheroid`, `Cylinder`/`Rod`/`Disk`, `Dumbbell`,
  `TabulatedParticle`; anisotropic polarisability tensors, orientation, gas damping for any convex
  shape (Martinetz 2018), optional Mie backend (`levisim[mie]`, which also installs numba).
  Refractive-index tables with interpolation (`RefractiveIndexTable`) for `SILICON`, `DIAMOND` and
  `GOLD`, each with its cited source (values at 1064 and 1550 nm unchanged); cited NdFeB recoil
  permeability and grades N35-N55 (`ndfeb_grade`), and electrical conductivities of bismuth and
  pyrolytic graphite (anisotropic); the eddy-current damping formula is verified against Xiong et
  al., Commun. Phys. 8, 80 (2025) and an independent numerical solution; `Particle.α_tensor`;
  sphere and tabulated-particle gas diffusion for any accommodation γs.
- Rotation: `RigidBodySimulation` (quaternions, torques from every element), librations in
  `MultiTrap` (`Dimension.is_libration`), Jones-vector tweezer polarisation, finite-size torque.
- `MultiTrap.from_nonlinear_equilibrium()` (linearisation at the true equilibrium with static
  cavity forces, recoil, gravity and pair forces), opt-in Brownian bath for ultrastrong direct
  coupling, and a warning when a dropped axis couples to the light.
- Paper reproductions (notebooks/papers): Penny 2023 (on `PaulTrap`), Krisnanda 2020, Conangla
  2020, Dania 2024, Rieser 2022, Reisenbauer 2024, Slezak 2018, Hoang 2016, Reimann 2018, Ahn 2020,
  van der Laan 2021, Pontin 2023, Kamba 2023, Vinante 2020; docs pages for every feature, and a
  cavity-output squeezing figure with the parameters of Deplano et al. 2025.

## [0.1.1] - 2026-11-25

### Added
- `MultiTrap(cavity_wavenumber=True)`: the cavity standing wave, and so the couplings
  gx, gy, gxy, gxz, gyz, uses the cavity wavenumber `Cavity.k` = (ωtw - Δ)/c instead of the
  tweezer's (a relative change of |Δ|/ωtw ~ 6 × 10⁻¹⁰). The axial factor kz = k - 1/zR keeps the
  tweezer wavenumber. New `Dimension.standing_wave_k`.
- `MultiTrap(static_cavity_shifts=True)`: the trap frequencies include the static stiffness of
  the cavity field, i.e. the cotrapping (diagonal) terms of the coherent-scattering potential and
  the dispersive potential of the cavity intensity (which also adds to gxy). The shifted
  frequencies are used throughout: mechanical susceptibilities and drift matrix, zero-point
  fluctuations (and so every coupling), thermal occupancies and photon-recoil heating rates.
  Measured frequencies (`override_Ω_dict`) are not shifted again. New
  `MultiTrap.static_cavity_hessians`, `MultiTrap.static_shifted_Ωj`,
  `Dimension.mechanical_frequency`, `Dimension.static_cavity_shift`.
- `levisim.physics.coherent_scattering`: the full (un-linearised) optical potential of a bead in
  its tweezer and the static cavity field (`optical_potential`), its static Hessian
  (`static_cavity_hessian`) and force (`static_cavity_force`), with tests against finite
  differences of the potential. Documentation page "Cavity wavenumber and static cavity shifts".

Both keywords are off by default: every result of 0.1.0 is unchanged unless they are passed.

### Changed
- README figures for the quickstart and the stochastic simulations redrawn and extended: an
  annotated homodyne spectrum with the displacement spectrum of each motion, a cavity-detuning sweep
  of the spectrum and of the phonon occupancies, the simulated traces, and a 200 ms simulation
  (was 40 ms) for smoother spectra, with the co-spectrum's statistical band and the window leakage
  shown as a ratio to theory.
- Installation instructions for the PyPI release (upgrade, version check, editable install of a
  clone with `pip install -e ".[dev]"`, running the notebooks against a release tag), badges in the
  README, the PyPI link in `CITATION.cff`, and the release notes for the `testpypi` environment.

## [0.1.0] - 2026-11-20

First release on PyPI. Compared with the code before October 2026:

### Added
- Packaging that ships every subpackage, declared dependencies with tested minimum versions
  (Python 3.10 with numpy 2.0, scipy 1.13, matplotlib 3.8, sympy 1.12, loguru 0.7, tqdm 4.60), optional
  extras (`elk`, `notify`, `video`, `notebooks`, `docs`, `dev`) and the `levisim-logs` command.
- A test suite (pytest), including a comparison against the Fortran reference implementation.
- A documentation site (Sphinx, MyST): user guide, calculation methods with a flowchart,
  conventions, logging, API reference, and a gallery of notebooks.
- Logging: WARNING-level console output by default, a rotating JSON-lines log file, an optional
  Elasticsearch sink, and `levisim-logs` to run Elasticsearch and Kibana without Docker.
- Covariance-matrix dynamics, multi-tone (Floquet) drives, Gaussian-state entanglement measures,
  and two-tone trap construction from the focal field.
- `MultiTrap` options: `tweezers=`, `beads=`, `trap_kwargs=` and `φn_dict=` in physical mode;
  `photon_recoil` (recoil heating in the quantum linear theory, on by default in physical mode);
  `cavity_drive_tweezer` / `cavity_drive_field_ratio` trap options, for a particle that scatters
  another tweezer's light into the cavity; more than 26 dimensions;
  couplings of either sign; stitched frequency ranges; `ΩModel` toy frequency models.
- Free-space `Simulation` options: `seed=`, `detection_floor=` on the recorded traces,
  photon recoil, gravity, Malitson SiO2 permittivity, stored momenta and radial traces, and a working
  second-order (Heun) integrator.
- A linear-stability report on `MultiTrap` (`drift_eigenvalues`, `max_growth_rate`, `is_stable`,
  `unstable_modes`, shown in `info`), a table of which presets are stable, a concepts page on the
  eigenvalue stability of coupled resonators, and a warning when a system is unstable or when
  `toy_params` contain keys nothing reads.
- Output-field quadrature covariance and squeezing (`levisim.multiphysics.quadrature_squeezing`) and
  the covariance of filtered cavity-output modes (`levisim.multiphysics.output_modes`).
- A README and docs showcase of three of the author's papers (PRR 5, 6 and 8), with rendered
  schematics and interactive (Plotly) figures on the documentation site.
- Curated notebooks: demos, a 100-particle showcase, and reproductions of Delić et al. (2020),
  Piotrowski et al. (2023), Vijayan et al. (2024), Pontin et al. (2025, a strongly coupled pair),
  Pontin et al. (2026, optical squeezing) and Deplano et al. (2026, stationary entanglement).

### Changed
- `MultiTrap` is now a package (`levisim.multiphysics.multitrap`): a base class holding the
  construction and the method dispatch, and three mixins for the linear-amplifier,
  hybridisation and analytic methods; the 11,294 arrays compared before and after the split
  (71 systems, all seven methods) are bit-identical.
- Physical-mode `MultiTrap`s include photon-recoil heating unless built with
  `photon_recoil=False`, which raises their occupations (e.g. one x motion at the defaults from
  44.0 to 47.3 phonons); the drift matrix, and so stability and mode frequencies, are unchanged.

### Fixed
- The analytic (closed-form) calculation methods now agree with the others for unequal tweezer
  phases and for particles off the cavity node (intra-particle couplings).
- Latent NameErrors, a doubled `info` property, single-precision complex susceptibilities in the
  Fortran reference, the free-space PSD fit normalisation, the detection floor feeding back into the
  dynamics, a stale-property hazard in `Cavity`, the cavity wavelength defaulting to 1064 nm for
  other tweezer wavelengths, and the homodyne quadrature of `homodyne_and_heterodyne_spectra`.
