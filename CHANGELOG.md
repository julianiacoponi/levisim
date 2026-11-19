# Changelog

All notable changes to `levisim` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor versions).

## [0.1.0] - unreleased

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
