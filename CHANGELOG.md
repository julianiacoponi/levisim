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
  `photon_recoil=True` (recoil heating in the quantum linear theory); more than 26 dimensions;
  couplings of either sign; stitched frequency ranges; `ΩModel` toy frequency models.
- Free-space `Simulation` options: `seed=`, `detection_floor=` on the recorded traces,
  photon recoil, gravity, Malitson SiO2 permittivity, stored momenta and radial traces, and a working
  second-order (Heun) integrator.
- Warnings when a system's drift matrix is unstable and when `toy_params` contain keys nothing reads.
- Curated notebooks: demos, a 100-particle showcase, and reproductions of Delić et al. (2020),
  Piotrowski et al. (2023) and Vijayan et al. (2024).

### Fixed
- The analytic (closed-form) calculation methods now agree with the others for unequal tweezer
  phases and for particles off the cavity node (intra-particle couplings).
- Latent NameErrors, a doubled `info` property, single-precision complex susceptibilities in the
  Fortran reference, the free-space PSD fit normalisation, the detection floor feeding back into the
  dynamics, a stale-property hazard in `Cavity`, the cavity wavelength defaulting to 1064 nm for
  other tweezer wavelengths, and the homodyne quadrature of `homodyne_and_heterodyne_spectra`.
