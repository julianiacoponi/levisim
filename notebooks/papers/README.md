# Paper reproductions

Each notebook here builds a published levitated-optomechanics system in levisim from the paper's
stated physical parameters (power, wavelength, waists, particle size, cavity geometry, detuning,
position, polarisation) and compares levisim's results with the paper's numbers and figures.
Every notebook states where each parameter comes from (section, table or figure) and which
modelling assumptions the paper does not fix, and ends with a summary of agreements and
discrepancies. The notebooks keep their outputs (`"keep_output": true`), so they can be read without
running them.

| Notebook | Paper | What levisim reproduces | Agreement | Known discrepancies |
|---|---|---|---|---|
| [delic2020_ground_state_cooling](delic2020_ground_state_cooling.ipynb) (~1 s) | U. Delić, M. Reisenbauer, K. Dare, D. Grass, V. Vuletić, N. Kiesel and M. Aspelmeyer, *Cooling of a levitated nanoparticle to the motional quantum ground state*, Science **367**, 892 (2020), [doi:10.1126/science.aba3993](https://doi.org/10.1126/science.aba3993) | cavity drive, $g_x$, gas and recoil heating; $n_x$ against detuning (Fig. 2c); heterodyne sidebands and sideband thermometry (Eqs. S2, S3) | $n_x$ = 0.397 at 315 kHz against measured 0.43 ± 0.03, inside the paper's theory band (0.31-0.54); $\omega_d/2\pi$ = 4.33 × 10⁹ Hz (paper ≈ 4 × 10⁹); recoil 5.6 kHz (paper 6 kHz); recoil-limited minimum 0.145 (paper 0.154) | $g_x$ 12 % above the inferred 71 kHz, gas heating 16 % above measured; measured $n_x$ above every model (and above the paper's band) at large detuning; at the optimum ($g_x/\kappa$ = 0.41) the paper's thermometry estimators deviate from the true $n_x$ by +9 % (Lorentzian fit) and −34 % (band power) |
| [piotrowski2023_two_mode_ground_state](piotrowski2023_two_mode_ground_state.ipynb) (~2 s) | J. Piotrowski, D. Windey, J. Vijayan, C. Gonzalez-Ballestero, A. de los Ríos Sommer, N. Meyer, R. Quidant, O. Romero-Isart, R. Reimann and L. Novotny, *Simultaneous ground-state cooling of two mechanical modes of a levitated nanoparticle*, Nat. Phys. **19**, 1009 (2023), [doi:10.1038/s41567-023-01956-1](https://doi.org/10.1038/s41567-023-01956-1) | trap frequencies, recoil and gas heating; $n_x$, $n_y$ against detuning (Fig. 2b) and polarisation angle (Fig. 3e); the sideband-thermometry error map (Fig. 4d,e) | paraxial frequencies and recoil rates within 1 % of the paper; with the fitted $g$, $\Gamma$ the lower edge of the uncertainty band matches the paper's $n_x$ band within 0.01 at all five detunings (identical to two decimals at four); thermometry error 1.2 % / 1.0 % at the Fig. 2 parameters (paper: "about 1 %"); dipole law ($\propto\sin\theta$) matches $n_x(\theta)$, a θ-independent $G_0$ does not | the coupling from the paper's own Eq. (1) and stated parameters is 2.1-2.5× the fitted one (the paper uses the fitted values); measured $n_y$ above both levisim and the paper's model at 362-662 kHz; Methods typos ($G_0 \propto \cos\theta$ convention slip, gas heating a factor 2 below its Eq. (3)) |
| [vijayan2024_cavity_mediated_interactions](vijayan2024_cavity_mediated_interactions.ipynb) (~1 s) | J. Vijayan, J. Piotrowski, C. Gonzalez-Ballestero, K. Weber, O. Romero-Isart and L. Novotny, *Cavity-mediated long-range interactions in levitated optomechanics*, Nat. Phys. **20**, 859 (2024), [doi:10.1038/s41567-024-02405-3](https://doi.org/10.1038/s41567-024-02405-3) | two particles in one cavity (levisim's multi-trap case): coupling estimates, spectrogram of the avoided crossing (Fig. 2a), splitting against detuning (Fig. 2d) and particle position (Fig. 3b), full linear model against the paper's adiabatic Eq. (1) | with waists inferred from the measured frequencies, the full model matches 11 of 14 measured splittings within 8 % (6.12 against 6.61 kHz at 0.45 MHz) and the splitting at $d = 4\lambda$; the paper's $g_y$ and $g_z$ estimates are reproduced if the zero-point motion is taken at paraxial frequencies | the stated 0.85 μm waist is inconsistent with the measured frequencies and gives ~50 % larger splittings; the bare cavity-axis frequency (70 kHz) is inferred, not stated; Eq. (1) underestimates the full model by up to 5.5 %; Fig. 4 not reproduced (detuning and $\Omega_z$ not stated) |

Runtimes are the computation time printed at the end of each notebook on a laptop; importing levisim
adds a second or two.

## Shared pieces

- The systems are built with levisim directly: `MultiTrap(tweezers=[...], beads=[...], trap_kwargs=[...])`
  takes each paper's `Tweezer` and `Bead` (and measured frequencies through `override_Ω_dict`),
  drives the cavity at the tweezer's wavelength, and `MultiTrap(photon_recoil=True)` adds the
  photon-recoil heating (`Trap.photon_recoil_heating_rates`) to the gas bath of every mode.
- [`paper_tools.py`](paper_tools.py): thin helpers over levisim's public API (see its docstring):
  - `toy_multitrap` builds a toy-parameter `MultiTrap` with a paper's fitted couplings and *total*
    heating rates, which `add_heating` imposes by raising each mode's bath occupation
    (`photon_recoil=True` needs physical parameters, so toy systems still need this);
  - `heterodyne` is a vectorised version of `MultiTrap.heterodyne_spectrum`, checked against it in
    every notebook that uses it;
  - `occupations`, `mechanical_eigenfrequencies` and `load_digitised` read out a built system and
    load the digitised figure data.

  The notebooks put their own folder on `sys.path` to import it.
- [`data/`](data): the measured points and the papers' theory curves, extracted from the vector
  graphics of the published figures with PyMuPDF. Each JSON file records its axis calibration and
  the values from the text it was checked against.

## Conventions used throughout

levisim's detuning is $\Delta = \omega_\text{tweezer} - \omega_\text{cavity}$ (red is $\Delta < 0$), and
all three papers use the opposite sign; plots use the papers' sign. levisim's tweezer is polarised
along its $y$ axis and `Tweezer.θ` is the polarisation angle to the cavity axis, so each notebook
states how the paper's axes map onto levisim's. Recoil heating uses the standard dipole-scattering
rates with coefficients (1, 2, 7) along, perpendicular to and along the beam relative to the
polarisation (Piotrowski et al. 2023, Eq. (4); Gonzalez-Ballestero et al. 2019, Table II).

## Papers considered and not reproduced

- **D. Windey et al., PRL 122, 123601 (2019)** ([doi:10.1103/PhysRevLett.122.123601](https://doi.org/10.1103/PhysRevLett.122.123601)),
  first 3D cavity cooling by coherent scattering. Not reproduced: the tweezer waist was not measured,
  and the companion theory paper (below) treats the particle radius as a free parameter (R = 50 nm
  against a nominal 136 nm diameter). The cavity is not sideband-resolved ($\kappa/2\pi$ = 1.06 MHz),
  and the same physics is tested more sharply by the three notebooks above.
- **U. Delić et al., PRL 122, 123602 (2019)** ([doi:10.1103/PhysRevLett.122.123602](https://doi.org/10.1103/PhysRevLett.122.123602)),
  first cavity cooling by coherent scattering. Same apparatus as Delić et al. (2020), whose ground-state
  result is the stronger test; used here for the cavity waist (41.1 μm).
- **C. Gonzalez-Ballestero et al., PRA 100, 013805 (2019)** ([doi:10.1103/PhysRevA.100.013805](https://doi.org/10.1103/PhysRevA.100.013805)),
  master-equation theory of coherent-scattering cooling. Its recoil-heating coefficients and
  polarisation dependence are cross-checked through Piotrowski et al. (2023), which uses them. A full
  reproduction needs the static coherent-scattering frequency shifts and equilibrium displacements
  (its Fig. 3 and Eq. (92)), which levisim does not model yet. This is a natural follow-up if those
  become an opt-in.
- **M. Toroš, U. Delić, F. Hales and T. S. Monteiro, PRR 3, 023071 (2021)** ([doi:10.1103/PhysRevResearch.3.023071](https://doi.org/10.1103/PhysRevResearch.3.023071)),
  2D coherent-scattering cooling theory. It shares its lineage with levisim's model, so it would not
  be an independent check.
- **A. Ranfagni et al., Nat. Phys. 17, 1120 (2021)** ([doi:10.1038/s41567-021-01307-y](https://doi.org/10.1038/s41567-021-01307-y)),
  vectorial polaritons, and **Ranfagni et al., PRR 4, 033051 (2022)** ([doi:10.1103/PhysRevResearch.4.033051](https://doi.org/10.1103/PhysRevResearch.4.033051)),
  2D quantum motion. Their parameters are sufficient ($\kappa/2\pi$ = 57 kHz, $\theta$ = 72°,
  $g/2\pi \approx$ 30 kHz), and the strong-coupling 2D spectra would test levisim's hybridisation away
  from weak coupling. Not done within this round; the best candidate for the next notebook.
- **Free-space measurement-based feedback cooling** (Tebbenjohanns et al. 2021 and Magrini et al.
  2021, as named in the task; bibliographic details not verified here). Not reproduced: their results
  rest on quantum-limited measurement backaction and optimal (Kalman/LQG) control, whereas levisim's
  stochastic simulator is classical (Langevin dynamics with cold damping). It could reproduce only
  the classical part of the feedback physics, which would not test those papers' claims.
- The owner's own papers (PRR 5, 013013 (2023); PRR 6, 013129 (2024); PRR 8, 033286 (2026)) and
  Deplano et al., Nat. Commun. 16, 4215 (2025) (used in `docs/two_tone_entanglement`) are excluded
  by design.
