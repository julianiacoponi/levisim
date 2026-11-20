# User guides

```{toctree}
:maxdepth: 1

cavity_spectra
stability
stochastic_simulations
paul_traps
cavity_simulations
logging
../two_tone_entanglement/README
../two_tone_entanglement/polarisation_and_two_particles
```

- [Spectra of several particles in one cavity](cavity_spectra.md): building a `MultiTrap`, and
  calculating and cross-checking its optical and mechanical spectra.
- [Linear stability of a system](stability.md): the stability report of a `MultiTrap`, which
  ready-made systems are unstable, and the window of tweezer phases that destabilises the default
  two-particle system.
- [Stochastic simulations of a bead in a free-space tweezer](stochastic_simulations.md): the
  Langevin simulator, its spectral estimates and their comparison with theory.
- [Paul and hybrid traps](paul_traps.md): the `PaulTrap` element (full rf force and secular
  stiffness), Mathieu parameters and the exact stability region, micromotion, sag and stray fields,
  electrode noise and heating, and hybrid Paul-optical traps in `Trap` and `MultiTrap`.
- [Time traces of particles in a cavity](cavity_simulations.md): `CavitySimulation` and
  `MultiTrap.simulate`, the exact linear model, its homodyne, heterodyne, photon-flux and position
  records, and their validation against the `MultiTrap` spectra.
- [Logging](logging.md): where levisim's log records go, what each level means, reading the
  JSON-lines log file, and browsing it in Kibana with `levisim-logs` (no Docker) or Docker.
- [Steady-state entanglement with a two-tone drive](../two_tone_entanglement/README.md): whether a
  second, blue-detuned tweezer tone can entangle one particle's $x$ and $y$ motions, at the
  parameters of Deplano et al. (2025).
- [Two-tone traps: polarisation, mech-mech coupling, the node, and a second particle](../two_tone_entanglement/polarisation_and_two_particles.md):
  the follow-up study, with traps built from the focal field of each tone.
