# Showcase

[`many_particles_one_cavity`](many_particles_one_cavity.ipynb): a comb of 100 levitated particles
coupled to one cavity mode, solved exactly in the linearised quantum theory and cross-checked by
independent routes (LAM, TNT and the closed form for the spectra; the Lyapunov equation against
the integrated spectra for the occupations; an exact sum rule; an analytic instability threshold).
It shows the comb turning collective - one superradiant mode, the rest subradiant - once the
particles' summed cooling rate exceeds the spread of their frequencies, and what that costs in
phonons. Runtime: about a minute and a half (83 s on the 8-core laptop that produced the stored
outputs). Outputs are kept in git (`"keep_output": true`).
