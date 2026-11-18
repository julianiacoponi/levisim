# Showcase

[`many_particles_one_cavity`](many_particles_one_cavity.ipynb): a comb of 100 levitated particles
coupled to one cavity mode, solved exactly in the linearised quantum theory and cross-checked by
independent routes (LAM, TNT and the closed form for the spectra; the Lyapunov equation against
the integrated spectra for the occupations; an exact sum rule; an analytic instability threshold).
It shows the comb turning collective - one superradiant mode, the rest subradiant - once the
particles' summed cooling rate exceeds the spread of their frequencies, and what that costs in
phonons. Runtime: several minutes (6-8 minutes on the heavily loaded machine that produced the stored outputs). Outputs are kept in git (`"keep_output": true`).

The notebook lifts levisim's current 26-dimension limit with a small patch in its first section;
remove that cell once `levisim.multiphysics.hybridisation.generate_subscripts_and_labels` supports
more dimensions.
