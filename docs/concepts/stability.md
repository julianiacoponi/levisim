# Eigenvalue stability of coupled resonators

Every calculation in levisim rests on one linear system of equations, and whether that system has a
steady state is decided by the eigenvalues of a single matrix. This page explains that eigenvalue
analysis, which is the same for any set of linearly coupled, damped and driven resonators: what the
eigenvalues mean, when they signal an instability, why a cooled system can nevertheless grow, why
the stationary spectra are blind to it, and which other physical systems the same machinery would
describe. The practical side - the stability report of a `MultiTrap` and the table of ready-made
systems - is the guide page [Linear stability of a system](../guide/stability.md).

## The linearised Langevin equations and the drift matrix

After linearising about the static cavity field, the quantum Langevin equations of the cavity mode
and of the $N$ mechanical dimensions (see [the system and its conventions](conventions.md)) are a
set of $2N + 2$ coupled first-order equations for the vector of operators
$\hat{\mathbf c} = (\hat a, \hat a^\dagger, \ldots, \hat b_{j_n}, \hat b_{j_n}^\dagger, \ldots)^\top$,

$$
\frac{d\hat{\mathbf c}(t)}{dt} = \mathbf A\, \hat{\mathbf c}(t) + \hat{\mathbf c}_\text{in}(t),
$$

where the drift matrix $\mathbf A$ (`MultiTrap.drift_matrix`) holds the free evolution
($+i\Delta - \kappa/2$ for $\hat a$, $-i\Omega_{j_n} - \Gamma_{j_n}/2$ for $\hat b_{j_n}$), the
light-mechanics couplings $\pm iG_{j_n}$, $\pm iG_{j_n}^*$ and the direct mechanical couplings
$g_{jk}$, and $\hat{\mathbf c}_\text{in}$ collects the input noises, each multiplied by the square
root of its damping rate. Nothing in this form is specific to levitated particles: any set of
resonators that interact through terms bilinear in their amplitudes, and that are each damped into
a Markovian bath, obeys an equation of exactly this shape, and levisim's seven calculation methods
are seven ways of solving it in the frequency domain (Clerk et al. 2010; Aspelmeyer, Kippenberg and
Marquardt 2014; Gardiner and Zoller 2004).

## Eigenvalues are normal modes

The homogeneous solutions are $\hat{\mathbf c}(t) \propto \mathbf v_k e^{\lambda_k t}$, with
$\mathbf A \mathbf v_k = \lambda_k \mathbf v_k$. With levisim's sign conventions each eigenvalue of a
mechanical normal mode can be written

$$
\lambda_k = -i\Omega_k - \gamma_k/2,
$$

so that $\lvert\mathrm{Im}\,\lambda_k\rvert = \Omega_k$ is the angular frequency of the normal mode,
and $-\mathrm{Re}\,\lambda_k = \gamma_k/2$ its amplitude damping rate, which is half its energy
damping rate $\gamma_k$ and equal to the half-width at half-maximum of its Lorentzian peak. The
eigenvector $\mathbf v_k$ says which bare motions, and how much of the cavity field, take part in
the mode. Because $\hat{\mathbf c}$ contains every operator together with its Hermitian conjugate,
the eigenvalues come in pairs $\lambda_k$, $\lambda_k^*$: the mode seen through its annihilation and
through its creation operators, at $\mp\Omega_k$ with the same damping. Equivalently, in the basis of
real quadratures $\hat X = \hat b + \hat b^\dagger$, $\hat P = i(\hat b^\dagger - \hat b)$ the drift
matrix becomes real, so its characteristic polynomial $\det(\lambda\mathbf I - \mathbf A)$ has real
coefficients and its roots are real or in complex-conjugate pairs. `MultiTrap.drift_eigenvalues`
lists all $2N + 2$ of them, and the frequencies of the peaks in every levisim spectrum are the
$\Omega_k$, shifted from the bare $\Omega_{j_n}$ by the optical spring and by the hybridisation of
the motions through the cavity.

## The stability criterion

A general solution is a sum of terms $e^{\lambda_k t}$, so it decays to the noise-driven steady state
if and only if every eigenvalue has a negative real part; a matrix with this property is called
Hurwitz (or stable). If even one eigenvalue has $\mathrm{Re}\,\lambda_k > 0$, the corresponding mode
grows as $e^{\mathrm{Re}\,\lambda_k t}$ from any fluctuation, the linearisation about the static field
breaks down after a few e-folding times $1/\mathrm{Re}\,\lambda_k$, and there is no steady state at
all. Two equivalent statements of the criterion are used in practice:

- **Lyapunov.** The steady-state covariance matrix $\mathbf V$ of the quadratures solves the Lyapunov
  equation $\mathbf A \mathbf V + \mathbf V \mathbf A^\top + \mathbf D = 0$, with $\mathbf D$ the
  (positive) diffusion matrix of the noises. It has a unique, positive-definite solution exactly when
  $\mathbf A$ is Hurwitz, which is why {func}`~levisim.multiphysics.covariance.steady_state_covariance`
  refuses an unstable system rather than returning the formal solution, which is not a covariance.
- **Routh-Hurwitz.** The signs of the real parts can be read from the real coefficients of the
  characteristic polynomial without finding its roots: all roots lie in the left half-plane if and only
  if a sequence of determinants built from the coefficients (the Hurwitz determinants) are all
  positive. This is the form in which the criterion is usually quoted for optomechanical and
  laser-type systems, often as explicit inequalities on the couplings (DeJesus and Kaufman 1987).

For the $(2N + 2)$-square matrices of levisim, a direct numerical diagonalisation is cheap and more
informative than the Routh-Hurwitz inequalities, since it also gives the frequency of the offending
mode, so `MultiTrap` computes the eigenvalues once at construction and counts as unstable any with
$\mathrm{Re}\,\lambda_k$ above a round-off tolerance of $10^{-8}$ of the largest $\lvert A_{ij}\rvert$.

**Static instability.** A dynamical instability is a growing mode of the full drift matrix. A
static one comes earlier: if the static cavity field softens a motion past its tweezer stiffness, that
motion has no restoring force at all. Blue detuning near an antinode can do this to $z$. `MultiTrap`
then raises a `ValueError` at construction (see [static cavity shifts](static_cavity_shifts.md)),
rather than building a drift matrix with an imaginary frequency.

## Why the stationary spectra cannot reveal an instability

Every levisim spectrum is built from the transfer matrix
$\mathbf T(\omega) = (-i\omega\mathbf I - \mathbf A)^{-1}$, which is finite at every real frequency as
long as no eigenvalue lies exactly on the imaginary axis. Near a single mode it reduces to the
susceptibility $1/[\gamma_k/2 - i(\omega - \Omega_k)]$, whose squared modulus

$$
\frac{1}{(\omega - \Omega_k)^2 + \gamma_k^2/4}
$$

is the same Lorentzian for $+\gamma_k$ and $-\gamma_k$. An anti-damped mode therefore gives a peak of
the same position and width as a damped mode of opposite damping, and the spectra of an unstable
system come out finite, smooth and plausible. They are nevertheless formal: for $\gamma_k > 0$,
$\mathbf T(\omega)$ is the Fourier transform of the causal response, which decays, whereas for
$\gamma_k < 0$ it is the transform of an anti-causal response that runs backwards in time, since the
causal one grows without bound and has no Fourier transform. The time-domain solution of an unstable
system is dominated by the growing mode and describes no stationary process, so the spectra describe
no measurement. This is why levisim checks the eigenvalues at construction instead of relying on the
spectra, and logs a WARNING for every unstable system it builds.

## Two coupled modes: avoided crossings, level attraction and exceptional points

The essential physics is already in two modes with amplitudes $\alpha_{1,2}$, bare frequencies
$\Omega_{1,2}$, equal damping $\gamma$ and couplings $g_{12}$ (of mode 1 to mode 2) and $g_{21}$,

$$
\frac{d}{dt}\begin{pmatrix}\alpha_1 \\ \alpha_2\end{pmatrix} =
\begin{pmatrix} -i\Omega_1 - \gamma/2 & -ig_{12} \\ -ig_{21} & -i\Omega_2 - \gamma/2 \end{pmatrix}
\begin{pmatrix}\alpha_1 \\ \alpha_2\end{pmatrix},
\qquad
\lambda_\pm = -i\bar\Omega - \frac{\gamma}{2} \pm \sqrt{-\frac{\delta^2}{4} - g_{12}\,g_{21}},
$$

with $\bar\Omega = (\Omega_1 + \Omega_2)/2$ and $\delta = \Omega_1 - \Omega_2$. Only the product
$g_{12}g_{21}$ matters, and its phase decides the character of the interaction:

- **Reciprocal (coherent) coupling**, $g_{21} = g_{12}^*$, so $g_{12}g_{21} = \lvert g\rvert^2 > 0$: the
  square root is imaginary for every $\delta$, the frequencies repel into an **avoided crossing** with a
  minimum splitting of $2\lvert g\rvert$, and both modes keep the damping $\gamma/2$. The coupling
  exchanges energy between the modes but neither creates nor removes any, so it can never destabilise a
  damped system.
- **Dissipative (non-reciprocal) coupling**, e.g. $g_{21} = -g_{12}^*$, so $g_{12}g_{21} = -\lvert g\rvert^2$:
  for $\lvert\delta\rvert < 2\lvert g\rvert$ the square root is real, the frequencies lock to $\bar\Omega$
  (**level attraction**) and the dampings split into $\gamma/2 \mp \sqrt{\lvert g\rvert^2 - \delta^2/4}$.
  One mode is then anti-damped as soon as $\sqrt{\lvert g\rvert^2 - \delta^2/4} > \gamma/2$, however
  well damped both modes are on their own.
- **Exceptional points.** Where $\delta^2/4 + g_{12}g_{21} = 0$ (here $\lvert\delta\rvert = 2\lvert g\rvert$),
  the two eigenvalues and also the two eigenvectors coalesce, the matrix is no longer diagonalisable,
  and the eigenvalues depend on the parameters as a square root. Such exceptional points are the
  generic boundary between avoided crossings and level attraction, and the central objects of
  non-Hermitian physics (Heiss 2012; Ashida, Gong and Ueda 2020).

```{figure} figures/stability_two_mode_model.png
:width: 95%
:alt: Eigenvalues of the two-mode model against detuning, for reciprocal and dissipative coupling

Two-mode model with $\gamma = g/2$. Top: frequencies relative to $\bar\Omega$; bottom: real parts
(the red line is the stability boundary). Left, reciprocal coupling: an avoided crossing at constant
damping. Right, dissipative coupling: the frequencies lock between the two exceptional points (EP,
$\lvert\delta\rvert = 2g$), and the damping splits until one mode grows.
```

The general case has a complex $g_{12}g_{21}$ and interpolates between the two: the frequencies
repel less and the dampings split, more the larger the dissipative part.

## Non-reciprocal coupling through the cavity

Without direct interactions, particles in one cavity do not couple to each other directly (only the
motions of the same particle couple through $g_{jk}$), but through the cavity field, which each particle drives and
which acts back on all of them. Eliminating the cavity field gives the hybridisation functions
$R_{j_n k_m}(\omega)$ of [the calculation methods](calculation_methods.md), whose cavity-mediated part
is a combination of $\mathrm{Re}(G_{j_n}G_{k_m}^*)$ and $\mathrm{Im}(G_{j_n}G_{k_m}^*)$, weighted by the
cavity susceptibilities $\chi_\text{c}(\omega)$ and $\chi_\text{c}^*(-\omega)$ (`Cavity.ηm`,
`Cavity.ηp`). For two motions that couple to the amplitude quadrature, $G_{j_n} = g_{j_n}e^{i\phi_n}$
and

$$
\mathrm{Re}(G_{j_n}G_{k_m}^*) = g_{j_n}g_{k_m}\cos(\phi_n - \phi_m), \qquad
\mathrm{Im}(G_{j_n}G_{k_m}^*) = g_{j_n}g_{k_m}\sin(\phi_n - \phi_m).
$$

The first term is symmetric under exchange of the two motions; the second changes sign, so the
response of motion $j_n$ to $k_m$ differs from that of $k_m$ to $j_n$. That antisymmetric part is a
non-reciprocal, partly dissipative coupling of exactly the kind that produces level attraction in the
two-mode model, and its strength is set by the difference of the tweezer phases. With equal phases the
particles couple to the same light quadrature, the coupling is reciprocal, and a red-detuned cavity
only cools; as $\lvert\phi_2 - \phi_1\rvert$ grows towards 90° the non-reciprocal part grows, and for two
motions that are closer in frequency than the cavity can resolve, one collective mode can take up
energy faster than the cavity removes it, even though every motion on its own is cooled. Tunable,
non-reciprocal interactions of this kind between levitated particles have been demonstrated by
choosing the phases of the trapping beams (Rieser et al. 2022), and cavity-mediated interactions
between two levitated particles by Vijayan et al. (2024).

```{figure} figures/stability_eigenvalue_trajectories.png
:width: 100%
:alt: Mechanical eigenvalues of the default two-particle system in the complex plane as the second tweezer phase is swept

The four mechanical normal modes of the default 2×2 system (shelf gas and cavity kwargs) in the
complex plane, as the second tweezer's phase $\phi_2$ runs from 0° to 90° (colour); 90° to 180° retrace
the same paths back. Right: the two hybridised $y$ modes near 135 kHz. The narrow (nearly dark) $y$
mode crosses into the unstable half-plane at $\phi_2 \approx 54.4°$ and reaches
$\mathrm{Re}\,\lambda = +2677$ s⁻¹ at 90°, at 134.92 kHz, while the broad (bright) $y$ mode is damped
more strongly.
```

```{figure} figures/stability_phase_sweep.png
:width: 80%
:alt: Normal-mode frequencies and real parts against the second tweezer phase

The same sweep against $\phi_2$. Top: the two $y$ modes are drawn together (level attraction) inside
the unstable window (grey), while the two $x$ modes barely move. Bottom: real parts; one $y$ mode
crosses $\mathrm{Re}\,\lambda = 0$ at 54.4° and 125.6°. Dashed: the largest real part with the
constructor-default cavity, closer to resonance, whose window is wider (35.6° to 144.4°).
```

In the default system the two $y$ motions are 1.8 kHz apart, less than their optical damping of about
1 kHz each can separate, so they behave as the near-degenerate pair of the two-mode model, whereas the
two $x$ motions are 4.2 kHz apart and stay stable at every phase. The growth rate is largest at
$\phi_2 = 90°$, where the coupling is purely non-reciprocal, and the window repeats every 180°, since
$\phi_2 \to \phi_2 + 180°$ only flips the sign of the second particle's couplings. A second, different
route to instability appears in levisim's off-node toy systems with three axes per particle: there a
strong coupling of the slow $z$ motion shifts its frequency through zero (the optical spring exceeds
the trap stiffness), the eigenvalue becomes real and positive, and the particle is pushed out of the
trap rather than driven into growing oscillation.

## Non-reciprocal direct coupling

Direct couplings between particles (`MultiTrap(interactions=[...])`, `toy_params['g_direct']`) enter
the drift matrix through `mechanical_coupling_matrix`, which need not be symmetric: entry $g_{jk}$ is
the rate at which motion $k$ drives motion $j$. A conservative interaction (Coulomb, gravity, Casimir)
has a symmetric stiffness, $K_{jk} = K_{kj}$, and so $g_{jk} = g_{kj}$: a reciprocal coupling; non-conservative forces (optical binding, or feedback that drives one particle with the
signal of another) need not. For two degenerate motions in the rotating-wave picture the coupled
amplitudes evolve with $\begin{pmatrix}-\Gamma/2 & ig_{12}\\ ig_{21} & -\Gamma/2\end{pmatrix}$, so the
eigenvalues are $-\Gamma/2 \pm \sqrt{-g_{12}g_{21}}$:

- $g_{12}g_{21} > 0$ (reciprocal, or of equal sign): a splitting $2\sqrt{g_{12}g_{21}}$, always stable;
- $g_{12}g_{21} = 0$ (one-way): no splitting and no growth, the driven motion just follows the other;
- $g_{12}g_{21} < 0$ (opposite signs): level attraction, and growth at $\sqrt{\lvert g_{12}g_{21}\rvert} - \Gamma/2$
  once the coupling exceeds the damping.

Checked on the drift matrix of two 300 kHz toy motions with $\Gamma/2\pi$ = 100 Hz and no cavity
coupling: $g_{12}/2\pi = -g_{21}/2\pi$ = 1 kHz gives a largest real part of +5969.0 s⁻¹ (the
rotating-wave value is 5969.0 s⁻¹) at a single frequency of 300.002 kHz, while $g_{21} = g_{12}$ gives
two modes 2 kHz apart with $\mathrm{Re}\,\lambda = -\Gamma/2$. `MultiTrap.is_stable`,
`max_growth_rate` and the construction-time warning treat such systems like any other. The
non-reciprocity of the cavity-mediated coupling above adds to that of the direct coupling.

## Other systems the same machinery describes

The drift-matrix analysis above, and levisim's frequency-domain solution of
$\dot{\hat{\mathbf c}} = \mathbf A\hat{\mathbf c} + \hat{\mathbf c}_\text{in}$, apply unchanged to any
linear network of damped bosonic modes. What is specific to levisim is how $\mathbf A$ and the noise
correlations are filled in, which is concentrated in `MultiTrapBase.setup_*` (the couplings and
frequencies), `LinearAmplifierMixin.setup_linear_amplifier_model_matrices` (the drift and noise
matrices) and `Dimension` (the one-dimensional susceptibilities). The following systems could be
modelled by changing those parts, roughly in increasing order of effort.

| System | Same structure | What levisim would need |
|---|---|---|
| Multimode optomechanics: membrane-in-the-middle (Thompson et al. 2008) and arrays of optomechanical-crystal cavities (Eichenfield et al. 2009) | several mechanical modes coupled to one or more optical modes by $G\,\hat q\,(\hat a + \hat a^\dagger)$ | a constructor that takes mode frequencies, masses and couplings directly (the toy-parameter route nearly does this already) and, for crystal arrays, several cavity modes with their own $\Delta$, $\kappa$ and an optical hopping matrix: $\hat{\mathbf c}$ would hold $2M$ light operators instead of 2 |
| Clamped resonator networks (MEMS/NEMS): coupled beams, strings and membranes | mechanical modes coupled elastically or electrostatically, with or without a readout cavity | direct mechanical couplings between different resonators (levisim only has $g_{jk}$ within one particle), possibly a classical readout instead of a cavity, and thermal baths with frequency-dependent (structural) damping |
| Circuit optomechanics and electromechanics: drums and membranes coupled to superconducting microwave resonators (Teufel et al. 2011) | identical Hamiltonian, with microwave photons instead of optical ones | thermal occupations for the microwave mode ($n_\text{c} \neq 0$, already a factor in the noise vector), several microwave modes or drive tones (the multi-tone code is a starting point), and parameters in the microwave regime, often with resolved sidebands $\kappa < \Omega$ |
| Coupled superconducting circuit resonators and parametric amplifiers | linear modes with beam-splitter and two-mode-squeezing couplings | couplings of the form $\hat a_1^\dagger\hat a_2$ and $\hat a_1\hat a_2$ between arbitrary modes, rather than through positions $\hat q$; the TNT and LAM methods carry over directly, the hybridisation and analytic methods would need new derivations |
| Trapped ions and atomic ensembles in cavities (Ritsch et al. 2013) | collective motional or spin modes (in the Holstein-Primakoff limit) coupled to a cavity mode | the scattering couplings of ions or atoms instead of dielectric particles, and, for spins, the linearised spin operators with their own (possibly inverted, $n < 0$) baths |
| Non-Hermitian and PT-symmetric photonic or mechanical lattices (Rüter et al. 2010; El-Ganainy et al. 2018) | coupled resonators with designed gain and loss and non-reciprocal hopping, whose spectra show exceptional points (Xu et al. 2016) | per-mode gain (negative damping, with the noise of an inverted bath), arbitrary complex couplings in $\mathbf A$, and lattice-structured couplings; the stability report applies as is, and the eigenvalues are often the quantity of interest themselves |
| Levitated rotors: librational and rotational modes (Stickler et al. 2021) | librations are harmonic modes coupled to the cavity through the particle's orientation | `Trap` and `Dimension` for angular degrees of freedom (moments of inertia, torques from an anisotropic polarisability, recoil heating of the librations) as extra dimensions alongside $x$, $y$, $z$ |
| Cavity magnomechanics (Zhang et al. 2016) | a magnon mode, a microwave mode and a mechanical mode, coupled linearly after linearisation | a third kind of mode in $\hat{\mathbf c}$ with its own damping and coupling terms; the drift-matrix and noise-matrix construction is the only part that changes |

In every case the eigenvalues of $\mathbf A$ remain the first thing to compute, for the same reason as
here: a system with gain, parametric driving, or non-reciprocal couplings can be unstable while its
stationary spectra look entirely reasonable.

## References

- M. Aspelmeyer, T. J. Kippenberg and F. Marquardt, *Cavity optomechanics*, Rev. Mod. Phys. **86**, 1391 (2014).
- A. A. Clerk, M. H. Devoret, S. M. Girvin, F. Marquardt and R. J. Schoelkopf, *Introduction to quantum noise, measurement, and amplification*, Rev. Mod. Phys. **82**, 1155 (2010).
- C. W. Gardiner and P. Zoller, *Quantum Noise*, 3rd edition (Springer, 2004).
- E. X. DeJesus and C. Kaufman, *Routh-Hurwitz criterion in the examination of eigenvalues of a system of nonlinear ordinary differential equations*, Phys. Rev. A **35**, 5288 (1987).
- Y. Ashida, Z. Gong and M. Ueda, *Non-Hermitian physics*, Adv. Phys. **69**, 249 (2020).
- W. D. Heiss, *The physics of exceptional points*, J. Phys. A **45**, 444016 (2012).
- R. El-Ganainy, K. G. Makris, M. Khajavikhan, Z. H. Musslimani, S. Rotter and D. N. Christodoulides, *Non-Hermitian physics and PT symmetry*, Nat. Phys. **14**, 11 (2018).
- C. E. Rüter, K. G. Makris, R. El-Ganainy, D. N. Christodoulides, M. Segev and D. Kip, *Observation of parity-time symmetry in optics*, Nat. Phys. **6**, 192 (2010).
- H. Xu, D. Mason, L. Jiang and J. G. E. Harris, *Topological energy transfer in an optomechanical system with exceptional points*, Nature **537**, 80 (2016).
- J. Rieser, M. A. Ciampini, H. Rudolph, N. Kiesel, K. Hornberger, B. A. Stickler, M. Aspelmeyer and U. Delić, *Tunable light-induced dipole-dipole interaction between optically levitated nanoparticles*, Science **377**, 987 (2022).
- J. Vijayan, J. Piotrowski, C. Gonzalez-Ballestero, K. Weber, O. Romero-Isart and L. Novotny, *Cavity-mediated long-range interactions in levitated optomechanics*, Nat. Phys. **20**, 859 (2024).
- J. D. Thompson, B. M. Zwickl, A. M. Jayich, F. Marquardt, S. M. Girvin and J. G. E. Harris, *Strong dispersive coupling of a high-finesse cavity to a micromechanical membrane*, Nature **452**, 72 (2008).
- M. Eichenfield, J. Chan, R. M. Camacho, K. J. Vahala and O. Painter, *Optomechanical crystals*, Nature **462**, 78 (2009).
- J. D. Teufel, T. Donner, D. Li, J. W. Harlow, M. S. Allman, K. Cicak, A. J. Sirois, J. D. Whittaker, K. W. Lehnert and R. W. Simmonds, *Sideband cooling of micromechanical motion to the quantum ground state*, Nature **475**, 359 (2011).
- H. Ritsch, P. Domokos, F. Brennecke and T. Esslinger, *Cold atoms in cavity-generated dynamical optical potentials*, Rev. Mod. Phys. **85**, 553 (2013).
- B. A. Stickler, K. Hornberger and M. S. Kim, *Quantum rotations of nanoparticles*, Nat. Rev. Phys. **3**, 589 (2021).
- X. Zhang, C.-L. Zou, L. Jiang and H. X. Tang, *Cavity magnomechanics*, Sci. Adv. **2**, e1501286 (2016).
