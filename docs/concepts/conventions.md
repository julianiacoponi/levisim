# The system and its conventions

This page fixes the notation used across `levisim` and these docs. The equations are those of the
{class}`~levisim.multiphysics.enums.quantum_linear_theory_method.QuantumLinearTheoryMethod`
docstring, which is the reference in the code.

## The system

$N_\text{p}$ particles are each held in an optical tweezer inside one optical cavity. Each
particle $n$ moves along up to three Cartesian axes $j \in \{x, y, z\}$, and each such
"dimension" $j_n$ is a harmonic oscillator with phonon operators $\hat b_{j_n}$, $\hat b_{j_n}^\dagger$,
frequency $\Omega_{j_n}$, mass $M_n$ and gas damping rate $\Gamma_{j_n}$. The tweezer light
coherently scatters into one cavity mode $\hat a$ of linewidth $\kappa$, which couples every
dimension to the same light field. There are $N$ dimensions in total, and $2N + 2$ operators.

In the frame rotating at the tweezer frequency,

$$
\hat H/\hbar = -\Delta\,\hat a^\dagger \hat a + \sum_{j_n} \Omega_{j_n} \hat b_{j_n}^\dagger \hat b_{j_n}
- \sum_{j_n} \hat q_{j_n} \hat L_{j_n} - \sum_n \sum_{j < k} g_{jk}\, \hat q_{j_n} \hat q_{k_n},
$$

with $\hat q_{j_n} \equiv \hat b_{j_n} + \hat b_{j_n}^\dagger$ and
$\hat L_{j_n} \equiv G_{j_n} \hat a^\dagger + G_{j_n}^* \hat a$. The light-mechanics couplings
$G_{j_n} = g_{j_n} e^{i\phi_n}$ carry the tweezer phase $\phi_n$ of particle $n$; the last term is
the intra-particle mechanical coupling, written in the code as $\sum\sum g_{jk}$ and appearing in
the equations of motion as $+i\sum_k g_{jk}(\hat b_{k_n}^\dagger + \hat b_{k_n})$.

**Detuning.** $\Delta \equiv \omega_\text{tw} - \omega_\text{c}$, the tweezer frequency minus
the cavity resonance, so a red-detuned (cooling) drive has $\Delta < 0$ and the cavity term in
$\hat H$ is $-\hbar\Delta\,\hat a^\dagger \hat a$. The default cavity in
{mod}`levisim.multiphysics.utils.shelf` has $\Delta = -2\pi \times 450$ kHz.

**Quantum Langevin equations.** With input noises $\hat a_\text{in}$ and $\hat b_{j_n,\text{in}}$,

$$
\begin{aligned}
\dot{\hat a} &= (+i\Delta - \kappa/2)\,\hat a + i \sum_{j_n} G_{j_n} \hat q_{j_n} + \sqrt{\kappa}\, \hat a_\text{in}, \\
\dot{\hat b}_{j_n} &= (-i\Omega_{j_n} - \Gamma_{j_n}/2)\,\hat b_{j_n} + i \hat L_{j_n} + i \sum_k g_{jk} \hat q_{k_n} + \sqrt{\Gamma_{j_n}}\, \hat b_{j_n,\text{in}},
\end{aligned}
$$

and their Hermitian conjugates. Collected into $\hat{\mathbf c} = (\hat a, \hat a^\dagger, \ldots, \hat b_{j_n}, \hat b_{j_n}^\dagger, \ldots)^\top$
they read $\dot{\hat{\mathbf c}} = \mathbf A \hat{\mathbf c} + \hat{\mathbf c}_\text{in}$, with
$\mathbf A$ the drift matrix.

## Fourier convention and susceptibilities

$f(t) = \int \frac{d\omega}{2\pi}\, e^{-i\omega t} f(\omega)$, so $d/dt \to -i\omega$ and

$$
\hat{\mathbf c}(\omega) = \mathbf T(\omega)\, \hat{\mathbf c}_\text{in}(\omega),
\qquad \mathbf T(\omega) \equiv (-i\omega \mathbf I - \mathbf A)^{-1}.
$$

The optical and mechanical susceptibilities are

$$
\chi_\text{c}(\omega) \equiv \frac{1}{\kappa/2 - i(\omega + \Delta)}, \qquad
\chi_{j_n}(\omega) \equiv \frac{1}{\Gamma_{j_n}/2 - i(\omega - \Omega_{j_n})}.
$$

Creation operators pick up the conjugate susceptibility at $-\omega$, because the Fourier
transform of $\hat a^\dagger(t)$ is $[\hat a(-\omega)]^\dagger$. These docs write it out in full as
$\chi^*(-\omega)$, never as a barred shorthand: for example

$$
\hat a(\omega) = \chi_\text{c}(\omega)\big[\,i\textstyle\sum G_{j_n} \hat q_{j_n} + \sqrt{\kappa}\,\hat a_\text{in}\big],
\qquad
\hat a^\dagger(\omega) = \chi_\text{c}^*(-\omega)\big[-i\textstyle\sum G_{j_n}^* \hat q_{j_n} + \sqrt{\kappa}\,\hat a_\text{in}^\dagger\big],
$$

with $\chi_\text{c}^*(-\omega) = [\kappa/2 - i(\omega - \Delta)]^{-1}$. A position responds through
$\mu_{j_n}(\omega) \equiv \chi_{j_n}(\omega) - \chi_{j_n}^*(-\omega)$ (`dim.μm` in the code).

## Units

- **Frequencies are angular**, in rad/s, everywhere in the code. The unit constants in
  {mod}`levisim.core.units` build this in: `Hz` is $2\pi$, so `kHz` is $2\pi \times 10^3$ rad/s.
  Multiplying by `kHz` makes an angular frequency from a number in kHz, and dividing by it turns
  one back. Docstrings mark such quantities `[2π·Hz]`.
- **Positions are dimensionless**, in units of the zero-point fluctuation:
  $\hat q_{j_n} = \hat b_{j_n} + \hat b_{j_n}^\dagger$, so the physical position is
  $\hat x_{j_n} = x_{\text{zpf},j_n} \hat q_{j_n}$ with
  $x_{\text{zpf},j_n} = \sqrt{\hbar / (2 M_n \Omega_{j_n})}$, and $\langle \hat q^2 \rangle = 2n + 1$ for
  a thermal state of $n$ phonons.
- **Light** is in photon-number units: the input noise $\hat a_\text{in}$ has
  $\langle \hat a_\text{in}(\omega) \hat a^\dagger_\text{in}(\omega') \rangle \propto (n_\text{c} + 1)$ with
  $n_\text{c} = 0$ for shot noise, so the homodyne spectrum is in shot-noise units.

## Spectral densities

Spectra are **two-sided** functions of $\omega \in (-\infty, \infty)$ (the default `ω_range` runs
from $-2\pi \times 400$ to $+2\pi \times 400$ kHz) and **not symmetrised**. For an operator
$\hat A$,

$$
S_{AA}(\omega) = \int dt\, e^{i\omega t} \langle \hat A(t) \hat A(0) \rangle,
\qquad
\int \frac{d\omega}{2\pi}\, S_{AA}(\omega) = \langle \hat A^2 \rangle .
$$

In the code a spectrum is a sum over input noises of
$\eta_\text{noise}\,|\hat A[\text{noise}]|^2$, where $\eta$ are the occupation factors
$(n_\text{c} + 1, n_\text{c}, \ldots, n_{j_n} + 1, n_{j_n}, \ldots)$
(`MultiTrap.quantum_factor_per_noise_operator`).

Two consequences, both checked by `docs/guide/scripts/conventions_psd_normalisation.py` for
one $x$ motion ($\Omega/2\pi = 156.3$ kHz) at the default parameters, whose bath includes photon recoil:

- **Normalisation.** $\int S_{qq}\, d\omega/2\pi = 95.509$ over $|\omega|/2\pi < 3$ MHz, against
  $\langle \hat q^2 \rangle = 95.520$ from the Lyapunov steady state of
  {mod}`levisim.multiphysics.covariance` (relative difference $-1.1 \times 10^{-4}$, from the
  truncated tails), i.e. $n = 47.3$ phonons.
- **Sideband asymmetry.** $S_{qq}(+\Omega)/S_{qq}(-\Omega) = 1.0215$, against
  $(n + 1)/n = 1.0212$: the positive-frequency peak carries $n + 1$ and the negative one $n$.

To compare with a measured **one-sided** displacement PSD in m²/Hz, fold and rescale:
$S^{(1)}_{xx}(f) = x_\text{zpf}^2\,[S_{qq}(2\pi f) + S_{qq}(-2\pi f)]$ for $f > 0$, which keeps
$\int_0^\infty S^{(1)}_{xx}\, df = \langle \hat x^2 \rangle$. Calling the position-spectrum
functions with `zpf_scaled=False` applies the $x_\text{zpf}^2$ factor but not the fold.

**Detection.** The homodyne signal is
$\hat h = \hat a_\text{out}^\dagger e^{i\theta} + \hat a_\text{out} e^{-i\theta}$ with the angle
$\theta$ = `cavity.θhom` ($\pi/4$ by default), and the heterodyne spectrum is
$S_\text{het}(\omega) = S_{aa^\dagger}(\omega + \Omega_\text{het}) + S_{a^\dagger a}(\omega - \Omega_\text{het})$.
