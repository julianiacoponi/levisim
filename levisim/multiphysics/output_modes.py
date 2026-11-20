r"""
Covariance of propagating (filtered) cavity-output modes, jointly with the intracavity modes

`levisim.multiphysics.covariance` gives the steady state of the modes INSIDE the cavity. Light
that leaves the cavity is a continuum of frequencies, so a propagating optical mode has to be
defined by an envelope ξ (a temporal mode or, equivalently, a spectral filter)::

    a_ξ = ∫ ξ(ω) a_out(ω) dω/2π,        ∫ |ξ(ω)|² dω/2π = 1   ⇔   [a_ξ, a_ξ†] = 1

with levisim's Fourier convention O(ω) = ∫ O(t) e^{iωt} dt and a_out(ω) the detected output field
(cavity input-output a_out = √κ a − a_in, then a beam splitter of efficiency η that mixes in
vacuum). The quadratures Q_ξ = a_ξ + a_ξ† and P_ξ = i(a_ξ† − a_ξ) have the same normalisation as
every other mode in levisim (vacuum covariance = identity), so the covariance matrix of
(Q_ξ, P_ξ, X_b, P_b, ...) can be handed straight to `levisim.analysis.gaussian_states`.

Everything here works on a `QuadratureSystem`, the linear Langevin equations in real quadratures::

    du/dt = A u + B w(t),    ½〈{w(t), w(t')ᵀ}〉 = N δ(t − t')

Only the symmetrised noise correlation N matters for the symmetrised covariance of a linear system.
`QuadratureSystem.from_multitrap` takes it from a `MultiTrap`, and `stack_cavity_modes` joins
several single-cavity-mode MultiTraps that share the same mechanics into ONE system with several
cavity modes (e.g. two longitudinal modes driven by two tweezer lasers at different detunings,
as in Deplano et al., Science 394, 113 (2026)), which a single `MultiTrap` cannot describe.

Two independent routes give the filtered-mode covariance:

- `output_mode_covariance` integrates the cross-spectra of the output against the filter in the
  frequency domain (any filter shape, e.g. the flat window of `rectangular_filter`); the
  intracavity block is the exact Lyapunov solution.
- `cascade_lorentzian_filter` appends a virtual one-sided 'filter cavity' fed by the output field
  to the drift matrix; its intracavity mode IS the temporal mode with the Lorentzian envelope of
  `lorentzian_filter`, so a single Lyapunov solve gives the same covariance. The tests check the
  two routes against each other.
"""
from dataclasses import dataclass, field

import numpy as np

from scipy.linalg import solve_continuous_lyapunov

from levisim.multiphysics.covariance import quadrature_drift_matrix


__all__ = [
    'QuadratureSystem',
    'stack_cavity_modes',
    'add_mechanical_mode',
    'SpectralFilter',
    'rectangular_filter',
    'lorentzian_filter',
    'filter_norm',
    'output_mode_covariance',
    'rectangular_filter_scan',
    'cascade_lorentzian_filter',
]


# ----------------------------------------------------------------------------------------------
# LINEAR SYSTEM IN QUADRATURES
# ----------------------------------------------------------------------------------------------
@dataclass
class QuadratureSystem:
    """
    du/dt = A u + B w with ½〈{w(t), w(t')ᵀ}〉 = N δ(t − t'), in the quadrature basis
    u = (X_1, P_1, X_2, P_2, ...) with vacuum covariance = identity

    - A: (n, n) real drift matrix [2π·Hz]
    - B: (n, m) real noise-input matrix [√(2π·Hz)]
    - N: (m, m) real symmetric noise correlation (identity for vacuum inputs) [·]
    - labels: one label per mode (quadrature pair) of u
    - cavities: {label: (κ, (column of Q_in, column of P_in))} for every cavity mode whose output
      field can be detected; the input-noise columns are those of w that enter that cavity
    """
    A: np.ndarray
    B: np.ndarray
    N: np.ndarray
    labels: list
    cavities: dict = field(default_factory=dict)

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.N = np.asarray(self.N, dtype=float)
        n, m = self.B.shape
        assert self.A.shape == (n, n), f'A has shape {self.A.shape}, expected {(n, n)}'
        assert self.N.shape == (m, m), f'N has shape {self.N.shape}, expected {(m, m)}'
        assert len(self.labels) == n // 2, f'{len(self.labels)} labels for {n // 2} modes'

    @classmethod
    def from_multitrap(cls, mtrap, cavity_label='cavity'):
        """
        The `MultiTrap`'s Linear Amplifier Model in quadratures

        A = T·A_c·T⁻¹ (levisim.multiphysics.covariance), B = diag(√γ_k) and N = diag(2n_k + 1) on
        both quadratures of each operator pair, so that B·N·Bᵀ is
        `covariance.quadrature_diffusion_matrix(mtrap)`. Each pair's noise enters only its own
        mode, so the noise columns are in the same order as the modes.
        """
        A = quadrature_drift_matrix(mtrap.drift_matrix)
        γ = np.asarray(mtrap.γ, dtype=float)
        factors = np.asarray(mtrap.quantum_factor_per_noise_operator, dtype=float)
        num_modes = len(γ) // 2
        per_mode_γ = np.array([γ[2*k] for k in range(num_modes)])
        per_mode_N = np.array([factors[2*k] + factors[2*k + 1] for k in range(num_modes)])
        B = np.diag(np.sqrt(np.repeat(per_mode_γ, 2)))
        N = np.diag(np.repeat(per_mode_N, 2))
        labels = [cavity_label, *mtrap.labels]
        return cls(A, B, N, labels, cavities={cavity_label: (float(mtrap.cavity.κ), (0, 1))})

    @property
    def diffusion_matrix(self):
        """
        D = B·N·Bᵀ [2π·Hz]
        """
        return self.B @ self.N @ self.B.T

    def quadrature_indices(self, label):
        """
        (index of X, index of P) of a mode in u
        """
        k = self.labels.index(label)
        return 2*k, 2*k + 1

    def largest_growth_rate(self):
        """
        max Re λ over the eigenvalues λ of A; the steady state exists only if this is negative
        [2π·Hz]
        """
        return float(np.max(np.linalg.eigvals(self.A).real))

    def steady_state_covariance(self):
        """
        Solves A·V + V·Aᵀ + B·N·Bᵀ = 0 for the intracavity covariance; raises if A is unstable
        """
        growth = self.largest_growth_rate()
        if growth >= 0:
            raise ValueError(f'drift matrix is unstable (largest Re eigenvalue {growth:.3e}), so there is no steady state')
        V = solve_continuous_lyapunov(self.A, -self.diffusion_matrix)
        return (V + V.T) / 2

    def transfer_matrix(self, ω):
        """
        G(ω)·B with G(ω) = (−iω − A)⁻¹, for an array of ω: shape (len(ω), n, m), u(ω) = G·B·w(ω)
        """
        ω = np.atleast_1d(np.asarray(ω, dtype=float))
        n = self.A.shape[0]
        M = -1j * ω[:, None, None] * np.eye(n)[None] - self.A[None]
        return np.linalg.solve(M, np.broadcast_to(self.B.astype(complex), (len(ω), *self.B.shape)))


def stack_cavity_modes(mtraps, cavity_labels, rtol=1e-9):
    """
    One `QuadratureSystem` with several cavity modes coupled to the same mechanical modes

    Each `MultiTrap` in `mtraps` describes the same particle(s) coupled to ONE cavity mode (its own
    κ, Δ and couplings); the result has the cavity modes first, in the order of `cavity_labels`,
    then the mechanical modes. The mechanical block of the drift matrix and the mechanical noise
    must be the same in every MultiTrap (checked to `rtol`), because they describe the same
    mechanics. Cavity modes interact only through the mechanics (no direct mode-mode coupling),
    which is the linearised coherent-scattering model with well-separated cavity modes.
    """
    systems = [QuadratureSystem.from_multitrap(mtrap, label) for mtrap, label in zip(mtraps, cavity_labels, strict=True)]
    K = len(systems)
    first = systems[0]
    for system in systems[1:]:
        assert system.labels[1:] == first.labels[1:], 'the MultiTraps have different mechanical modes'
        for name, a, b in (('drift', system.A[2:, 2:], first.A[2:, 2:]), ('noise input', system.B[2:, 2:], first.B[2:, 2:]),
                           ('noise', system.N[2:, 2:], first.N[2:, 2:])):
            scale = max(np.abs(b).max(), 1e-300)
            assert np.abs(a - b).max() <= rtol * scale, f'the MultiTraps have different mechanical {name} matrices'
    num_mech = first.A.shape[0] - 2
    n = 2*K + num_mech
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    N = np.zeros((n, n))
    A[2*K:, 2*K:] = first.A[2:, 2:]
    B[2*K:, 2*K:] = first.B[2:, 2:]
    N[2*K:, 2*K:] = first.N[2:, 2:]
    cavities = {}
    for k, (system, label) in enumerate(zip(systems, cavity_labels)):
        c = slice(2*k, 2*k + 2)
        A[c, c] = system.A[:2, :2]
        A[c, 2*K:] = system.A[:2, 2:]
        A[2*K:, c] = system.A[2:, :2]
        B[c, c] = system.B[:2, :2]
        N[c, c] = system.N[:2, :2]
        cavities[label] = (system.cavities[label][0], (2*k, 2*k + 1))
    return QuadratureSystem(A, B, N, [*cavity_labels, *first.labels[1:]], cavities)


def add_mechanical_mode(system, label, Ω, couplings, heating, damping):
    """
    The system with one more mechanical mode, coupled to cavity modes like levisim's dimensions
    with a real coupling G = g (dP_c/dt ∋ 2g·X, dp/dt ∋ 2g·Q_c)

    This is for motion that couples to the cavity through the SAME quadrature as the transverse
    modes at a node, e.g. the tweezer-axis motion projected onto a slightly tilted cavity axis. A
    `MultiTrap` z dimension instead couples with G = i·g, the node coherent-scattering phase.

    Ω: frequency [2π·Hz]; couplings: {cavity label: g} [2π·Hz]; heating: phonon heating rate
    [2π·Hz]; damping: bath damping of both quadratures [2π·Hz] (the bath occupation is then
    heating/damping, as in levisim's own baths)
    """
    n, m = system.B.shape
    A = np.zeros((n + 2, n + 2))
    B = np.zeros((n + 2, m + 2))
    N = np.zeros((m + 2, m + 2))
    A[:n, :n] = system.A
    B[:n, :m] = system.B
    N[:m, :m] = system.N
    X, P = n, n + 1
    A[X, X] = A[P, P] = -damping/2
    A[X, P] = Ω
    A[P, X] = -Ω
    for cavity, g in couplings.items():
        iQ, iP = system.quadrature_indices(cavity)
        A[iP, X] += 2*g
        A[P, iQ] += 2*g
    B[X, m] = B[P, m + 1] = np.sqrt(damping)
    N[m, m] = N[m + 1, m + 1] = 2*heating/damping + 1
    return QuadratureSystem(A, B, N, [*system.labels, label], dict(system.cavities))


# ----------------------------------------------------------------------------------------------
# FILTERS (ENVELOPES OF PROPAGATING MODES)
# ----------------------------------------------------------------------------------------------
@dataclass
class SpectralFilter:
    """
    Envelope ξ(ω) of a propagating mode a_ξ = ∫ ξ(ω) a_out(ω) dω/2π

    - ξ: callable of an array of ω [2π·Hz] returning complex values [1/√(2π·Hz)]
    - support: list of (ω_lo, ω_hi) intervals outside which ξ = 0, or None for a filter without
      compact support
    - breakpoints: frequencies where ξ or its derivative jumps inside the support (integration
      panels end there)
    - center, scale: for a filter without compact support, the whole real axis is integrated over
      θ ∈ (−π/2, π/2) with ω = center + scale·tan θ, so no tail is cut off [2π·Hz]
    """
    ξ: object
    support: list = None
    breakpoints: tuple = ()
    description: str = ''
    center: float = 0.0
    scale: float = None

    def __call__(self, ω):
        return self.ξ(np.asarray(ω, dtype=float))


def rectangular_filter(center, width):
    """
    Flat window ξ(ω) = √(2π/Γ_ξ) for abs(ω − center) < Γ_ξ/2, else 0 (normalised: ∫ abs(ξ)² dω/2π = 1)

    center, width [2π·Hz]. With levisim's Fourier convention a cavity mode of detuning Δ responds
    at ω = −Δ, so the Stokes sideband of a blue-detuned drive (Δ ≈ +Ω) is selected by center = −Ω.
    """
    amplitude = np.sqrt(2*np.pi / width)
    lo, hi = center - width/2, center + width/2

    def ξ(ω):
        return np.where((ω > lo) & (ω < hi), amplitude, 0.0).astype(complex)

    return SpectralFilter(ξ, support=[(lo, hi)], description=f'rectangular(center={center:.6g}, width={width:.6g})')


def lorentzian_filter(center, width):
    """
    ξ(ω) = √γ / (γ/2 − i(ω − center)) with γ = width (normalised: ∫ abs(ξ)² dω/2π = 1)

    This is the envelope of the intracavity mode of a one-sided filter cavity of linewidth γ fed by
    the output field, da_f/dt = (−i·center − γ/2) a_f + √γ a_out (see `cascade_lorentzian_filter`).
    center, width [2π·Hz]
    """
    def ξ(ω):
        return np.sqrt(width) / (width/2 - 1j*(ω - center))

    return SpectralFilter(ξ, support=None, description=f'lorentzian(center={center:.6g}, width={width:.6g})',
                          center=center, scale=width/2)


def _gauss_legendre_nodes(intervals, points_per_panel, panels_per_interval):
    """
    Composite Gauss-Legendre nodes and weights over a list of intervals
    """
    x, w = np.polynomial.legendre.leggauss(points_per_panel)
    nodes, weights = [], []
    for lo, hi in intervals:
        edges = np.linspace(lo, hi, panels_per_interval + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            nodes.append((b - a)/2 * x + (a + b)/2)
            weights.append((b - a)/2 * w)
    return np.concatenate(nodes), np.concatenate(weights)


def _merge_intervals(intervals):
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [tuple(interval) for interval in merged]


def _split_at(intervals, points):
    out = []
    for lo, hi in intervals:
        cuts = sorted(p for p in points if lo < p < hi)
        edges = [lo, *cuts, hi]
        out.extend(zip(edges[:-1], edges[1:]))
    return out


def filter_norm(spectral_filter, ω_limits=None, points_per_panel=64, panels=200):
    """
    ∫ abs(ξ(ω))² dω/2π, which is 1 for a properly normalised envelope
    """
    intervals = spectral_filter.support if spectral_filter.support is not None else [ω_limits]
    ω, weights = _gauss_legendre_nodes(intervals, points_per_panel, panels)
    return float(np.sum(weights * np.abs(spectral_filter(ω))**2) / (2*np.pi))


# ----------------------------------------------------------------------------------------------
# COVARIANCE OF PROPAGATING AND INTRACAVITY MODES
# ----------------------------------------------------------------------------------------------
def _output_transfer(system, cavity, ω, η):
    """
    Detected output quadratures (Q_out, P_out)(ω) = R(ω)·(w, v) of one cavity, shape (len(ω), 2, m + 2)

    a_out = √η (√κ a − a_in) + √(1 − η) a_v, i.e. in quadratures y = √η(√κ u_c − w_c) + √(1 − η) v,
    with v the two quadratures of the loss vacuum (appended after the m system noise columns).
    """
    κ, (iQ_in, iP_in) = system.cavities[cavity]
    iQ, iP = system.quadrature_indices(cavity)
    GB = system.transfer_matrix(ω)
    m = system.B.shape[1]
    R = np.zeros((len(GB), 2, m + 2), dtype=complex)
    R[:, :, :m] = np.sqrt(η) * np.sqrt(κ) * GB[:, [iQ, iP], :]
    R[:, 0, iQ_in] -= np.sqrt(η)
    R[:, 1, iP_in] -= np.sqrt(η)
    R[:, 0, m] = np.sqrt(1 - η)
    R[:, 1, m + 1] = np.sqrt(1 - η)
    return R


def _filtered_rows(spectral_filter, ω):
    """
    Rows h(ω) with (Q_ξ, P_ξ) = ∫ h(ω)·(Q_out, P_out)(ω) dω/2π, shape (len(ω), 2, 2)

    a_out(ω) = (Q + iP)/2 and a_out†(ω) = (Q − iP)/2 (transforms of the Hermitian quadratures), so
    with ξ̄(ω) = ξ*(−ω): Q_ξ = ∫ [½(ξ + ξ̄) Q + ½i(ξ − ξ̄) P], P_ξ = ∫ [−½i(ξ − ξ̄) Q + ½(ξ + ξ̄) P].
    """
    ξ = spectral_filter(ω)
    ξ_bar = np.conjugate(spectral_filter(-ω))
    h = np.empty((len(ω), 2, 2), dtype=complex)
    h[:, 0, 0] = (ξ + ξ_bar) / 2
    h[:, 0, 1] = 1j * (ξ - ξ_bar) / 2
    h[:, 1, 0] = -1j * (ξ - ξ_bar) / 2
    h[:, 1, 1] = (ξ + ξ_bar) / 2
    return h


def output_mode_covariance(system, cavity, spectral_filter, η=1.0, modes=None, ω_limits=None,
                           points_per_panel=32, panels=16, extra_breakpoints=()):
    """
    Joint covariance of the filtered output mode of `cavity` and intracavity modes

    Returns V for u = (Q_ξ, P_ξ, X_1, P_1, ...) with the intracavity modes `modes` (labels of
    `system`, default: all modes except `cavity`) after the filtered mode; vacuum = identity.

    - intracavity block: exact Lyapunov solution (`QuadratureSystem.steady_state_covariance`)
    - filtered-mode block and its cross-correlations with the intracavity modes:
      V_kl = ∫ Re[r_k(ω)* N r_l(ω)ᵀ] dω/2π with r the transfer rows from the noise to each
      observable, by composite Gauss-Legendre quadrature over the filter's support and its mirror
      image (ξ*(−ω) enters P_ξ and Q_ξ). Equal-time intracavity quadratures have r = e_i·G(ω)·B.

    - η: detection efficiency (the losses mix vacuum into the detected field) [·]
    - ω_limits: (ω_lo, ω_hi) to integrate over when the filter has no compact support; by default
      such a filter is integrated over the whole axis through its tan mapping (`SpectralFilter.scale`) [2π·Hz]
    - extra_breakpoints: frequencies to start new integration panels at (e.g. sharp resonances)
    """
    if modes is None:
        modes = [label for label in system.labels if label != cavity]
    V_cavity = system.steady_state_covariance()
    state_indices = np.ravel([system.quadrature_indices(label) for label in modes])

    points = [*spectral_filter.breakpoints, *[-p for p in spectral_filter.breakpoints], *extra_breakpoints]
    if spectral_filter.support is not None:
        intervals = list(spectral_filter.support) + [(-hi, -lo) for lo, hi in spectral_filter.support]
        intervals = _split_at(_merge_intervals(intervals), points)
        ω, weights = _gauss_legendre_nodes(intervals, points_per_panel, panels)
    elif ω_limits is not None:
        intervals = _split_at([tuple(ω_limits)], points)
        ω, weights = _gauss_legendre_nodes(intervals, points_per_panel, panels)
    else:
        assert spectral_filter.scale is not None, 'a filter without compact support needs ω_limits or a scale'
        c, a = spectral_filter.center, spectral_filter.scale
        θ_points = [np.arctan((p - c) / a) for p in points]
        intervals = _split_at([(-np.pi/2, np.pi/2)], θ_points)
        θ, θ_weights = _gauss_legendre_nodes(intervals, points_per_panel, panels)
        ω = c + a*np.tan(θ)
        weights = θ_weights * a / np.cos(θ)**2

    m = system.B.shape[1]
    R = _output_transfer(system, cavity, ω, η)
    r_filtered = np.einsum('wij,wjk->wik', _filtered_rows(spectral_filter, ω), R)
    GB = system.transfer_matrix(ω)
    r_state = np.zeros((len(ω), len(state_indices), m + 2), dtype=complex)
    r_state[:, :, :m] = GB[:, state_indices, :]

    N_total = np.zeros((m + 2, m + 2))
    N_total[:m, :m] = system.N
    N_total[m:, m:] = np.eye(2)

    def integrate(r_k, r_l):
        integrand = np.einsum('wik,kl,wjl->wij', np.conjugate(r_k), N_total, r_l)
        return np.einsum('w,wij->ij', weights, integrand).real / (2*np.pi)

    V_ff = integrate(r_filtered, r_filtered)
    V_fs = integrate(r_filtered, r_state)
    V_fs_mirror = integrate(r_state, r_filtered).T
    V_fs = (V_fs + V_fs_mirror) / 2

    num = 2 + len(state_indices)
    V = np.zeros((num, num))
    V[:2, :2] = (V_ff + V_ff.T) / 2
    V[:2, 2:] = V_fs
    V[2:, :2] = V_fs.T
    V[2:, 2:] = V_cavity[np.ix_(state_indices, state_indices)]
    return V


def rectangular_filter_scan(system, cavity, center, widths, η=1.0, modes=None, points_per_panel=16, max_panel_width=None):
    """
    `output_mode_covariance` for rectangular windows of several widths around one center, at the
    cost of one evaluation of the transfer matrix

    The flat window has amplitude √(2π/Γ_ξ), so its filtered-mode block scales as 2π/Γ_ξ and the
    cross-correlations as √(2π/Γ_ξ) times integrals of the unit window. Those integrals are summed
    panel by panel, with a panel edge at every window edge center ± Γ_ξ/2, and reused for every width.
    Returns an array of shape (len(widths), 2 + 2·len(modes), 2 + 2·len(modes)), in the order of
    `widths`. max_panel_width: longest panel [2π·Hz] (default: an eighth of the smallest width).
    """
    widths = np.atleast_1d(np.asarray(widths, dtype=float))
    assert np.all(widths > 0), 'widths must be positive'
    if modes is None:
        modes = [label for label in system.labels if label != cavity]
    half_max = widths.max() / 2
    assert abs(center) > half_max, 'the window and its mirror image overlap (|center| <= largest width/2)'
    if max_panel_width is None:
        max_panel_width = widths.min() / 8
    V_cavity = system.steady_state_covariance()
    state_indices = np.ravel([system.quadrature_indices(label) for label in modes])
    m = system.B.shape[1]

    offsets = np.unique(np.concatenate([[0.0], widths/2, -widths/2]))
    panels = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        num = max(1, int(np.ceil((hi - lo) / max_panel_width)))
        edges = np.linspace(lo, hi, num + 1)
        panels.extend(zip(edges[:-1], edges[1:]))
    panels = np.array(panels)                       # offsets from the center
    x, w = np.polynomial.legendre.leggauss(points_per_panel)
    nodes = (panels[:, 1:2] - panels[:, 0:1])/2 * x[None] + panels.mean(axis=1)[:, None]
    weights = (panels[:, 1:2] - panels[:, 0:1])/2 * w[None]
    num_panels = len(panels)

    N_total = np.zeros((m + 2, m + 2))
    N_total[:m, :m] = system.N
    N_total[m:, m:] = np.eye(2)

    # per panel: the window part (ξ = 1 at center + offset) and its mirror (ξ̄ = 1 at −center − offset)
    I_ff = np.zeros((num_panels, 2, 2))
    I_fs = np.zeros((num_panels, 2, len(state_indices)))
    for sign in (+1, -1):
        ω = (sign * (center + nodes)).ravel()
        R = _output_transfer(system, cavity, ω, η)
        ξ = np.ones_like(ω, dtype=complex) if sign > 0 else np.zeros_like(ω, dtype=complex)
        ξ_bar = 1 - ξ
        h = np.empty((len(ω), 2, 2), dtype=complex)
        h[:, 0, 0] = (ξ + ξ_bar) / 2
        h[:, 0, 1] = 1j * (ξ - ξ_bar) / 2
        h[:, 1, 0] = -1j * (ξ - ξ_bar) / 2
        h[:, 1, 1] = (ξ + ξ_bar) / 2
        r_filtered = np.einsum('wij,wjk->wik', h, R)
        GB = system.transfer_matrix(ω)
        r_state = np.zeros((len(ω), len(state_indices), m + 2), dtype=complex)
        r_state[:, :, :m] = GB[:, state_indices, :]
        ww = weights.ravel()
        ff = np.einsum('wik,kl,wjl->wij', np.conjugate(r_filtered), N_total, r_filtered)
        fs = np.einsum('wik,kl,wjl->wij', np.conjugate(r_filtered), N_total, r_state)
        sf = np.einsum('wik,kl,wjl->wji', np.conjugate(r_state), N_total, r_filtered)
        I_ff += (ww[:, None, None] * ff).reshape(num_panels, points_per_panel, 2, 2).sum(axis=1).real
        I_fs += (ww[:, None, None] * (fs + sf) / 2).reshape(num_panels, points_per_panel, 2, -1).sum(axis=1).real

    out = np.zeros((len(widths), 2 + len(state_indices), 2 + len(state_indices)))
    middles = panels.mean(axis=1)
    for i, width in enumerate(widths):
        inside = np.abs(middles) < width/2
        amplitude = np.sqrt(2*np.pi / width)
        V_ff = amplitude**2 * I_ff[inside].sum(axis=0) / (2*np.pi)
        V_fs = amplitude * I_fs[inside].sum(axis=0) / (2*np.pi)
        out[i, :2, :2] = (V_ff + V_ff.T) / 2
        out[i, :2, 2:] = V_fs
        out[i, 2:, :2] = V_fs.T
        out[i, 2:, 2:] = V_cavity[np.ix_(state_indices, state_indices)]
    return out


def cascade_lorentzian_filter(system, cavity, center, width, η=1.0, label='filter'):
    """
    The system plus a virtual one-sided filter cavity driven by the detected output of `cavity`

    da_f/dt = (−i·center − γ/2) a_f + √γ a_out with γ = width and
    a_out = √η(√κ a − a_in) + √(1 − η) a_v: the filter cavity does not act back on the system (it is
    cascaded), and in the steady state a_f is exactly the propagating mode with the envelope
    `lorentzian_filter(center, width)`. Its covariance with the intracavity modes then follows from
    one Lyapunov solve of the returned (larger) system, with the filter mode last, labelled `label`.
    center, width [2π·Hz]
    """
    κ, (iQ_in, iP_in) = system.cavities[cavity]
    iQ, iP = system.quadrature_indices(cavity)
    n, m = system.B.shape
    γ = width
    A = np.zeros((n + 2, n + 2))
    B = np.zeros((n + 2, m + 2))
    N = np.zeros((m + 2, m + 2))
    A[:n, :n] = system.A
    B[:n, :m] = system.B
    N[:m, :m] = system.N
    N[m:, m:] = np.eye(2)
    # filter cavity in quadratures: d(Q_f, P_f)/dt = [[-γ/2, center], [-center, -γ/2]]·(Q_f, P_f) + √γ·y
    A[n:, n:] = [[-γ/2, center], [-center, -γ/2]]
    A[n, iQ] = np.sqrt(γ * η * κ)
    A[n + 1, iP] = np.sqrt(γ * η * κ)
    # y also contains -√η·(input quadratures of the cavity), i.e. -√η times those rows of B·w
    B[n, :m] = -np.sqrt(γ * η) * _input_rows(system, iQ_in)
    B[n + 1, :m] = -np.sqrt(γ * η) * _input_rows(system, iP_in)
    B[n, m] = np.sqrt(γ * (1 - η))
    B[n + 1, m + 1] = np.sqrt(γ * (1 - η))
    cavities = dict(system.cavities)
    return QuadratureSystem(A, B, N, [*system.labels, label], cavities)


def _input_rows(system, column):
    """
    Unit row selecting one input-noise column of w
    """
    row = np.zeros(system.B.shape[1])
    row[column] = 1.0
    return row
