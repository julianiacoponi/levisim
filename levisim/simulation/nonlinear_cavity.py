r"""
The nonlinear coherent-scattering model of `CavitySimulation(model='nonlinear')`

State, energy and equations of motion
-------------------------------------
The state is the complex intracavity amplitude α (levisim's frame: the frame of the tweezer
light, detuning Δ = ω_tweezer - ω_cavity) and, for every particle n, its displacement r_n from its
trap centre and its momentum p_n, all three Cartesian axes. Everything follows from ONE classical
energy, the time-averaged dipole energy of every particle in its tweezer plus the cavity field
(`levisim.physics.coherent_scattering.optical_potential`, the same function whose second
derivatives give `Trap.kj`, `Dimension.light_mech_coupling`, `Dimension.mech_mech_coupling` and the
static cavity shifts) ::

    H = -ħΔ|α|² + Σ_n [ p_n²/2m_n + U_n(r_n, α e^{-iφ_n}) + (other force elements of trap n) ]

    U_n(r, α_n) = U_tw(r) - 2·Ed_n·Re(α_n*·D_n(r)) - α_pol·E0²·|α_n|²·cos²(k·d·r + Φ_n)

with D_n(r) = sin θ·f(r)·cos(k·d·r + Φ_n)·e^{iψ(z)} the bead's scattering amplitude
(`scattering_amplitude`: standing-wave factor, drive-tweezer envelope f, tweezer phase ψ = k·z -
arctan(z/zR) including the Gouy phase) and α_n = α·e^{-iφ_n} the cavity amplitude in the frame of
tweezer n (φ_n its phase). The classical equations dα/dt = -(i/ħ)·∂H/∂α* and dp/dt = -∇H, plus
cavity decay, gas damping and the noises, are ::

    dα   = [(iΔ - κ/2)·α + Σ_n i·ωd_n·e^{iφ_n}·D_n(r_n) + Σ_n i·δ_n·cos²(k·d·r_n + Φ_n)·α] dt
           + Σ_j √κ_j dα_in,j                                       (every port and loss channel)
    dr_n = p_n/m_n dt
    dp_n = [F_elements,n(r_n) + 2·Ed_n·Re(α_n*·∇D_n) - α_pol·E0²·|α|²·k·sin(2(k·d·r_n + Φ_n))·d
            + Σ_m F_pair,nm - Γ_n·p_n] dt + thermal and photon-recoil force noise

with ωd = Ed/ħ, Ed = α_pol·E_d·E0/2 (`Trap.Ed`) and δ = α_pol·E0²/ħ (the cavity pull of one bead,
a few Hz for a 60 nm bead). The drive and the force are conjugate by construction
(`cavity_drive`, `optical_force`). Expanding H to second order in r and in α - ᾱ about the trap
centre gives exactly `MultiTrap`'s linearised Hamiltonian H/ħ = -Δa†a - Σ q_j(G_j a† + G_j* a) - ...:
the linear term in r of -2·Ed·Re(α_n*·D_n) is -ħ·g_j·q_j·(a_n + a_n†) for x, y (∝ sin Φ) and
-ħ·g_z·q_z·i(a_n† - a_n) for z (∝ cos Φ, through ψ ≈ (k - 1/zR)·z), with G_j = g_j·e^{iφ_n}
(`Dimension.G`); the r_j·r_k terms are gxy, gxz, gyz and the static shifts of
`static_cavity_hessian`. The tests check this numerically: the Jacobian of the drift below at
the node equals `MultiTrap.drift_matrix` (in quadratures) to round-off.

Quadratures and scaling
-----------------------
The model works in levisim's quadratures (vacuum covariance = identity), u = (A, P, X_x1, P_x1,
X_y1, P_y1, X_z1, P_z1, X_x2, ...) with α = (A + iP)/2, r_j = x_zpf,j·X_j and
p_j = m·Ω_j·x_zpf,j·P_j, where Ω_j is `Dimension.Ω` (the tweezer frequency plus the static cavity
shift, as `MultiTrap` uses) and x_zpf,j = √(ħ/2m·Ω_j). These are the coordinates of
`QuadratureSystem.from_multitrap`, so records, ports and the linear propagator are shared with
`model='linear'`. Unlike `MultiTrap` the model always has all three axes of every particle; axes
that `MultiTrap` leaves out are simply extra modes (labels 'z1', ...).

Damping and noise: `damping='velocity'` (default) is the physical Langevin form, a force -Γ·p
with momentum diffusion 2Γ(2n̄ + 1) + 4Γ_rec in P-quadrature units (thermal occupation n̄ of the
bath, photon-recoil heating rate Γ_rec, both as `MultiTrap` uses them), as in
`FreeSpaceSimulation`. `damping='symmetric'` is the rotating-wave form that `MultiTrap`'s drift
matrix uses (both quadratures damped at Γ/2, each with diffusion Γ(2n̄ + 1) + 2Γ_rec); it makes the
Jacobian equal `MultiTrap.drift_matrix` exactly. The two differ in the drift by O(Γ/Ω) and give
the same spectra near resonance. The cavity input is vacuum noise (symmetrised, unit level),
one pair of noises per Fabry-Perot port and loss channel, as in the linear model.

Equilibrium and linearisation
-----------------------------
`equilibrium()` solves drift(u) = 0 for the whole state by Newton's method from the trap centres
(the static cavity field and, off the node, the coherent-scattering force displace the particles,
e.g. tens of nm along z for a 60 nm bead at Φ = π/4). `jacobian` is the derivative of the drift
there (fourth-order central differences of the closed-form forces, per term), and
`quadrature_system` is the linear model at the TRUE equilibrium, which the split-step integrator
advances exactly.
"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.sparse import csr_matrix

from levisim.core.constants import kB, ħ
from levisim.core.logger import logger
from levisim.multiphysics.output_modes import QuadratureSystem
from levisim.physics.fields.elements import (
    Feedback,
    Gravity,
    HarmonicPotential,
    PhotonRecoil,
)
from levisim.physics.tweezer import Tweezer

__all__ = [
    'NONLINEAR_TERMS',
    'NonlinearCavityModel',
]

# the terms of the drift beyond its exact linear part (free cavity, p/m, damping), each of which
# can be left out (`terms`) or linearised about the equilibrium (`linearised_terms`)
NONLINEAR_TERMS = ('elements', 'coherent_scattering', 'cavity', 'interactions')
DAMPING_FORMS = ('velocity', 'symmetric')

# columns of the per-particle parameter table shared by the NumPy and numba drifts
_ZPF = slice(0, 3)          # x_zpf per axis [m]
_INV_P = slice(3, 6)        # 1/(m·Ω·x_zpf) per axis, force → dP/dt [1/(N·s)]
_TW = 6                     # Re α_eff, E0, wx0, wy0, zR of the trapping tweezer; flag at _TW + 5
_REC = 12                   # ½·Im α_eff, k of the recoil element; flag at _REC + 2
_GRAV = 15                  # constant force (3) [N]
_HARM_K = 18                # stiffness (9, row-major) [N/m]
_HARM_KC = 27               # K·c (3) [N]
_CS = 30                    # s, c, k, Φ, wx0, wy0, zR (drive tweezer), ωd, 2·Ed, cos φ, sin φ, δ, α_pol·E0²
NUM_PARAMETERS = 43


def _bose_einstein(Ω, T):
    return 1 / np.expm1(ħ * Ω / (kB * T))


class NonlinearCavityModel:
    """
    The nonlinear drift, noise, equilibrium and linearisation of particles coherently scattering
    into one cavity mode (see the module docstring)

    ``NonlinearCavityModel(mtrap, **options)``, with a physical-parameter `MultiTrap` (its traps,
    beads, tweezers, cavity, gas and bath occupations). Options:

        damping ['velocity']: 'velocity' (-Γ·p, the physical Langevin form) or 'symmetric' (the
            rotating-wave form of MultiTrap's drift matrix)
        terms [all of NONLINEAR_TERMS]: which terms enter the drift at all: 'elements' (the trap's
            force elements: tweezer gradient force, PhotonRecoil, Gravity, HarmonicPotential, ...),
            'coherent_scattering' (the scattering drive of the cavity and its back-action force),
            'cavity' (the dispersive term: the cavity's own standing-wave force and the beads'
            pull of the cavity frequency), 'interactions'
        linearised_terms [()]: terms replaced by their linearisation about the equilibrium (a
            diagnostic: e.g. ('elements',) keeps the tweezer harmonic)
        interactions [()]: objects with forces(positions (N, 3), particles, t) -> (N, 3) [N] for
            lab-frame positions (the round-5 Interaction interface)
        trap_positions [None]: lab-frame trap centres (N, 3) [m] for the interactions; default each
            trap's `position` attribute if it has one, else the origin
        feedback [False]: include the damping of the traps' Feedback elements (MultiTrap ignores it)

    Attributes: labels, n (state dimension), u_eq (absolute quadratures at equilibrium), α_eq,
    r_eq (N, 3) [m], jacobian, quadrature_system, parameters (per-particle table), zpf, Ω.
    """

    def __init__(self, mtrap, damping='velocity', terms=NONLINEAR_TERMS, linearised_terms=(),
                 interactions=(), trap_positions=None, feedback=False):
        if getattr(mtrap, 'toy_params', None):
            raise ValueError(
                "model='nonlinear' needs a MultiTrap with physical parameters (tweezers and particles):"
                ' toy parameters have no potential to simulate. Use model=\'linear\''
            )
        if damping not in DAMPING_FORMS:
            raise ValueError(f'damping={damping!r}: use one of {DAMPING_FORMS}')
        unknown = (set(terms) | set(linearised_terms)) - set(NONLINEAR_TERMS)
        if unknown:
            raise ValueError(f'unknown terms {sorted(unknown)}; the terms are {NONLINEAR_TERMS}')
        self.mtrap = mtrap
        self.damping = damping
        self.interactions = list(interactions or [])
        self.terms = tuple(t for t in NONLINEAR_TERMS if t in terms and (t != 'interactions' or self.interactions))
        self.linearised_terms = tuple(t for t in self.terms if t in linearised_terms)
        self.nonlinear_terms = tuple(t for t in self.terms if t not in self.linearised_terms)
        self.feedback = bool(feedback)
        self.traps = list(mtrap.traps)
        self.particles = [trap.bead for trap in self.traps]
        self.num_particles = N = len(self.traps)
        self.n = 2 + 6 * N
        self.labels = ['cavity'] + [f'{axis}{n + 1}' for n in range(N) for axis in 'xyz']
        if trap_positions is None:
            trap_positions = [getattr(trap, 'position', None) for trap in self.traps]
            trap_positions = [np.zeros(3) if p is None else np.asarray(p, dtype=float) for p in trap_positions]
        self.trap_positions = np.asarray(trap_positions, dtype=float).reshape(N, 3)

        cavity = mtrap.cavity
        self.Δ, self.κ, self.E0 = float(cavity.Δ), float(cavity.κ), float(cavity.E0)
        self._setup_particles()
        self._setup_linear_part()
        self._setup_noise()
        self.equilibrium()
        self.linearise()

    # ------------------------------------------------------------------------------------------
    # parameters
    # ------------------------------------------------------------------------------------------
    def _setup_particles(self):
        mtrap = self.mtrap
        N = self.num_particles
        P = np.zeros((N, NUM_PARAMETERS))
        self.Ω = np.zeros((N, 3))
        self.Γ = np.zeros((N, 3))
        self.generic_elements = [[] for _ in range(N)]
        self.compilable = True
        for n, trap in enumerate(self.traps):
            index = trap.trap_index
            if trap.override_Ω_dict:
                logger.warning(
                    'trap {} has measured frequencies (override_Ω_dict): the nonlinear model integrates its force'
                    " elements, whose curvature need not match them, so its Jacobian differs from MultiTrap's", index,
                )
            m = trap.bead.m
            for a, axis in enumerate('xyz'):
                label = f'{axis}{index}'
                if label in mtrap.dimensions:
                    dim = mtrap.dimensions[label]
                    Ω, zpf, Γ = dim.Ω, dim.zpf, dim.Γ
                else:
                    Ω = mtrap.static_shifted_Ωj[index][axis] if mtrap.static_cavity_shifts else trap.Ωj[axis]
                    zpf = np.sqrt(ħ / 2 / (m * Ω))
                    Γ = trap.Γth
                self.Ω[n, a] = Ω
                self.Γ[n, a] = Γ
                P[n, a] = zpf
                P[n, 3 + a] = 1 / (m * Ω * zpf)

            tweezers = [el for el in trap.fields if isinstance(el, Tweezer)]
            recoils = [el for el in trap.fields if isinstance(el, PhotonRecoil)]
            for element in trap.fields:
                if isinstance(element, (Tweezer, PhotonRecoil)):
                    continue
                if isinstance(element, Gravity):
                    P[n, _GRAV:_GRAV + 3] += element.force(np.zeros(3), 0.0, trap.bead)
                elif isinstance(element, HarmonicPotential):
                    P[n, _HARM_K:_HARM_K + 9] += element.K.ravel()
                    P[n, _HARM_KC:_HARM_KC + 3] += element.K @ element.centre
                elif isinstance(element, Feedback):
                    if self.feedback:
                        self.Γ[n] += element.damping_rates(trap.bead)
                else:
                    self.generic_elements[n].append(element)
            if len(tweezers) == 1:
                tw = tweezers[0]
                α_eff = tw.effective_polarisability(trap.bead)
                P[n, _TW:_TW + 6] = [np.real(α_eff), tw.E0, tw.wx0, tw.wy0, tw.zR, 1.0]
            else:
                self.generic_elements[n].extend(tweezers)
            if len(recoils) == 1 and len(tweezers) == 1 and recoils[0].tweezer is tweezers[0]:
                tw = recoils[0].tweezer
                P[n, _REC:_REC + 3] = [0.5 * np.imag(tw.effective_polarisability(trap.bead)), tw.k, 1.0]
            else:
                self.generic_elements[n].extend(recoils)
            if self.generic_elements[n]:
                self.compilable = False

            drive = trap.drive_tweezer
            Ed = 0.5 * trap.bead.α * trap.cavity_drive_field * self.E0
            φ = float(trap.tweezer.φ)
            P[n, _CS:_CS + 13] = [
                np.sin(drive.θ), np.cos(drive.θ), drive.k, trap.bead.Φ, drive.wx0, drive.wy0, drive.zR,
                Ed / ħ, 2 * Ed, np.cos(φ), np.sin(φ), trap.bead.α * self.E0**2 / ħ, trap.bead.α * self.E0**2,
            ]
        if self.interactions:
            self.compilable = False
        self.parameters = P
        self.zpf = P[:, _ZPF].copy()

    def _setup_linear_part(self):
        """The exactly linear part L0 of the drift: free cavity, p/m and damping"""
        n = self.n
        L0 = np.zeros((n, n))
        Δ, κ = self.Δ, self.κ
        L0[0, 0], L0[0, 1] = -κ / 2, -Δ
        L0[1, 0], L0[1, 1] = Δ, -κ / 2
        for p in range(self.num_particles):
            for a in range(3):
                iX = 2 + 6 * p + 2 * a
                Ω, Γ = self.Ω[p, a], self.Γ[p, a]
                L0[iX, iX + 1] = Ω
                if self.damping == 'velocity':
                    L0[iX + 1, iX + 1] = -Γ
                else:
                    L0[iX, iX] = -Γ / 2
                    L0[iX + 1, iX + 1] = -Γ / 2
        self.L0 = L0

    def _setup_noise(self):
        """
        B and N of the noise (cavity ports as `QuadratureSystem.from_multitrap`, then an (X, P)
        pair of columns per mechanical axis) and the ports
        """
        mtrap = self.mtrap
        cavity = mtrap.cavity
        factors = np.asarray(mtrap.quantum_factor_per_noise_operator, dtype=float)
        γ = np.asarray(mtrap.γ, dtype=float)
        N_cavity = factors[0] + factors[1]
        if getattr(cavity, 'has_default_ports', True):
            channels = [self.κ]
            rates = [self.κ]
        else:
            rates = [float(κ_j) for κ_j in cavity.port_rates]
            channels = rates + ([cavity.κ_loss] if cavity.κ_loss > 0 else [])
        c = 2 * len(channels)
        m_noise = c + 6 * self.num_particles
        B = np.zeros((self.n, m_noise))
        Nn = np.eye(m_noise)
        for j, κ_j in enumerate(channels):
            B[0, 2 * j] = B[1, 2 * j + 1] = np.sqrt(κ_j)
            Nn[2 * j, 2 * j] = Nn[2 * j + 1, 2 * j + 1] = N_cavity
        self.diffusion_per_axis = np.zeros((self.num_particles, 3))
        T = mtrap.gas.T
        for p, trap in enumerate(self.traps):
            index = trap.trap_index
            for a, axis in enumerate('xyz'):
                label = f'{axis}{index}'
                if label in mtrap.labels:
                    k = mtrap.labels.index(label) + 1
                    Γ_bath = γ[2 * k]
                    two_n_plus_1 = factors[2 * k] + factors[2 * k + 1]
                else:
                    Γ_bath = trap.Γth
                    Ω = self.Ω[p, a]
                    n_bar = _bose_einstein(Ω, T) if T > 0 else 0.0
                    if mtrap.photon_recoil and Γ_bath > 0:
                        Γ_rec = trap.photon_recoil_heating_rates[axis] * trap.Ωj[axis] / Ω
                        n_bar = n_bar + Γ_rec / Γ_bath
                    two_n_plus_1 = 2 * n_bar + 1
                D = Γ_bath * two_n_plus_1        # per quadrature in the symmetric form
                iX = 2 + 6 * p + 2 * a
                col = c + 6 * p + 2 * a
                if self.damping == 'velocity':
                    self.diffusion_per_axis[p, a] = 2 * D
                    B[iX + 1, col + 1] = np.sqrt(2 * D)
                else:
                    self.diffusion_per_axis[p, a] = D
                    B[iX, col] = np.sqrt(D)
                    B[iX + 1, col + 1] = np.sqrt(D)
        self.B = B
        self.N = Nn
        ports = [(κ_j, (2 * j, 2 * j + 1)) for j, κ_j in enumerate(rates)]
        if getattr(cavity, 'has_default_ports', True):
            self.cavities = {'cavity': (self.κ, (0, 1))}
            self.ports = {}
        else:
            self.cavities = {'cavity': ports[cavity.detected_port]}
            self.ports = {'cavity': ports}

    # ------------------------------------------------------------------------------------------
    # the drift
    # ------------------------------------------------------------------------------------------
    def positions(self, U):
        """Displacements r (K, N, 3) [m] of absolute quadratures U (K, n)"""
        K = U.shape[0]
        X = U[:, 2:].reshape(K, self.num_particles, 3, 2)[..., 0]
        return X * self.zpf[None]

    def term_drift(self, U, terms, t=0.0):
        """
        The drift of the given terms (a subset of NONLINEAR_TERMS) at absolute quadratures U,
        shape (K, n) -> (K, n); the full drift is L0·U plus this for every term in `self.terms`
        [2π·Hz]
        """
        U = np.atleast_2d(np.asarray(U, dtype=float))
        K, N = U.shape[0], self.num_particles
        P = self.parameters
        out = np.zeros_like(U)
        if not terms:
            return out
        r = self.positions(U)
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        force = np.zeros((K, N, 3))
        dα = np.zeros(K, dtype=complex)

        if 'elements' in terms:
            tw = P[:, _TW + 5] > 0
            if np.any(tw):
                aRe, E0, wx0, wy0, zR = (P[:, _TW + i] for i in range(5))
                w = np.sqrt(1 + (z / zR)**2)
                wx, wy, wz = wx0 * w, wy0 * w, np.sqrt(2) * zR * w
                ellipse = (x / wx)**2 + (y / wy)**2
                E = (E0 / w) * np.exp(-ellipse)
                Utw = -(aRe / 2) * E**2
                Ftw = np.stack([0.5 * (4 * x / wx**2) * Utw, 0.5 * (4 * y / wy**2) * Utw,
                                0.5 * (4 * z / wz**2) * Utw * (1 - 2 * ellipse)], axis=-1)
                force += np.where(tw[None, :, None], Ftw, 0.0)
                rec = P[:, _REC + 2] > 0
                if np.any(rec):
                    Fz = P[:, _REC] * E**2 * (P[:, _REC + 1] - zR / (zR**2 + z**2))
                    force[..., 2] += np.where(rec[None], Fz, 0.0)
            force += P[None, :, _GRAV:_GRAV + 3]
            Kh = P[:, _HARM_K:_HARM_K + 9].reshape(N, 3, 3)
            if np.any(Kh):
                force += -np.einsum('nij,knj->kni', Kh, r) + P[None, :, _HARM_KC:_HARM_KC + 3]
            for p, elements in enumerate(self.generic_elements):
                for element in elements:
                    for k in range(K):
                        force[k, p] += element.force(r[k, p], t, self.particles[p])

        if 'coherent_scattering' in terms or 'cavity' in terms:
            s, c, kk, Φ, wx0, wy0, zR = (P[:, _CS + i] for i in range(7))
            u = kk * (s * x + c * y) + Φ
            cos_u, sin_u = np.cos(u), np.sin(u)
            α = (U[:, 0] + 1j * U[:, 1]) / 2
        if 'coherent_scattering' in terms:
            ωd, two_Ed, cos_φ, sin_φ = (P[:, _CS + i] for i in range(7, 11))
            w2 = 1 + (z / zR)**2
            e = (x**2 / wx0**2 + y**2 / wy0**2) / w2
            f = np.exp(-e) / np.sqrt(w2)
            ψ = kk * z - np.arctan(z / zR)
            phase = np.exp(1j * ψ)
            D = s * f * cos_u * phase
            dD_sin = s * f * sin_u * phase * kk
            gx = D * (-2 * x / (wx0**2 * w2)) - dD_sin * s
            gy = D * (-2 * y / (wy0**2 * w2)) - dD_sin * c
            gz = D * ((z / zR**2) / w2 * (2 * e - 1)) + 1j * (kk - zR / (zR**2 + z**2)) * D
            e_iφ = cos_φ + 1j * sin_φ
            α_n_conj = np.conj(α)[:, None] * e_iφ          # (α·e^{-iφ})*
            force[..., 0] += two_Ed * np.real(α_n_conj * gx)
            force[..., 1] += two_Ed * np.real(α_n_conj * gy)
            force[..., 2] += two_Ed * np.real(α_n_conj * gz)
            dα = dα + 1j * np.sum(ωd * e_iφ * D, axis=1)
        if 'cavity' in terms:
            δ, αpol_E0sq = P[:, _CS + 11], P[:, _CS + 12]
            Fcav = -αpol_E0sq * (np.abs(α)**2)[:, None] * kk * 2 * sin_u * cos_u
            force[..., 0] += Fcav * s
            force[..., 1] += Fcav * c
            dα = dα + 1j * α * np.sum(δ * cos_u**2, axis=1)
        if 'interactions' in terms and self.interactions:
            for k in range(K):
                lab = self.trap_positions + r[k]
                for interaction in self.interactions:
                    force[k] += np.asarray(interaction.forces(lab, self.particles, t), dtype=float)

        out[:, 0] = 2 * dα.real
        out[:, 1] = 2 * dα.imag
        dP = force * P[None, :, _INV_P]
        out[:, 3::2] = dP.reshape(K, 3 * N)
        return out

    def drift(self, U, t=0.0):
        """The full drift dU/dt at absolute quadratures U, shape (K, n) or (n,) [2π·Hz]"""
        U = np.asarray(U, dtype=float)
        single = U.ndim == 1
        U2 = np.atleast_2d(U)
        out = U2 @ self.L0.T + self.term_drift(U2, self.terms, t)
        return out[0] if single else out

    # ------------------------------------------------------------------------------------------
    # equilibrium and linearisation
    # ------------------------------------------------------------------------------------------
    def jacobian_steps(self):
        """Central-difference steps per quadrature: 10⁻⁴ of the optical length 1/k for X, 1 otherwise"""
        h = np.ones(self.n)
        k = self.parameters[:, _CS + 2]
        for p in range(self.num_particles):
            for a in range(3):
                h[2 + 6 * p + 2 * a] = 1e-4 / (k[p] * self.zpf[p, a])
        return h

    def term_jacobian(self, U0, terms, t=0.0):
        """∂(term drift)/∂U at U0 by fourth-order central differences, (n, n) [2π·Hz]"""
        n = self.n
        if not terms:
            return np.zeros((n, n))
        h = self.jacobian_steps()
        # only the X quadratures and the cavity enter nonlinearly; the drift is independent of P_mech
        columns = [0, 1] + [2 + 2 * j for j in range(3 * self.num_particles)]
        batch = np.repeat(U0[None], 4 * len(columns), axis=0)
        for i, col in enumerate(columns):
            for s, offset in enumerate((2, 1, -1, -2)):
                batch[4 * i + s, col] += offset * h[col]
        F = self.term_drift(batch, terms, t).reshape(len(columns), 4, n)
        J = np.zeros((n, n))
        for i, col in enumerate(columns):
            J[:, col] = (-F[i, 0] + 8 * F[i, 1] - 8 * F[i, 2] + F[i, 3]) / (12 * h[col])
        return J

    def equilibrium(self, max_iterations=200, tol=1e-12, max_displacement_step=20e-9):
        """
        Solves drift(U) = 0 by Newton's method from the trap centres (U = 0); sets u_eq, r_eq, α_eq

        Each Newton step is scaled down, if needed, so that no particle moves by more than
        max_displacement_step [m] (a trust region: the potential is periodic on the scale λ/2, and a
        full step from the trap centre can overshoot into the next well when the static field is
        strong). Converged when the Newton step is below tol relative to max(1, |U|) on every
        quadrature. Returns u_eq (absolute quadratures).
        """
        U = np.zeros(self.n)
        X_rows = np.array([2 + 2 * j for j in range(3 * self.num_particles)])
        zpf = self.zpf.ravel()
        for iteration in range(max_iterations):
            J = self.L0 + self.term_jacobian(U, self.terms)
            step = np.linalg.solve(J, -self.drift(U))
            largest = np.max(np.abs(step[X_rows]) * zpf) if len(X_rows) else 0.0
            if largest > max_displacement_step:
                step = step * (max_displacement_step / largest)
            U = U + step
            if np.all(np.abs(step) <= tol * np.maximum(1.0, np.abs(U))):
                break
        else:
            raise RuntimeError(f'the equilibrium solve did not converge in {max_iterations} Newton steps')
        self.equilibrium_iterations = iteration + 1
        self.u_eq = U
        self.r_eq = self.positions(U[None])[0]
        self.α_eq = complex(U[0] + 1j * U[1]) / 2
        self.residual = self.drift(U)
        logger.debug('Nonlinear equilibrium in {} Newton steps: α = {:.6g}, r = {} nm', iteration + 1,
                     self.α_eq, np.array2string(self.r_eq / 1e-9, precision=4))
        return U

    def linearise(self):
        """
        The Jacobian at the equilibrium, per term (`term_jacobians`), in total (`jacobian`) and as a
        `QuadratureSystem` (`quadrature_system`)
        """
        self.term_jacobians = {term: self.term_jacobian(self.u_eq, (term,)) for term in self.terms}
        self.jacobian = self.L0 + sum(self.term_jacobians.values(), np.zeros((self.n, self.n)))
        self.nonlinear_jacobian = sum((self.term_jacobians[t] for t in self.nonlinear_terms), np.zeros((self.n, self.n)))
        # sparse: each particle couples only to itself and to the cavity (pair forces aside)
        self._nonlinear_jacobian_sparse = csr_matrix(self.nonlinear_jacobian)
        self.nonlinear_drift_at_equilibrium = self.term_drift(self.u_eq[None], self.nonlinear_terms)[0]
        self.quadrature_system = QuadratureSystem(
            self.jacobian, self.B, self.N, list(self.labels), cavities=dict(self.cavities), ports=dict(self.ports),
        )
        return self.quadrature_system

    def remainder(self, u, t=0.0):
        """
        The nonlinear remainder R(u) = drift(u_eq + u) - J·u of the deviation u (K, n): the terms
        not linearised, minus their value and linearisation at the equilibrium
        [2π·Hz]
        """
        if not self.nonlinear_terms:
            return np.zeros_like(u)
        return (self.term_drift(self.u_eq[None] + u, self.nonlinear_terms, t)
                - self.nonlinear_drift_at_equilibrium[None] - (self._nonlinear_jacobian_sparse @ u.T).T)

    def steady_state_covariance(self):
        """Covariance of the deviation from equilibrium of the linearised model (raises if unstable)"""
        return self.quadrature_system.steady_state_covariance()

    def modelled_indices(self):
        """Indices in u of the quadratures that MultiTrap models (cavity, then its labels in order)"""
        indices = [0, 1]
        for label in self.mtrap.labels:
            i = self.labels.index(label)
            indices += [2 * i, 2 * i + 1]
        return np.array(indices)

    def stiffness_at_equilibrium(self):
        """
        The 3 × 3 stiffness tensors -∂F/∂r of every particle at the equilibrium (static field fixed),
        from the Jacobian, shape (N, 3, 3) [N/m]
        """
        J = self.jacobian
        K = np.zeros((self.num_particles, 3, 3))
        for p in range(self.num_particles):
            for a in range(3):
                for b in range(3):
                    iP = 2 + 6 * p + 2 * a + 1
                    iX = 2 + 6 * p + 2 * b
                    K[p, a, b] = -J[iP, iX] / (self.parameters[p, 3 + a] * self.zpf[p, b])
        return K

    def __repr__(self):
        return (f'NonlinearCavityModel({self.num_particles} particles, damping={self.damping!r}, terms={self.terms},'
                f' linearised={self.linearised_terms})')


def lyapunov_sample(system, normals):
    """Samples of the steady state of a stable QuadratureSystem: normals (K, n) -> (K, n)"""
    V = solve_continuous_lyapunov(system.A, -system.diffusion_matrix)
    V = (V + V.T) / 2
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        w, Q = np.linalg.eigh(V)
        L = Q * np.sqrt(np.clip(w, 0, None))
    return normals @ L.T
