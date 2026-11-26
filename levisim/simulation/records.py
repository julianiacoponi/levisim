r"""
Output records of a `CavitySimulation`: what a detector writes down while the system evolves

Every record is a linear functional of the simulated state u and of the white noises w that
drive it. It is integrated alongside the state as an AUGMENTED output, ::

    dY = (C·u) dt + D·dW + R·Y dt,        Y(t_k) = 0 at the start of every sample interval,

and the stored sample is the average ΔY/Δt over the interval [t_k, t_k + Δt], which is what a
digitiser records. The same noise increments dW that drive the cavity enter the record (through
D); that correlation is what gives ponderomotive squeezing and heterodyne sideband asymmetry, and
it is exact because the record is part of the same linear system that is discretised exactly
(`levisim.simulation.cavity`). Averaging over the interval multiplies the spectrum by
sinc²(fΔt) and does not alias white noise: a white input of two-sided level S gives a sampled
level S exactly.

Conventions (levisim's, see `MultiTrapBase.homodyne_quadrature`)
-----------------------------------------------------------------
- Output field of a cavity port j with rate κ_j: a_out = -√κ_j·a + a_in,j.
- Output quadratures X_out = a_out + a_out† and P_out = i(a_out − a_out†); note that P_out is
  minus the quadrature i(a† − a) of the state vector u (`levisim.multiphysics.covariance`).
- Homodyne angle θ: h = X_out·cos θ + P_out·sin θ = a_out·e^{iθ} + a_out†·e^{-iθ}, as in
  `MultiTrap.homodyne_spectrum` with `cavity.θhom = θ`.
- Optical records are in shot-noise units by default: the vacuum has the flat TWO-SIDED level 1
  (levisim's convention ∫S dω/2π = 〈h²〉), i.e. 2 in a one-sided `SpectralDensityEstimate`.
- A classical record has a symmetric spectrum, so it is compared with the symmetrised spectrum
  ½[S(ω) + S(-ω)] of `MultiTrap`.

Records
-------
- `Homodyne(θ, η)`: one output quadrature
- `Heterodyne(Ωhet, η)`: the beat of the output field with a local oscillator detuned by Ωhet
- `PhotonNumber(η)`: the fluctuation of the output photon flux about the static field (linearised)
- `Position(label)`: a mechanical position, x_zpf-scaled to metres; sample-averaged like every
  record by default, or the instantaneous value at the sample time (`sampling='instantaneous'`)

Detection realism (optical records): efficiency η (vacuum admixture), local-oscillator phase noise
(`PhaseNoise`), an electronic noise floor, and conversion to volts (`VoltageConversion`). Homodyne
and heterodyne records can also carry the static output carrier (`static_carrier=True`): the DC
term 2√η·|ᾱ_out|·cos(θ − θ_α) and, with phase noise, its conversion into the record.
"""

from dataclasses import dataclass

import numpy as np

from levisim.core.constants import ħ

__all__ = [
    'ELEMENTARY_CHARGE',
    'Heterodyne',
    'Homodyne',
    'OpticalRecord',
    'PhaseNoise',
    'PhotonNumber',
    'Position',
    'Record',
    'VoltageConversion',
]

# elementary charge [C]
ELEMENTARY_CHARGE = 1.602176634e-19


@dataclass
class RecordRows:
    """
    The rows a record adds to the augmented linear system (see the module docstring)

    - C: (q, n) rows on the state u [2π·Hz]^½-scaled as the outputs require
    - D: (q, m) rows on the system's noise w
    - E: (q, e) rows on e PRIVATE vacuum noises of this record (e.g. the vacuum mixed in by a
      detection efficiency η < 1), unit symmetrised correlation
    - R: (q, q) drift of the auxiliary integrals (zero except for a heterodyne, whose rotation
      demodulates the record within the sample interval exactly)
    """
    C: np.ndarray
    D: np.ndarray
    E: np.ndarray
    R: np.ndarray


class Record:
    """
    Base class of the records of a `CavitySimulation`

    A record supplies `rows(sim)` (a `RecordRows`) and `finish(values, sim, rng)`, which turns the
    sample averages of its auxiliary integrals, shape (K, num_samples, q), into the recorded values,
    shape (K, num_samples). `label` names the record in `sim.traces`; `units` is set by `rows`.
    """
    default_label = 'record'

    def __init__(self, label=None):
        self.label = label or self.default_label
        self.units = '·'

    def rows(self, sim):
        raise NotImplementedError

    def finish(self, values, sim):
        raise NotImplementedError

    def __repr__(self):
        return f'{type(self).__name__}(label={self.label!r})'


# --------------------------------------------------------------------------------------------------
# DETECTION REALISM
# --------------------------------------------------------------------------------------------------
class PhaseNoise:
    """
    Phase noise φ(t) of the local oscillator, added to the detection phase of an optical record

    The phase is held constant over each sample interval (it must be slow against Δt). Models:

    - `PhaseNoise.white(S_φ)`: independent φ_k each sample, one-sided PSD S_φ [rad²/Hz] (the same
      one-sided convention as `detection_floor`), i.e. Var φ_k = S_φ/(2Δt)
    - `PhaseNoise.random_walk(D)`: φ_{k+1} = φ_k + √(D·Δt)·N(0, 1), phase diffusion D [rad²/s]
      (a free-running laser of Lorentzian linewidth Δν has D = 2π·Δν), starting at φ_0 = 0
    - `PhaseNoise.ornstein_uhlenbeck(σ, τ)`: a stationary Gaussian phase of standard deviation σ
      [rad] and correlation time τ [s], stepped exactly, φ_0 ~ N(0, σ²). For τ much longer than
      the record, every trajectory has its own static phase error drawn from N(0, σ²); the spectrum
      averaged over an ensemble then equals
      `levisim.multiphysics.quadrature_squeezing.phase_noise_averaged` with σθ2 = σ².

    By default only the detected fluctuations are rotated: the large term ∝ abs(ᾱ_out)·sin(θ - θ_α)·δφ
    that LO phase noise produces from a static output field is not part of the linearised record.
    `Homodyne(..., static_carrier=True)` (or `Heterodyne`) adds the static carrier, and with it that
    conversion, to the record.
    """

    def __init__(self, model, **parameters):
        if model not in ('white', 'random_walk', 'ornstein_uhlenbeck'):
            raise ValueError(f"Unknown phase-noise model {model!r}: use 'white', 'random_walk' or 'ornstein_uhlenbeck'")
        self.model = model
        self.parameters = parameters

    @classmethod
    def white(cls, S_φ):
        """White phase noise of one-sided PSD S_φ [rad²/Hz]"""
        return cls('white', S_φ=float(S_φ))

    @classmethod
    def random_walk(cls, D):
        """Phase diffusion (random walk) at rate D [rad²/s]"""
        return cls('random_walk', D=float(D))

    @classmethod
    def ornstein_uhlenbeck(cls, σ, τ):
        """Stationary Gaussian phase noise, standard deviation σ [rad], correlation time τ [s]"""
        return cls('ornstein_uhlenbeck', σ=float(σ), τ=float(τ))

    def sample(self, normals, Δt):
        """
        Phases φ_k for an array of standard normals of shape (K, num_samples) [rad]
        """
        p = self.parameters
        if self.model == 'white':
            return np.sqrt(p['S_φ'] / (2 * Δt)) * normals
        if self.model == 'random_walk':
            steps = np.sqrt(p['D'] * Δt) * normals
            φ = np.cumsum(steps, axis=-1)
            return φ - steps[..., :1]        # φ_0 = 0
        σ, τ = p['σ'], p['τ']
        a = np.exp(-Δt / τ)
        b = σ * np.sqrt(-np.expm1(-2 * Δt / τ))
        φ = np.empty_like(normals)
        φ[..., 0] = σ * normals[..., 0]
        # exact AR(1) recursion of the Ornstein-Uhlenbeck process (scipy.signal.lfilter is the same
        # linear recursion in compiled code)
        from scipy.signal import lfilter
        φ[..., 1:] = lfilter([1.0], [1.0, -a], b * normals[..., 1:], axis=-1,
                             zi=(a * φ[..., :1]))[0]
        return φ

    def __repr__(self):
        args = ', '.join(f'{k}={v:.4g}' for k, v in self.parameters.items())
        return f'PhaseNoise.{self.model}({args})'


@dataclass
class VoltageConversion:
    """
    Conversion of a shot-noise-normalised optical record to the voltage of a photodetector

    ::

        V(t) = G·√(e·ℛ·P)·h(t)

    with G the transimpedance gain [V/A], ℛ the responsivity [A/W] and P the optical power that
    sets the shot noise: the local-oscillator power for (balanced) homodyne and heterodyne
    detection, the mean detected signal power for direct detection (`PhotonNumber`, where
    `power=None` uses the simulated mean output power). A record h of two-sided level 1 for the
    vacuum then has the one-sided shot-noise level 2·e·I·G² [V²/Hz] with I = ℛ·P, the textbook
    shot noise of the photocurrent. The detection efficiency (including the photodiode's quantum
    efficiency) is set separately by the record's η; ℛ only sets the shot-noise current.

    power [W], responsivity [A/W], transimpedance_gain [V/A]
    """
    power: float = None
    responsivity: float = 0.7
    transimpedance_gain: float = 1e4

    def volts_per_shot_noise_unit(self, power=None):
        """G·√(e·ℛ·P) [V·√s]"""
        P = self.power if self.power is not None else power
        if P is None:
            raise ValueError('VoltageConversion needs the optical power that sets the shot noise (power=...)')
        return self.transimpedance_gain * np.sqrt(ELEMENTARY_CHARGE * self.responsivity * P)


# --------------------------------------------------------------------------------------------------
# OPTICAL RECORDS
# --------------------------------------------------------------------------------------------------
class OpticalRecord(Record):
    """
    Base class of the records of the light leaving a cavity port

    Keyword arguments (all optional):

    - cavity: label of the cavity mode (default: the system's only cavity mode)
    - port: index of the cavity port (default: the cavity's detected port, see `Cavity.κ_ports`)
    - η [1]: detection efficiency; the detected field is √η·a_out + √(1 − η)·a_vac with a vacuum
      a_vac private to this record
    - phase_noise [None]: a `PhaseNoise` of the local oscillator
    - electronic_noise [0]: white noise added to the record, one-sided PSD in the record's final
      units² per Hz (shot-noise units, or V²/Hz when `volts` is given)
    - volts [None]: a `VoltageConversion`; the record is then in volts
    - static_carrier [False]: add the static output carrier (homodyne and heterodyne only, see below)
    - α [None]: the static intracavity field for static_carrier (and `PhotonNumber`) [√photons]
      (default: from the MultiTrap, or the nonlinear equilibrium; required for a QuadratureSystem)
    - label: name in `sim.traces`

    The record is built from the two detected quadratures (X_out, P_out), sample-averaged; the
    detection phase θ_k of sample k (plus φ_k of the phase noise) then selects
    h_k = X_k·cos θ_k + P_k·sin θ_k.

    Static carrier: the linearised record holds the fluctuations only. With static_carrier=True the
    static output field ᾱ_out = -√κ_j·α of the port is added, i.e. the sample average of ::

        2·√η·abs(ᾱ_out)·cos(θ(t) + φ_k − θ_α),      θ_α = −arg ᾱ_out,

    in the record's units (√(photons/s) for shot-noise units). It is the DC term of the record and,
    with phase noise φ, its conversion ≈ −2·√η·abs(ᾱ_out)·sin(θ − θ_α)·φ_k into the record; the
    full cosine is kept, so the conversion is exact for any φ. A heterodyne's carrier is its beat
    note at Ωhet, averaged exactly over each sample interval. The carrier draws no random numbers,
    so the fluctuations are unchanged. Traces subtract their mean when the simulation has
    zero_mean=True; `sim.records` keep it. `sim.record_spectrum` describes the fluctuations only
    (it excludes the carrier's DC line and LO phase noise).
    """

    def __init__(self, cavity=None, port=None, η=1.0, phase_noise=None, electronic_noise=0.0, volts=None,
                 static_carrier=False, α=None, label=None):
        super().__init__(label)
        if not 0 <= η <= 1:
            raise ValueError(f'detection efficiency η = {η} should be in [0, 1]')
        self.cavity = cavity
        self.port = port
        self.η = float(η)
        self.phase_noise = phase_noise
        self.electronic_noise = float(electronic_noise or 0)
        self.volts = volts
        self.static_carrier = bool(static_carrier)
        self.α = α

    # the rotation of the auxiliary integrals (non-zero for a heterodyne)
    rotation_rate = 0.0

    def rows(self, sim):
        system = sim.system
        self.cavity_label = sim.resolve_cavity(self.cavity)
        κ_j, (iQ_in, iP_in), self.port_index = sim.port(self.cavity_label, self.port)
        iQ, iP = system.quadrature_indices(self.cavity_label)
        n, m = system.B.shape
        C = np.zeros((2, n))
        D = np.zeros((2, m))
        s = np.sqrt(self.η)
        # X_out = -√κ_j·u_Q + w_Q ;  P_out = i(a_out - a_out†) = -(-√κ_j·u_P + w_P)
        C[0, iQ] = -s * np.sqrt(κ_j)
        D[0, iQ_in] = s
        C[1, iP] = +s * np.sqrt(κ_j)
        D[1, iP_in] = -s
        if self.η < 1:
            E = np.sqrt(1 - self.η) * np.eye(2)
        else:
            E = np.zeros((2, 0))
        Ω = self.rotation_rate
        R = np.array([[0.0, -Ω], [Ω, 0.0]])
        self.units = 'V' if self.volts is not None else 'shot noise units'
        self.κ_port = κ_j
        if self.static_carrier:
            α = complex(self.α) if self.α is not None else sim.static_field(self.cavity_label)
            α_out = -np.sqrt(κ_j) * α
            self.carrier_amplitude = 2 * np.sqrt(self.η) * abs(α_out)    # √(photons/s)
            self.θ_carrier = -np.angle(α_out)
        return RecordRows(C, D, E, R)

    def carrier(self, θ, sim):
        """
        Sample averages of the static carrier 2√η·abs(ᾱ_out)·cos(θ(t) − θ_α) for detection phases θ
        at the END of each sample interval (phase noise included), in shot-noise units; a rotating
        local oscillator (rotation_rate Ω ≠ 0) is averaged exactly over the interval
        """
        c = θ - self.θ_carrier
        Ω = self.rotation_rate
        if Ω == 0:
            return self.carrier_amplitude * np.cos(c)
        ΩΔt = Ω * sim.Δt
        return self.carrier_amplitude * (np.sin(c) - np.sin(c - ΩΔt)) / ΩΔt

    def detection_phases(self, sim, K):
        """θ_k of every sample, shape (K, num_samples) or broadcastable [rad]"""
        raise NotImplementedError

    def shot_noise_power(self, sim):
        """Optical power that sets the shot noise, for `VoltageConversion` when power=None [W]"""
        return

    def finish(self, values, sim):
        K, N, _ = values.shape
        θ = self.detection_phases(sim, K)
        self.phase_noise_samples = None
        if self.phase_noise is not None:
            # kept, shape (K, num_samples), so the spectrum can be compared with the phases actually drawn
            self.phase_noise_samples = self.phase_noise.sample(sim.standard_normals((K, N)), sim.Δt)
            θ = θ + self.phase_noise_samples
        h = values[..., 0] * np.cos(θ) + values[..., 1] * np.sin(θ)
        if self.static_carrier:
            h = h + self.carrier(θ, sim)
        h = h * self.scale(sim)
        if self.electronic_noise:
            h = h + np.sqrt(self.electronic_noise / (2 * sim.Δt)) * sim.standard_normals(h.shape)
        return h

    def scale(self, sim):
        """Factor from shot-noise units to the record's units"""
        if self.volts is None:
            return 1.0
        return self.volts.volts_per_shot_noise_unit(self.shot_noise_power(sim))


class Homodyne(OpticalRecord):
    """
    Homodyne record of one output quadrature, h = X_out·cos θ + P_out·sin θ

    θ [rad]: homodyne angle; default `cavity.θhom` of the simulated MultiTrap (required for a
    QuadratureSystem). The symmetrised spectrum of the record is
    ½[S(ω) + S(-ω)]·sinc²(fΔt) of `MultiTrap.homodyne_spectrum` at `cavity.θhom = θ`. Other
    keyword arguments: see `OpticalRecord`.
    """
    default_label = 'homodyne'

    def __init__(self, θ=None, **kwargs):
        super().__init__(**kwargs)
        self.θ = θ

    def rows(self, sim):
        rows = super().rows(sim)
        self.θ_used = float(self.θ) if self.θ is not None else sim.default_homodyne_angle()
        return rows

    def detection_phases(self, sim, K):
        return np.full((1, 1), self.θ_used)

    def __repr__(self):
        return f'Homodyne(label={self.label!r}, θ={self.θ!r}, η={self.η})'


class Heterodyne(OpticalRecord):
    r"""
    Heterodyne record: the beat of the output field with a local oscillator detuned by Ωhet

    h(t) = X_out·cos θ(t) + P_out·sin θ(t) with θ(t) = Ωhet·t + φ0, i.e. a homodyne whose angle
    rotates at Ωhet. The rotation inside each sample interval is integrated exactly (the record's
    auxiliary integrals rotate at Ωhet), so Ωhet may be a sizeable fraction of the sample rate.

    What it records, and how it relates to `MultiTrap.heterodyne_spectrum`: levisim's
    heterodyne spectrum is S_aout,aout(ω), the spectrum of a_out in the frame of the local
    oscillator (Ωhet = 0). With levisim's Fourier convention h(ω) = e^{iφ0}·a_out(ω + Ωhet) +
    e^{-iφ0}·a_out†(ω − Ωhet), so the record's spectrum is the sum of the two sidebands ::

        S_h(ω) = S_aout,aout(ω + Ωhet) + S_aout†,aout†(ω − Ωhet),

    symmetrised as ½[S_h(ω) + S_h(-ω)] (`heterodyne_record_spectrum`). The vacuum gives 1. The
    asymmetry between the Stokes and anti-Stokes sidebands of S_aout,aout appears at
    ω = ±(Ωhet + Ω) and ±(Ωhet − Ω), well apart if Ωhet exceeds the mechanical frequencies.

    Ωhet [2π·Hz]: default `cavity.Ωhet` of the simulated MultiTrap (must be non-zero); φ0 [rad].
    Other keyword arguments: see `OpticalRecord`.
    """
    default_label = 'heterodyne'

    def __init__(self, Ωhet=None, φ0=0.0, **kwargs):
        super().__init__(**kwargs)
        self.Ωhet = Ωhet
        self.φ0 = float(φ0)

    def rows(self, sim):
        Ω = float(self.Ωhet) if self.Ωhet is not None else sim.default_heterodyne_frequency()
        if Ω == 0:
            raise ValueError('Heterodyne needs a non-zero Ωhet (cavity.Ωhet is 0): pass Ωhet=..., or use Homodyne')
        self.Ωhet_used = Ω
        self.rotation_rate = Ω
        return super().rows(sim)

    def detection_phases(self, sim, K):
        # the auxiliary integral of sample k ends at t_{k+1}: X_k cos θ_{k+1} + P_k sin θ_{k+1} is
        # the exact average of h(t) over [t_k, t_{k+1}] (see the module docstring of cavity.py)
        k = np.arange(1, sim.num_samples + 1)
        return (self.φ0 + self.Ωhet_used * sim.Δt * k)[None, :]

    def __repr__(self):
        return f'Heterodyne(label={self.label!r}, Ωhet={self.Ωhet!r}, η={self.η})'


class PhotonNumber(OpticalRecord):
    """
    Fluctuation of the output photon flux about its static value, linearised

    With the static output field ᾱ_out = -√κ_j·α (α the static intracavity field, `cavity.A0`,
    `cavity.P0` of the MultiTrap) the flux is ``Φ = |ᾱ_out + δa_out|² ≈ |ᾱ_out|² + |ᾱ_out|·h(θ_α)``,
    a homodyne of the fluctuations at the amplitude angle θ_α = -arg ᾱ_out. Direct detection has
    no local oscillator, so phase noise does not apply.

    units: 'shot noise' (default; h(θ_α), vacuum level 1) or 'photons/s' (√(η·Φ̄)·h, whose
    two-sided shot-noise level is the detected mean flux η·Φ̄). With `volts`, the shot noise is
    set by the mean detected power η·Φ̄·ħω unless `volts.power` is given.
    α: the static intracavity field to use [√photons] (default: from the MultiTrap; required for
    a QuadratureSystem). Raises ValueError when the static field is zero (e.g. every particle at
    a node), where the flux has no linear fluctuation.
    """
    default_label = 'photon_number'

    def __init__(self, units='shot noise', α=None, **kwargs):
        if kwargs.get('phase_noise') is not None:
            raise ValueError('PhotonNumber is direct detection: it has no local oscillator, hence no phase noise')
        if kwargs.get('static_carrier'):
            raise ValueError('PhotonNumber records the flux fluctuation; its static part is the mean flux, record.mean_flux')
        if units not in ('shot noise', 'photons/s'):
            raise ValueError(f"PhotonNumber units {units!r}: use 'shot noise' or 'photons/s'")
        super().__init__(α=α, **kwargs)
        self.flux_units = units

    def rows(self, sim):
        rows = super().rows(sim)
        α = complex(self.α) if self.α is not None else sim.static_field(self.cavity_label)
        α_out = -np.sqrt(self.κ_port) * α
        if abs(α_out) == 0:
            raise ValueError(
                'the static output field is zero (e.g. every particle at a node), so the photon flux has no'
                ' linear fluctuation: PhotonNumber needs an off-node system or an explicit α'
            )
        self.α_used = α
        self.mean_flux = abs(α_out)**2                       # photons/s
        self.θ_used = -np.angle(α_out)
        if self.volts is None and self.flux_units == 'photons/s':
            self.units = 'photons/s'
        return rows

    def detection_phases(self, sim, K):
        return np.full((1, 1), self.θ_used)

    def shot_noise_power(self, sim):
        return self.η * self.mean_flux * ħ * sim.optical_angular_frequency()

    def scale(self, sim):
        if self.volts is None and self.flux_units == 'photons/s':
            return np.sqrt(self.η * self.mean_flux)
        return super().scale(sim)


# --------------------------------------------------------------------------------------------------
# MECHANICAL RECORDS
# --------------------------------------------------------------------------------------------------
class Position(Record):
    """
    Position of one mechanical mode, sample-averaged like every record (or sampled instantaneously)

    label: the mode label, e.g. 'x1'; units 'm' (default, x = x_zpf·q with q = b + b†) or 'zpf'
    (q itself, as `MultiTrap.position_power_spectrum(zpf_scaled=True)`); detection_floor: one-sided
    white noise PSD added to the record [units²/Hz] (default: the simulation's `detection_floor`,
    as in `FreeSpaceSimulation`).

    sampling ['average']: 'average' records the average of the position over each sample interval
    [t_k, t_k + Δt] (a sinc²(fΔt) response, no aliasing of white noise); 'instantaneous' records
    the position at t_k, as `FreeSpaceSimulation` does (no sinc² response; everything beyond the
    Nyquist frequency aliases). The instantaneous sample draws no random numbers, so the other
    records of the run are unchanged.

    trace_label [None]: the name in `sim.traces` and `sim.records` (default: the mode label), so that
    one mode can be recorded both ways, e.g. Position('x1', sampling='instantaneous', trace_label='x1_inst').
    """

    SAMPLINGS = ('average', 'instantaneous')

    def __init__(self, label, units='m', detection_floor=None, sampling='average', trace_label=None):
        super().__init__(trace_label or label)
        if units not in ('m', 'zpf'):
            raise ValueError(f"Position units {units!r}: use 'm' or 'zpf'")
        if sampling not in self.SAMPLINGS:
            raise ValueError(f'Position sampling {sampling!r}: use one of {self.SAMPLINGS}')
        self.mode = label
        self.position_units = units
        self.detection_floor = detection_floor
        self.sampling = sampling

    @property
    def instantaneous(self):
        """True if the record samples the state at t_k instead of averaging over the interval"""
        return self.sampling == 'instantaneous'

    def rows(self, sim):
        system = sim.system
        n, m = system.B.shape
        iX, _ = system.quadrature_indices(self.mode)
        self.factor = sim.zero_point_fluctuation(self.mode) if self.position_units == 'm' else 1.0
        self.units = 'm' if self.position_units == 'm' else 'zpf'
        self.state_index = iX
        if self.instantaneous:
            # no auxiliary integral: the simulation samples state component iX at every t_k
            return RecordRows(np.zeros((0, n)), np.zeros((0, m)), np.zeros((0, 0)), np.zeros((0, 0)))
        C = np.zeros((1, n))
        C[0, iX] = 1.0
        return RecordRows(C, np.zeros((1, m)), np.zeros((1, 0)), np.zeros((1, 1)))

    def finish(self, values, sim):
        x = self.factor * values[..., 0]
        floor = self.detection_floor if self.detection_floor is not None else sim.detection_floor_for(self.mode)
        if floor:
            x = x + np.sqrt(floor / (2 * sim.Δt)) * sim.standard_normals(x.shape)
        return x

    def __repr__(self):
        extra = '' if self.sampling == 'average' else f', sampling={self.sampling!r}'
        if self.label != self.mode:
            extra += f', trace_label={self.label!r}'
        return f'Position({self.mode!r}, units={self.position_units!r}{extra})'
