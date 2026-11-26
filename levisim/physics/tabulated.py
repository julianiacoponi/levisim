"""
TabulatedParticle: a levitated particle whose tensors were computed elsewhere

levisim does not contain an electromagnetic solver. For a particle whose polarisability is not
captured by the analytic shapes (levisim.physics.shapes), compute it with a T-matrix,
discrete-dipole (DDA) or finite-element code and pass the numbers here, together with the mass,
inertia and (optionally) gas-damping tensors. Everything is validated on construction: shapes,
finiteness, symmetry, positive-definiteness, and passivity of the polarisability.
"""
from collections.abc import Mapping
from functools import cached_property

import numpy as np

from levisim.core.constants import π
from levisim.core.units import fg, nm
from levisim.physics.gas import Gas
from levisim.physics.material import Material
from levisim.physics.particle import _NODE, Particle

__all__ = ['TabulatedParticle']


def _matrix(name, value, allow_complex=False):
    """A finite 3×3 array from a 3×3 array, a 3-vector (the diagonal) or a scalar (times 𝟙)"""
    array = np.asarray(value)
    if not allow_complex and np.iscomplexobj(array):
        raise ValueError(f'{name} must be real')
    if array.ndim == 0:
        array = array * np.eye(3)
    elif array.shape == (3,):
        array = np.diag(array)
    if array.shape != (3, 3):
        raise ValueError(f'{name} must be a 3×3 tensor, a 3-vector diagonal or a scalar, not shape {array.shape}')
    array = array.astype(complex if np.iscomplexobj(array) else float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite')
    return array


def _check_symmetric(name, array, rtol):
    scale = np.max(np.abs(array))
    if np.max(np.abs(array - array.T)) > rtol * scale:
        raise ValueError(f'{name} must be symmetric (to a relative {rtol:g})')


def _check_inertia(inertia, rtol):
    _check_symmetric('inertia', inertia, rtol)
    eigenvalues = np.linalg.eigvalsh(0.5 * (inertia + inertia.T))
    if eigenvalues[0] <= 0:
        raise ValueError(f'inertia must be positive definite; its principal moments are {eigenvalues}')
    I1, I2, I3 = eigenvalues
    if I1 + I2 < I3 * (1 - rtol):
        raise ValueError(
            f'the principal moments {eigenvalues} violate the triangle inequality I1 + I2 ≥ I3 of a rigid body'
        )


def _check_polarisability(α, rtol, where=''):
    _check_symmetric(f'polarisability{where}', α, rtol)
    # passivity: the anti-Hermitian part (α - α^H)/2i, which is Im α for a symmetric α, must be
    # positive semi-definite (the particle absorbs and scatters, it does not emit), e^{-iωt}
    loss = (α - α.conj().T) / 2j
    eigenvalues = np.linalg.eigvalsh(0.5 * (loss + loss.conj().T))
    if eigenvalues[0] < -rtol * np.max(np.abs(α)):
        raise ValueError(
            f'polarisability{where} is not passive: Im α has a negative eigenvalue {eigenvalues[0]:.3e} '
            '(levisim uses e^{-iωt}, so absorbing particles have Im α ≥ 0)'
        )


def _check_damping(name, Γ, rtol):
    eigenvalues = np.linalg.eigvals(Γ)
    scale = np.max(np.abs(Γ))
    if np.max(np.abs(eigenvalues.imag)) > rtol * scale or eigenvalues.real.min() < -rtol * scale:
        raise ValueError(f'{name} must have real, non-negative eigenvalues (a friction tensor), not {eigenvalues}')


class TabulatedParticle(Particle):
    """
    A levitated particle defined by tensors computed externally (T-matrix, DDA, FEM, ...)

    ::

        TabulatedParticle(mass, inertia, polarisability,
                          translational_damping=None, rotational_damping=None, damping_gas=None,
                          damping_accommodation=1.0, damping_surface_temperature_ratio=1.0,
                          volume=None, area=None, material=None, name='tabulated particle',
                          charge=0, λ=1064 nm, Φ=π/2, orientation=(1, 0, 0, 0), rtol=1e-8)

    - mass: m_n [kg]
    - inertia: body-frame inertia tensor about the centre of mass [kg·m²] (3×3, or the 3 principal
      moments); symmetric, positive definite, obeying I1 + I2 ≥ I3
    - polarisability: body-frame polarisability [Fm²] (without radiation reaction, which levisim
      adds), one of

      - a 3×3 tensor (or 3-vector diagonal, or scalar), used at every wavelength
      - a callable α(λ) returning one, λ the vacuum wavelength [m]
      - a table {'λ': (n,) increasing wavelengths [m], 'α': (n, 3, 3) tensors}, interpolated
        linearly (real and imaginary parts) between the tabulated wavelengths; outside the table
        it raises

      Each tensor must be symmetric (reciprocity) and passive (Im α positive semi-definite).
    - translational_damping, rotational_damping: body-frame gas damping-rate tensors [2π·Hz] (the
      Γ_cm and Γ_rot of Martinetz et al. (2018)), either 3×3 arrays valid in `damping_gas` (a `Gas`
      or {'P': .., 'T': .., 'm': ..}), scaled to any other gas by the free-molecular law
      Γ ∝ P √(m_g/T) at the same accommodation, or callables f(gas, accommodation,
      surface_temperature_ratio) → 3×3. Arrays are valid only for the accommodation coefficient and
      surface-temperature ratio they were computed with (`damping_accommodation`,
      `damping_surface_temperature_ratio`); asking for others raises.
    - volume, area: optional [m³], [m²] (needed for ρ, `Trap.F_gas`, ...)
    - material: optional `Material` (for information; the permittivity is not used, since the
      polarisability is given)
    - rtol: relative tolerance of the symmetry and definiteness checks

    `from_particle` tabulates any other particle; `to_dict` gives the keyword arguments back (a
    round trip).
    """
    def __init__(
        self,
        mass,
        inertia,
        polarisability,
        translational_damping=None,
        rotational_damping=None,
        damping_gas=None,
        damping_accommodation=1.0,
        damping_surface_temperature_ratio=1.0,
        volume=None,
        area=None,
        material=None,
        name='tabulated particle',
        charge=0,
        λ=1064*nm,
        Φ=_NODE,
        orientation=(1.0, 0.0, 0.0, 0.0),
        rtol=1e-8,
    ):
        """
        Initialises a TabulatedParticle instance
        """
        self.rtol = float(rtol)
        self.name = str(name)

        mass = float(mass)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(f'mass must be positive and finite, not {mass}')
        self._mass = mass

        inertia = _matrix('inertia', inertia)
        _check_inertia(inertia, self.rtol)
        self._inertia = inertia

        self._volume = None if volume is None else float(volume)
        self._area = None if area is None else float(area)
        for label, value in (('volume', self._volume), ('area', self._area)):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f'{label} must be positive and finite, not {value}')

        if material is None:
            ρ = mass / self._volume if self._volume is not None else 1.0
            material = Material(name=self.name, ρ=ρ, notes='placeholder for a TabulatedParticle')
            self._placeholder_material = True
        else:
            self._placeholder_material = False
        super().__init__(material, charge=charge, λ=λ, Φ=Φ, orientation=orientation)

        self._set_polarisability(polarisability)

        if damping_gas is not None and not isinstance(damping_gas, Gas):
            damping_gas = Gas(**dict(damping_gas))
        self.damping_gas = damping_gas
        self.damping_accommodation = float(damping_accommodation)
        self.damping_surface_temperature_ratio = float(damping_surface_temperature_ratio)
        self._translational_damping = self._set_damping('translational_damping', translational_damping)
        self._rotational_damping = self._set_damping('rotational_damping', rotational_damping)

    # ----------------------------------------------------------------------------------------------
    # input handling
    # ----------------------------------------------------------------------------------------------
    def _set_polarisability(self, polarisability):
        self._polarisability_table = None
        self._polarisability_callable = None
        self._polarisability_constant = None
        if isinstance(polarisability, Mapping):
            λs = np.asarray(polarisability['λ'], dtype=float)
            αs = np.asarray(polarisability['α'])
            if λs.ndim != 1 or λs.size < 1 or αs.shape != (λs.size, 3, 3):
                raise ValueError(
                    f"a polarisability table needs 'λ' of shape (n,) and 'α' of shape (n, 3, 3), "
                    f'not {λs.shape} and {αs.shape}'
                )
            if np.any(np.diff(λs) <= 0) or np.any(λs <= 0):
                raise ValueError("the table's wavelengths 'λ' must be positive and strictly increasing")
            αs = αs.astype(complex if np.iscomplexobj(αs) else float)
            for i, α in enumerate(αs):
                _matrix('polarisability', α, allow_complex=True)
                _check_polarisability(α, self.rtol, where=f' at λ = {λs[i]:.6g} m')
            self._polarisability_table = (λs, αs)
        elif callable(polarisability):
            self._polarisability_callable = polarisability
            # check at the particle's own wavelength now
            self.static_polarisability_tensor()
        else:
            α = _matrix('polarisability', polarisability, allow_complex=True)
            _check_polarisability(α, self.rtol)
            self._polarisability_constant = α

    def _set_damping(self, name, value):
        if value is None:
            return None
        if callable(value):
            return value
        Γ = _matrix(name, value)
        _check_damping(name, Γ, self.rtol)
        if self.damping_gas is None:
            raise ValueError(f'{name} given as an array needs damping_gas, the gas it was computed for')
        return Γ

    # ----------------------------------------------------------------------------------------------
    # shape
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def m(self):
        """
        Mass
        [kg]
        """
        return self._mass

    @property
    def ρ(self):
        """
        Mean density m_n/V (needs the volume)
        [kg/m³]
        """
        if self._volume is None:
            raise ValueError('this TabulatedParticle has no volume, so no density')
        return self._mass / self._volume

    @cached_property
    def V(self):
        """
        Volume (if given)
        [m³]
        """
        if self._volume is None:
            raise ValueError('this TabulatedParticle was given no volume')
        return self._volume

    @cached_property
    def A(self):
        """
        Surface area (if given)
        [m²]
        """
        if self._area is None:
            raise ValueError('this TabulatedParticle was given no surface area')
        return self._area

    @cached_property
    def inertia_tensor(self):
        """
        Body-frame inertia tensor about the centre of mass
        [kg·m²]
        """
        return self._inertia

    @property
    def wavelength_range(self):
        """(λ_min, λ_max) of a tabulated polarisability, None otherwise [m]"""
        if self._polarisability_table is None:
            return None
        λs = self._polarisability_table[0]
        return (float(λs[0]), float(λs[-1]))

    def static_polarisability_tensor(self, λ=None):
        """
        Body-frame polarisability (without radiation reaction) at vacuum wavelength λ (default: the
        particle's λ), from the constant, the callable or the table (linear interpolation)
        [Fm²]
        """
        λ = self.λ if λ is None else float(λ)
        if self._polarisability_constant is not None:
            return self._polarisability_constant
        if self._polarisability_callable is not None:
            α = _matrix('polarisability', self._polarisability_callable(λ), allow_complex=True)
            _check_polarisability(α, self.rtol, where=f' at λ = {λ:.6g} m')
            return α
        λs, αs = self._polarisability_table
        if not λs[0] * (1 - 1e-12) <= λ <= λs[-1] * (1 + 1e-12):
            raise ValueError(f'λ = {λ:.6g} m is outside the tabulated range {λs[0]:.6g}-{λs[-1]:.6g} m')
        exact = np.flatnonzero(np.isclose(λs, λ, rtol=1e-12, atol=0))
        if exact.size:
            return αs[exact[0]]
        flat = αs.reshape(len(λs), 9)
        real = np.array([np.interp(λ, λs, flat[:, i].real) for i in range(9)])
        if np.iscomplexobj(αs):
            imag = np.array([np.interp(λ, λs, flat[:, i].imag) for i in range(9)])
            return (real + 1j*imag).reshape(3, 3)
        return real.reshape(3, 3)

    # ----------------------------------------------------------------------------------------------
    # gas damping
    # ----------------------------------------------------------------------------------------------
    def _damping(self, stored, kind, gas, accommodation, surface_temperature_ratio):
        if stored is None:
            raise NotImplementedError(f'this TabulatedParticle was given no {kind} damping tensor')
        if callable(stored):
            return _matrix(f'{kind} damping', stored(gas, accommodation, surface_temperature_ratio))
        if (accommodation, surface_temperature_ratio) != (self.damping_accommodation, self.damping_surface_temperature_ratio):
            raise ValueError(
                f'the tabulated {kind} damping holds for αc = {self.damping_accommodation}, '
                f'γs = {self.damping_surface_temperature_ratio}; pass a callable to vary them'
            )
        reference = self.damping_gas
        scale = (gas.P / reference.P) * np.sqrt((gas.m / reference.m) * (reference.T / gas.T))
        return stored * scale

    def translational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Centre-of-mass gas damping-rate tensor in the body frame: the tabulated tensor scaled from
        `damping_gas` by P √(m_g/T), or the callable's value
        [2π·Hz]
        """
        return self._damping(self._translational_damping, 'translational', gas, accommodation, surface_temperature_ratio)

    def rotational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Rotational gas damping-rate tensor in the body frame: the tabulated tensor scaled from
        `damping_gas` by P √(m_g/T), or the callable's value
        [2π·Hz]
        """
        return self._damping(self._rotational_damping, 'rotational', gas, accommodation, surface_temperature_ratio)

    # ----------------------------------------------------------------------------------------------
    # construction from another particle, and back to keyword arguments
    # ----------------------------------------------------------------------------------------------
    @classmethod
    def from_particle(cls, particle, wavelengths=None, gas=None, accommodation=1.0, surface_temperature_ratio=1.0,
                      **kwargs):
        """
        Tabulates another particle: its mass, inertia, body-frame static polarisability (at its own λ,
        or a table at `wavelengths` [m]), and, if a `gas` is given, its damping tensors in that gas.
        Charge, λ, Φ, orientation, material, volume and area are copied; kwargs override.
        """
        if wavelengths is None:
            polarisability = np.array(particle.static_polarisability_tensor(), copy=True)
        else:
            λs = np.asarray(wavelengths, dtype=float)
            polarisability = {'λ': λs, 'α': np.array([particle.static_polarisability_tensor(λ) for λ in λs])}
        arguments = {
            'mass': particle.m,
            'inertia': np.array(particle.inertia_tensor, copy=True),
            'polarisability': polarisability,
            'material': particle.material,
            'name': f'tabulated {type(particle).__name__}',
            'charge': particle.charge,
            'λ': particle.λ,
            'Φ': particle.Φ,
            'orientation': np.array(particle.orientation, copy=True),
        }
        for key, attribute in (('volume', 'V'), ('area', 'A')):
            try:
                arguments[key] = getattr(particle, attribute)
            except (NotImplementedError, ValueError, AttributeError):
                pass
        if gas is not None:
            arguments.update(
                translational_damping=particle.translational_damping_tensor(gas, accommodation, surface_temperature_ratio),
                rotational_damping=particle.rotational_damping_tensor(gas, accommodation, surface_temperature_ratio),
                damping_gas={'P': gas.P, 'T': gas.T, 'm': gas.m},
                damping_accommodation=accommodation,
                damping_surface_temperature_ratio=surface_temperature_ratio,
            )
        arguments.update(kwargs)
        return cls(**arguments)

    def to_dict(self):
        """
        Keyword arguments that rebuild this particle, ``TabulatedParticle(**p.to_dict())`` (callables are
        passed through as they are; arrays are copies)
        """
        if self._polarisability_constant is not None:
            polarisability = self._polarisability_constant.copy()
        elif self._polarisability_callable is not None:
            polarisability = self._polarisability_callable
        else:
            λs, αs = self._polarisability_table
            polarisability = {'λ': λs.copy(), 'α': αs.copy()}

        def damping(stored):
            return stored if stored is None or callable(stored) else stored.copy()

        gas = self.damping_gas
        return {
            'mass': self._mass,
            'inertia': self._inertia.copy(),
            'polarisability': polarisability,
            'translational_damping': damping(self._translational_damping),
            'rotational_damping': damping(self._rotational_damping),
            'damping_gas': None if gas is None else {'P': gas.P, 'T': gas.T, 'm': gas.m},
            'damping_accommodation': self.damping_accommodation,
            'damping_surface_temperature_ratio': self.damping_surface_temperature_ratio,
            'volume': self._volume,
            'area': self._area,
            'material': None if self._placeholder_material else self.material,
            'name': self.name,
            'charge': self.charge,
            'λ': self.λ,
            'Φ': self.Φ,
            'orientation': self.orientation.copy(),
            'rtol': self.rtol,
        }

    @property
    def info_str(self):
        """
        String describing the TabulatedParticle instance
        """
        α = np.real_if_close(np.diag(self.static_polarisability_tensor()))
        return (
            f'TabulatedParticle(id={id(self)}'
            f'\n    name = {self.name}'
            f'\n    charge = {self.charge} e'
            f'\n    λ = {self.λ / nm:.1f} nm'
            f'\n    orientation (w, x, y, z) = {np.array2string(self.orientation, precision=5)}'
            f'\n'
            f'\n    mass = {self.m / fg:.5f} fg'
            f'\n    inertia (body, diagonal) = {np.array2string(np.diag(self.inertia_tensor), precision=5)} kg·m²'
            f'\n    α (body, diagonal) = {α} Fm²'
            f'\n'
            f'\n    trapping position phase, Φ/2π = x0/λtw = {self.Φ /(2*π):.3f}'
            f'\n)'
        )
