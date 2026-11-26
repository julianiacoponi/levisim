"""
The bulk material a levitated particle is made of (Material)

A `Material` holds the properties of a substance that do not depend on the particle's shape or
size: mass density ρ, relative permittivity εr (constant, dispersive, complex or a body-frame
tensor), magnetic volume susceptibility χ_m (scalar or body-frame tensor), remanent magnetisation
and electrical conductivity. A `Particle` (see `levisim.physics.particle`) combines a material with
a shape. A small library of materials, each number with its source, is in
`levisim.physics.materials`.
"""

from collections.abc import Mapping
from numbers import Number

import numpy as np

from levisim.core.units import μm


def SiO2_refractive_index_Malitson(λ):
    """
    Refractive index of fused silica (SiO2) at vacuum wavelength λ, from the Sellmeier formula of
    I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965), doi:10.1364/JOSA.55.001205, valid for
    0.21-3.71 μm (20 °C)::

        n² - 1 = Σi Bi λ² / (λ² - Ci²), λ in μm,
        B = (0.6961663, 0.4079426, 0.8974794), C = (0.0684043, 0.1162414, 9.896161) μm
        e.g. n(1064 nm) = 1.44963

    [·]
    """
    λ_μm = np.asarray(λ, dtype=float) / μm
    if np.any((λ_μm < 0.21) | (λ_μm > 3.71)):
        raise ValueError(f'Malitson formula is valid for 0.21-3.71 μm, got λ = {λ_μm} μm')
    B = (0.6961663, 0.4079426, 0.8974794)
    C = (0.0684043, 0.1162414, 9.896161)
    λ2 = λ_μm**2
    n2 = 1 + sum(Bi * λ2 / (λ2 - Ci**2) for Bi, Ci in zip(B, C))
    return float(np.sqrt(n2)) if np.ndim(n2) == 0 else np.sqrt(n2)


def SiO2_permittivity_Malitson(λ):
    """
    Relative permittivity εr = n(λ)² of fused silica from the Malitson (1965) Sellmeier formula,
    see `SiO2_refractive_index_Malitson`
    [·]
    """
    return SiO2_refractive_index_Malitson(λ)**2


# named dispersion models accepted as `Material(permittivity='<name>')` (case-insensitive)
PERMITTIVITY_MODELS = {
    'malitson': SiO2_permittivity_Malitson,
}

# the kinds of permittivity a Material can hold
_KINDS = ('none', 'constant', 'tensor', 'model', 'tabulated')


def _as_permittivity_value(value):
    """A scalar (float, or complex if it has an imaginary part) or a 3×3 complex/real ndarray"""
    if isinstance(value, Number) and not isinstance(value, bool):
        if isinstance(value, complex) or np.iscomplexobj(value):
            return complex(value)
        return float(value)
    array = np.asarray(value)
    if array.shape == (3, 3):
        return array.astype(complex if np.iscomplexobj(array) else float)
    if array.ndim == 0:
        return _as_permittivity_value(array.item())
    raise ValueError(f'a permittivity must be a scalar or a 3×3 body-frame tensor, got shape {array.shape}')


def _as_tensor(value):
    """A scalar or 3×3 value as a 3×3 array"""
    if np.ndim(value) == 0:
        return value * np.eye(3)
    return np.asarray(value)


def _is_isotropic(tensor):
    """Whether a 3×3 tensor is a multiple of the identity"""
    return bool(np.all(tensor == tensor[0, 0] * np.eye(3)))


class Material:
    """
    A bulk material: density, permittivity, magnetic susceptibility, magnetisation, conductivity

    ::

        Material(name, ρ, permittivity=None, χ_m=0.0, magnetisation=None, conductivity=None,
                 references=None, notes='')

    - name: a label, e.g. 'fused silica (Malitson 1965)'
    - ρ: mass density [kg/m³]
    - permittivity: relative permittivity εr [·], one of
        - None: no optical data (εr raises)
        - a real number, e.g. 1.98
        - a complex number ε' + iε'' with ε'' > 0 for an absorbing material (e^{-iωt} convention)
        - a 3×3 body-frame tensor (real or complex), e.g. diag(no², no², ne²) for a uniaxial crystal
          with its optic axis along the body z axis
        - a callable ε(λ) of the vacuum wavelength λ [m], returning any of the above
        - the name of a dispersion model in `PERMITTIVITY_MODELS`, e.g. 'Malitson'
        - a mapping {λ: εr} of values at specific vacuum wavelengths (no interpolation: asking for
          another wavelength raises)
    - χ_m: magnetic volume susceptibility (SI, dimensionless) [·], scalar or 3×3 body-frame tensor;
      negative for a diamagnet
    - magnetisation: remanent magnetisation M_r [A/m], a magnitude or a body-frame 3-vector;
      None for a material without one
    - conductivity: electrical conductivity σ [S/m]; None if not specified
    - references: {quantity: citation} with the source of each number
    - notes: free text on caveats, e.g. where a value depends on the sample

    A Material is immutable once built, because library materials are shared between particles:
    use `replace(...)` to derive a modified copy.
    """
    def __init__(
        self,
        name,
        ρ,
        permittivity=None,
        χ_m=0.0,
        magnetisation=None,
        conductivity=None,
        references=None,
        notes='',
    ):
        """
        Initialises a Material instance
        """
        self._init_kwargs = {
            'name': name, 'ρ': ρ, 'permittivity': permittivity, 'χ_m': χ_m, 'magnetisation': magnetisation,
            'conductivity': conductivity, 'references': references, 'notes': notes,
        }
        self.name = str(name)

        # Mass density
        # [kg/m³]
        self.ρ = float(ρ)
        if not np.isfinite(self.ρ) or self.ρ <= 0:
            raise ValueError(f'ρ must be a positive, finite density in kg/m³, not {ρ}')

        # Relative permittivity: see the class docstring for the accepted forms
        # [·]
        self.permittivity = permittivity
        if permittivity is None:
            self.permittivity_kind = 'none'
            self._εr = None
        elif isinstance(permittivity, str):
            key = permittivity.lower()
            if key not in PERMITTIVITY_MODELS:
                raise ValueError(
                    f'Unknown permittivity model {permittivity!r}; known models: {sorted(PERMITTIVITY_MODELS)}'
                )
            self.permittivity_kind = 'model'
            self._εr = PERMITTIVITY_MODELS[key]
        elif callable(permittivity):
            self.permittivity_kind = 'model'
            self._εr = permittivity
        elif isinstance(permittivity, Mapping):
            if not permittivity:
                raise ValueError('a tabulated permittivity needs at least one {λ: εr} entry')
            self.permittivity_kind = 'tabulated'
            self._εr = {float(λ): _as_permittivity_value(ε) for λ, ε in permittivity.items()}
        else:
            value = _as_permittivity_value(permittivity)
            self.permittivity_kind = 'constant' if np.ndim(value) == 0 else 'tensor'
            self._εr = value

        # Magnetic volume susceptibility (SI)
        # [·]
        χ = np.asarray(χ_m)
        if χ.shape == (3, 3):
            self.χ_m = χ.astype(float)
        elif χ.ndim == 0:
            self.χ_m = float(χ)
        else:
            raise ValueError(f'χ_m must be a scalar or a 3×3 body-frame tensor, got shape {χ.shape}')

        # Remanent magnetisation
        # [A/m]
        if magnetisation is None:
            self.magnetisation = None
        else:
            M = np.asarray(magnetisation, dtype=float)
            if M.shape not in ((), (3,)):
                raise ValueError(f'magnetisation must be a magnitude or a 3-vector, got shape {M.shape}')
            self.magnetisation = float(M) if M.ndim == 0 else M

        # Electrical conductivity
        # [S/m]
        self.conductivity = None if conductivity is None else float(conductivity)

        self.references = dict(references or {})
        self.notes = str(notes)
        self._frozen = True

    def __setattr__(self, key, value):
        if getattr(self, '_frozen', False):
            raise AttributeError(
                f'Material is immutable (it may be shared between particles); use material.replace({key}=...)'
            )
        super().__setattr__(key, value)

    def replace(self, **changes):
        """
        A copy of this material with some constructor arguments changed, e.g.
        `FUSED_SILICA.replace(name='porous silica', ρ=1850.0)`
        """
        unknown = set(changes) - set(self._init_kwargs)
        if unknown:
            raise TypeError(f'Material.replace() got unknown arguments {sorted(unknown)}')
        return Material(**{**self._init_kwargs, **changes})

    def __repr__(self):
        return f'Material({self.name!r}, ρ={self.ρ!r}, permittivity={self.permittivity!r})'

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info(self):
        """
        Prints information on the Material instance
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        String describing the Material instance
        """
        lines = [
            f'Material({self.name!r}',
            f'    ρ = {self.ρ:.1f} kg/m³',
            f'    permittivity ({self.permittivity_kind}) = {self.permittivity!r}',
            f'    χ_m = {self.χ_m!r}',
            f'    magnetisation = {self.magnetisation!r} A/m',
            f'    conductivity = {self.conductivity!r} S/m',
        ]
        lines += [f'    source of {quantity}: {source}' for quantity, source in self.references.items()]
        if self.notes:
            lines.append(f'    notes: {self.notes}')
        return '\n'.join(lines) + '\n)'

    ################################################################################################
    # OPTICAL PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @property
    def has_permittivity(self):
        """Whether the material has optical data (a permittivity) [bool]"""
        return self.permittivity_kind != 'none'

    @property
    def is_dispersive(self):
        """Whether εr depends on the wavelength (a model or a table) [bool]"""
        return self.permittivity_kind in ('model', 'tabulated')

    @property
    def tabulated_wavelengths(self):
        """
        Vacuum wavelengths at which a tabulated permittivity is given (empty for other kinds)
        [m]
        """
        return tuple(sorted(self._εr)) if self.permittivity_kind == 'tabulated' else ()

    def εr(self, λ=None):
        """
        Relative permittivity at vacuum wavelength λ: a scalar (float, or complex if absorbing) or,
        for an anisotropic material, the 3×3 body-frame tensor. λ may be omitted for a constant.
        [·]
        """
        kind = self.permittivity_kind
        if kind == 'none':
            raise ValueError(f'Material {self.name!r} has no permittivity (no optical data)')
        if kind in ('constant', 'tensor'):
            return self._εr
        if λ is None:
            raise ValueError(f'Material {self.name!r} has a dispersive permittivity: pass the vacuum wavelength λ')
        if kind == 'model':
            return _as_permittivity_value(self._εr(λ))
        for λ_table, value in self._εr.items():
            if np.isclose(λ_table, λ, rtol=1e-9, atol=0.0):
                return value
        available = ', '.join(f'{λ_table / μm * 1e3:.1f} nm' for λ_table in self.tabulated_wavelengths)
        raise ValueError(
            f'Material {self.name!r} has tabulated permittivity only at {available}, not at λ = {λ / μm * 1e3:.3f} nm'
        )

    def εr_tensor(self, λ=None):
        """
        Relative permittivity at vacuum wavelength λ as a 3×3 body-frame tensor
        [·]
        """
        return _as_tensor(self.εr(λ))

    def refractive_index(self, λ=None):
        """
        Refractive index n = √εr (μr = 1) at vacuum wavelength λ; complex (n + iκ) if absorbing.
        For an anisotropic material, the three principal indices: √ of the diagonal for a tensor that
        is diagonal in the body frame, else √ of its eigenvalues.
        [·]
        """
        ε = self.εr(λ)
        if np.ndim(ε) == 0:
            return np.sqrt(ε)
        if np.all(ε == np.diag(np.diag(ε))):
            return np.sqrt(np.diag(ε))
        if np.allclose(ε, ε.conj().T):
            return np.sqrt(np.linalg.eigvalsh(ε).astype(complex if np.iscomplexobj(ε) else float))
        return np.sqrt(np.linalg.eigvals(ε))

    # short alias
    n = refractive_index

    def is_absorbing(self, λ=None):
        """Whether εr has a positive imaginary part (absorption) at λ [bool]"""
        return bool(np.any(np.imag(self.εr(λ)) > 0))

    def is_anisotropic(self, λ=None):
        """Whether εr is a tensor that is not a multiple of the identity at λ [bool]"""
        ε = self.εr(λ)
        return np.ndim(ε) == 2 and not _is_isotropic(ε)

    ################################################################################################
    # MAGNETIC PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @property
    def χ_m_tensor(self):
        """
        Magnetic volume susceptibility as a 3×3 body-frame tensor
        [·]
        """
        return _as_tensor(self.χ_m)

    @property
    def is_magnetic(self):
        """Whether the material responds to a magnetic field (χ_m ≠ 0) or is magnetised [bool]"""
        return bool(np.any(self.χ_m_tensor != 0)) or self.magnetisation is not None

    @property
    def is_diamagnetic(self):
        """Whether χ_m is negative along every principal axis and there is no remanence [bool]"""
        eigenvalues = np.linalg.eigvalsh(self.χ_m_tensor)
        return bool(np.all(eigenvalues < 0)) and self.magnetisation is None

    @property
    def is_magnetically_anisotropic(self):
        """Whether χ_m is a tensor that is not a multiple of the identity [bool]"""
        return not _is_isotropic(self.χ_m_tensor)

    @property
    def is_conductor(self):
        """Whether a conductivity is given and positive [bool]"""
        return self.conductivity is not None and self.conductivity > 0
