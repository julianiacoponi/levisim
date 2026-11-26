"""
The bulk material a levitated particle is made of (Material)

A `Material` holds the properties of a substance that do not depend on the particle's shape or
size: mass density ρ, relative permittivity εr (constant, dispersive, complex or a body-frame
tensor), magnetic volume susceptibility χ_m (scalar or body-frame tensor), remanent magnetisation
and electrical conductivity. A `Particle` (see `levisim.physics.particle`) combines a material with
a shape. A small library of materials, each number with its source, is in
`levisim.physics.materials`.
"""

import os
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


class RefractiveIndexTable:
    """
    A tabulated complex refractive index n + iκ against the vacuum wavelength, interpolated linearly

    ::

        RefractiveIndexTable(λ, n, κ=None, source='', name='')

    - λ: strictly increasing vacuum wavelengths [m]
    - n, κ: refractive index and extinction coefficient at each λ [·] (κ = 0 if omitted)
    - source: citation of the data; name: a label

    Calling the table, ``table(λ)``, returns the relative permittivity εr = (n + iκ)² at λ (a float if
    κ = 0 there, else complex, e^{-iωt} convention). n and κ are interpolated separately and linearly
    in λ between the two neighbouring rows, v1 + (λ - λ1)(v2 - v1)/(λ2 - λ1) with λ in μm, and a
    tabulated wavelength returns its row exactly. Outside [λ_min, λ_max] it raises: there is no
    extrapolation. A `Material` built with a table has `permittivity_kind == 'nk table'`.
    """
    def __init__(self, λ, n, κ=None, source='', name='', λ_μm=None):
        """
        Initialises a RefractiveIndexTable instance (`λ_μm`: the same wavelengths as tabulated in μm,
        used for the interpolation so that the source's numbers enter unchanged; default λ/μm)
        """
        λ = np.array(λ, dtype=float)
        n = np.array(n, dtype=float)
        κ = np.zeros_like(n) if κ is None else np.array(κ, dtype=float)
        if λ.ndim != 1 or λ.size < 2 or n.shape != λ.shape or κ.shape != λ.shape:
            raise ValueError('λ, n and κ must be 1D arrays of the same length (at least 2 rows)')
        if not (np.all(np.isfinite(λ)) and np.all(np.isfinite(n)) and np.all(np.isfinite(κ))):
            raise ValueError('λ, n and κ must be finite')
        if np.any(np.diff(λ) <= 0) or λ[0] <= 0:
            raise ValueError('the wavelengths of a RefractiveIndexTable must be positive and strictly increasing')
        if np.any(κ < 0):
            raise ValueError('κ must be ≥ 0 (absorption, e^{-iωt} convention)')
        for array in (λ, n, κ):
            array.setflags(write=False)
        # rows are stored with λ in μm, as tabulated, so that interpolation uses the source's numbers
        self._λ_μm = λ / μm if λ_μm is None else np.array(λ_μm, dtype=float)
        if self._λ_μm.shape != λ.shape or not np.allclose(self._λ_μm * μm, λ, rtol=1e-12, atol=0.0):
            raise ValueError('λ_μm must be the wavelengths λ in μm')
        self._λ_μm.setflags(write=False)
        self.λ, self.n_values, self.κ_values = λ, n, κ
        self.source = str(source)
        self.name = str(name)

    @classmethod
    def from_csv(cls, path, source=None, name=None):
        """
        A table from a text file of rows 'λ [μm], n, κ' (comma separated; lines starting with '#' are
        comments). Without `source`, the comment lines are the source.
        """
        comments, rows = [], []
        with open(path, encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    comments.append(line.lstrip('#').strip())
                    continue
                rows.append([float(value) for value in line.split(',')])
        data = np.array(rows, dtype=float)
        return cls(
            data[:, 0] * μm, data[:, 1], data[:, 2] if data.shape[1] > 2 else None,
            source=' '.join(comments) if source is None else source, λ_μm=data[:, 0],
            name=os.path.splitext(os.path.basename(str(path)))[0] if name is None else name,
        )

    @property
    def wavelength_range(self):
        """(λ_min, λ_max) of the table [m]"""
        return (float(self.λ[0]), float(self.λ[-1]))

    def nk(self, λ):
        """
        (n, κ) at vacuum wavelength λ [m], interpolated linearly in λ
        [·]
        """
        x = float(λ) / μm
        grid = self._λ_μm
        if not grid[0] * (1 - 1e-12) <= x <= grid[-1] * (1 + 1e-12):
            raise ValueError(
                f'λ = {x * 1e3:.3f} nm is outside the table {self.name!r} '
                f'({grid[0] * 1e3:.1f}-{grid[-1] * 1e3:.1f} nm); there is no extrapolation'
            )
        exact = np.flatnonzero(np.isclose(grid, x, rtol=1e-12, atol=0.0))
        if exact.size:
            i = int(exact[0])
            return float(self.n_values[i]), float(self.κ_values[i])
        i = int(np.clip(np.searchsorted(grid, x) - 1, 0, grid.size - 2))
        λ1, λ2 = float(grid[i]), float(grid[i + 1])

        def interpolate(values):
            v1, v2 = float(values[i]), float(values[i + 1])
            return v1 + (x - λ1) * (v2 - v1) / (λ2 - λ1)

        return interpolate(self.n_values), interpolate(self.κ_values)

    def refractive_index(self, λ):
        """n + iκ at vacuum wavelength λ [m] (a float if κ = 0) [·]"""
        n, κ = self.nk(λ)
        return n if κ == 0 else complex(n + 1j*κ)

    def __call__(self, λ):
        """
        Relative permittivity εr = (n + iκ)² at vacuum wavelength λ [m]; a float if κ = 0
        [·]
        """
        n, κ = self.nk(λ)
        return n**2 if κ == 0 else (n + 1j*κ)**2

    def __len__(self):
        return int(self.λ.size)

    def __repr__(self):
        λ_min, λ_max = self.wavelength_range
        return f'RefractiveIndexTable({self.name!r}, {len(self)} rows, {λ_min / μm:.4g}-{λ_max / μm:.4g} μm)'


def optical_data_path(filename):
    """Path of a data file shipped in levisim/physics/optical_data"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optical_data', filename)


def load_refractive_index_table(filename, **kwargs):
    """
    A `RefractiveIndexTable` from a file shipped in levisim/physics/optical_data (e.g.
    'gold_olmon2012_evaporated.csv'); each file's header comments give its source
    """
    return RefractiveIndexTable.from_csv(optical_data_path(filename), **kwargs)


# named dispersion models accepted as `Material(permittivity='<name>')` (case-insensitive)
PERMITTIVITY_MODELS = {
    'malitson': SiO2_permittivity_Malitson,
}

# the kinds of permittivity a Material can hold
_KINDS = ('none', 'constant', 'tensor', 'model', 'tabulated', 'nk table')


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
        - a `RefractiveIndexTable` of n, κ against λ, interpolated linearly at any wavelength inside
          the table (e.g. the library's SILICON, DIAMOND and GOLD)
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
        elif isinstance(permittivity, RefractiveIndexTable):
            self.permittivity_kind = 'nk table'
            self._εr = permittivity
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
        return self.permittivity_kind in ('model', 'tabulated', 'nk table')

    @property
    def wavelength_range(self):
        """
        (λ_min, λ_max) over which an interpolated n, κ table (`RefractiveIndexTable`) holds; None for
        other kinds of permittivity
        [m]
        """
        return self._εr.wavelength_range if self.permittivity_kind == 'nk table' else None

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
        if kind in ('model', 'nk table'):
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
