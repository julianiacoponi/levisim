"""
Combined physics of a levitated particle held by a list of force elements (Trap)
"""

from functools import cached_property

import numpy as np
from scipy.optimize import brentq, fsolve

from levisim.analysis.spectral_density_theory import SpectralDensityTheory
from levisim.core.constants import kB, ħ, ε0, π
from levisim.core.units import (
    Hz,
    degrees,
    eV,
    kHz,
    ms,
    zN,
    μm,
)
from levisim.physics.bead import Bead
from levisim.physics.fields.base import ForceElement, as_position
from levisim.physics.fields.elements import Feedback, PhotonRecoil
from levisim.physics.gas import Gas
from levisim.physics.kick import Kick
from levisim.physics.particle import Particle, Sphere
from levisim.physics.tweezer import Tweezer


class MissingOpticalFieldError(ValueError):
    """Raised when an optical-only quantity is asked of a Trap whose fields hold no optical element"""


class FieldList(list):
    """
    The list of a Trap's force elements: a plain list that counts its modifications (`version`), so
    that the Trap can cache how its elements are grouped and rebuild that only after a change
    """

    version = 0

    def _modified(self):
        self.version += 1


def _counting(name):
    method = getattr(list, name)

    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._modified()
        return result
    wrapper.__name__ = name
    wrapper.__doc__ = method.__doc__
    return wrapper


for _name in ('append', 'extend', 'insert', 'remove', 'pop', 'clear', 'sort', 'reverse',
              '__setitem__', '__delitem__', '__iadd__', '__imul__'):
    setattr(FieldList, _name, _counting(_name))
del _name


class Trap:
    """
    Combined physics of a levitated particle in a trap made of force elements, i.e. of
    - a particle (a Bead by default), held by
    - a list of force elements (`fields`: a Tweezer by default, plus feedback damping, ...), floating in
    - a Gas instance

    Two ways to build one::

        Trap(tweezer=Tweezer(...), bead=Bead(...), Γ_fbx=..., photon_recoil=..., kicks=[...])
        Trap(fields=[Tweezer(...), HarmonicPotential(...), Feedback(...)], particle=Sphere(...))

    Either form accepts ``extra_fields=[...]``, appended to the elements (e.g. a `PaulTrap` beside
    the shorthand's tweezer, which is how a hybrid Paul-optical trap enters `MultiTrap` through its
    ``trap_kwargs``).

    The first is shorthand for ``fields=[tweezer, Feedback(Γ_fbx, Γ_fby, Γ_fbz)]`` (plus a
    `PhotonRecoil` element if photon_recoil=True and the kicks), with the historical defaults
    Γ_fbx, Γ_fby = 2.1, 2.5 kHz. With ``fields=`` given, the trap holds exactly those elements:
    feedback damping is zero unless a `Feedback` element is among them (or Γ_fbx, ... are given, which
    then add one), and ``tweezer=`` may not be given as well.

    The total stiffness tensor is the sum of the elements' stiffness tensors at the
    `linearisation_point` (the origin, i.e. the tweezer focus, by default; a position, or 'equilibrium'
    for `find_equilibrium` started from ``equilibrium_guess``); its diagonal gives kx, ky,
    kz and the frequencies Ωj = √(kj/m), and its off-diagonal part the `intra_trap_couplings`.
    Optical-only quantities (tweezer, α_eff, U_depth, photon_recoil_heating_rates, σ_scattering,
    z_eq, Utw, Ftw, ...) use the first optical element and raise a `MissingOpticalFieldError` (a
    ValueError) for a trap without one.

    `position` [m] places the trap (its origin, the tweezer focus) in the laboratory frame, default
    (0, 0, 0); only interactions between particles (`levisim.physics.interactions`) use it.

    This class also allows for:
    - Arbitrary values of feedback damping in x, y, z
    - Stochastic, directed force noise in x-y plane
    - Kicks at specified times (step-like impulses) in the x-y plane (to be used in Simulation, or deduced from Experiment)
    - Photon recoil (turned on or off)
    - Overriding the Tweezer potential's natural frequencies
    """
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialises a Trap instance
        """
        ############################################################################################
        # DEFINING INSTANCES
        # ------------------------------------------------------------------------------------------
        # the levitated particle; `particle=` is accepted as an alias of `bead=`
        if 'bead' in kwargs and 'particle' in kwargs:
            raise ValueError('Pass either bead or particle, not both')
        self.bead = kwargs['bead'] if 'bead' in kwargs else kwargs.get('particle', Bead())
        self.gas = kwargs.get('gas', Gas())

        # ------------------------------------------------------------------------------------------
        # The force elements acting on the particle (see levisim.physics.fields)
        feedback_keys = [key for key in ('Γ_fbx', 'Γ_fby', 'Γ_fbz') if key in kwargs]
        if 'fields' in kwargs:
            if 'tweezer' in kwargs:
                raise ValueError('Pass the tweezer inside fields=[...], not as tweezer= as well')
            self.fields = list(kwargs['fields'])
            for element in self.fields:
                if not isinstance(element, ForceElement):
                    raise TypeError(f'Trap fields must be ForceElement instances, not {type(element).__name__}')
            if feedback_keys:
                if any(isinstance(element, Feedback) for element in self.fields):
                    raise ValueError(f'{", ".join(feedback_keys)} given as well as a Feedback element in fields')
                # Feedback mechanical damping rates of (x, y, z)-motion oscillations [2π·Hz]
                self.fields.append(Feedback(*(kwargs.get(key, 0.0) for key in ('Γ_fbx', 'Γ_fby', 'Γ_fbz'))))
        else:
            # shorthand: one tweezer plus feedback cold damping with the historical defaults
            # Feedback mechanical damping rates of (x, y, z)-motion oscillations [2π·Hz]
            self.fields = [
                kwargs.get('tweezer', Tweezer()),
                Feedback(
                    Γ_fbx=kwargs.get('Γ_fbx', 2.1*kHz),
                    Γ_fby=kwargs.get('Γ_fby', 2.5*kHz),
                    Γ_fbz=kwargs.get('Γ_fbz', 0.0*kHz),
                ),
            ]
        # Whether or not to include the photon recoil (scattering) force along the beam axis in
        # Simulation, see F_scattering (Tania doesn't use it) [bool]: adds a PhotonRecoil element
        if bool(kwargs.get('photon_recoil', False)):
            self.photon_recoil = True
        # Kick() instances for simulating discrete impacts
        # generate this with kicks.generate_kick_list()
        self.fields.extend(kwargs.get('kicks', []))
        # further force elements appended to either form, e.g. a PaulTrap next to the shorthand's
        # tweezer (Trap(tweezer=..., extra_fields=[PaulTrap(...)]), as MultiTrap's trap_kwargs can pass)
        extra_fields = list(kwargs.get('extra_fields', []))
        for element in extra_fields:
            if not isinstance(element, ForceElement):
                raise TypeError(f'Trap extra_fields must be ForceElement instances, not {type(element).__name__}')
        self.fields.extend(extra_fields)

        # ------------------------------------------------------------------------------------------
        # Point about which the stiffness tensor (and so kj, Ωj) is evaluated: the origin (the
        # tweezer focus) by default, as levisim always has; give a position, or 'equilibrium' for
        # the root of the static force (find_equilibrium, started from equilibrium_guess if given)
        # [m]
        linearisation_point = kwargs.get('linearisation_point', None)
        if isinstance(linearisation_point, str):
            if linearisation_point != 'equilibrium':
                raise ValueError(f"linearisation_point must be a position or 'equilibrium', not {linearisation_point!r}")
            # equilibrium_guess: where find_equilibrium starts (default the origin)
            linearisation_point = self.find_equilibrium(r_guess=kwargs.get('equilibrium_guess'))
        self.linearisation_point = (
            np.zeros(3) if linearisation_point is None else as_position(linearisation_point)
        )

        # ------------------------------------------------------------------------------------------
        # Position of the trap (the origin of its coordinates x, y, z, i.e. the tweezer focus) in the
        # laboratory frame, whose axes are parallel to the trap's. Only interactions between
        # particles use it (levisim.physics.interactions): a particle at displacement r in this trap
        # is at position + r in the laboratory. The phase Φ of the bead in a cavity standing wave
        # stays `bead.Φ` (MultiTrap checks the two agree)
        # [m]
        self.position = as_position(kwargs.get('position', (0.0, 0.0, 0.0)))

        ############################################################################################
        # DEFINED PROPERTIES

        # ------------------------------------------------------------------------------------------
        # Scale factor for directed force noise compared to the thermal force noise
        # [·]
        # fdir = β.fth
        self.β = float(kwargs.get('β', np.sqrt(2)))

        # ------------------------------------------------------------------------------------------
        # Angle of the directed force noise in the x-y plane
        # [rad]
        self.ψ = float(kwargs.get('ψ', 45*degrees))

        # ------------------------------------------------------------------------------------------
        # Override resonant frequencies (that ignore Tweezer properties)
        # useful to get SpectralDensityTheory curves to match Experiment or Simulation data
        self.override_Ω_dict = kwargs.get('override_Ω_dict', None)
        # Opt-in finite-size alignment torque of an extended particle in the anisotropic trap
        # (levisim.physics.rotation.finite_size_potential): off by default, so the point-dipole
        # torques are unchanged
        self.finite_size_torque = bool(kwargs.get('finite_size_torque', False))

        # ------------------------------------------------------------------------------------------
        # Cavity drive: the light this bead coherently scatters into a cavity (used by MultiTrap)
        # By default it is the trapping tweezer's own light at its focus. A bead can instead be
        # trapped by one beam and scatter another into the cavity, e.g. a bead held by a 976 nm
        # tweezer that the cavity does not resonate with, while it scatters the out-of-focus field
        # of a 1064 nm tweezer that the cavity does (Pontin et al., PRL 135, 073602 (2025)):
        #   - cavity_drive_tweezer: the `Tweezer` whose light the bead scatters into the cavity; its
        #     wavelength, polarisation angle θ and focal field E0 enter the coherent-scattering
        #     couplings in place of the trapping tweezer's (default: the trapping tweezer)
        #   - cavity_drive_field_ratio: amplitude of that light at the bead relative to its focal
        #     field E0, e.g. ζ < 1 for a bead away from the focus (default 1)
        # The phase of the scattered light stays the trapping tweezer's φ (MultiTrap's φn_dict).
        self.cavity_drive_tweezer = kwargs.get('cavity_drive_tweezer', None)
        self.cavity_drive_field_ratio = float(kwargs.get('cavity_drive_field_ratio', 1.0))
        if not np.isfinite(self.cavity_drive_field_ratio) or self.cavity_drive_field_ratio < 0:
            raise ValueError(
                f'cavity_drive_field_ratio must be a finite, non-negative amplitude ratio, not {self.cavity_drive_field_ratio}'
                ' (put a phase into the tweezer phase φ instead)'
            )

        # to calculate theoretical spectral density curves
        self.theory = SpectralDensityTheory(
            self.bead.m,
            self.Ωx, self.Ωy, self.Ωz,
            # NOTE: the theory spectra need the *total* damping Γj = Γth + Γfb,j (earlier versions
            # passed Γfb,j alone, which left Szz undamped for the default Γfb,z = 0)
            self.Γx, self.Γy, self.Γz,
            self.fth, self.β, self.ψ,
        )

        ############################################################################################
        # MULTI-TRAP PROPERTIES
        # ------------------------------------------------------------------------------------------
        self.trap_index = kwargs.get('trap_index')

    @property
    def particle(self):
        """
        The levitated particle (the same object as `bead`)
        """
        return self.bead

    @particle.setter
    def particle(self, value):
        self.bead = value

    ################################################################################################
    # FORCE ELEMENTS
    # ----------------------------------------------------------------------------------------------
    @property
    def fields(self):
        """
        The force elements acting on the particle, in order (a list; see levisim.physics.fields)
        """
        return self._fields

    @fields.setter
    def fields(self, value):
        self._fields = FieldList(value)

    @property
    def _element_plan(self):
        """
        The elements of `fields` sorted by role (optical element, simulation groups, kicks, recoil),
        recomputed only when the list of elements changes, so the simulation loop does not rescan it
        """
        fields = self._fields
        cached = self.__dict__.get('_element_plan_cache')
        if cached is not None and cached[0] is fields and cached[1] == fields.version:
            return cached[2]
        groups = {}
        for element in self.fields:
            groups.setdefault(element.simulation_group, []).append(element)
        optical = [element for element in self.fields if element.is_optical]
        field_group = groups.get('field', [])
        plan = {
            # the cache holds the elements themselves, so their ids cannot be reused while cached
            'elements': tuple(self.fields),
            'groups': groups,
            'tweezer': optical[0] if optical else None,
            'single_tweezer': len(field_group) == 1 and bool(optical) and field_group[0] is optical[0],
            'kicks': [element for element in self.fields if isinstance(element, Kick)],
            'recoils': [element for element in self.fields if isinstance(element, PhotonRecoil)],
        }
        self.__dict__['_element_plan_cache'] = (fields, fields.version, plan)
        return plan

    def fields_of_type(self, element_type):
        """The elements of `fields` that are instances of element_type, in order"""
        return [element for element in self.fields if isinstance(element, element_type)]

    def fields_in_group(self, group):
        """The elements of `fields` whose simulation_group is group, in order"""
        return list(self._element_plan['groups'].get(group, ()))

    @property
    def has_optical_field(self):
        """Whether any element of `fields` is optical (is_optical)"""
        return self._element_plan['tweezer'] is not None

    @property
    def tweezer(self):
        """
        The trap's optical element: the first element of `fields` with is_optical (the Tweezer)

        Raises MissingOpticalFieldError (a ValueError) if the trap has no optical element.
        """
        tweezer = self._element_plan['tweezer']
        if tweezer is not None:
            return tweezer
        names = ', '.join(element.label for element in self.fields) or 'none'
        raise MissingOpticalFieldError(
            f'This Trap has no optical field (its fields: {names}). Optical-only quantities (tweezer, '
            'α_eff, U_depth, photon_recoil_heating_rates, σ_scattering, z_eq, Utw, Ftw, F_scattering, '
            'and the cavity drive unless cavity_drive_tweezer is given) need a Tweezer in fields.'
        )

    @tweezer.setter
    def tweezer(self, value):
        # replace the optical element in place (or put one first if there is none)
        for index, element in enumerate(self.fields):
            if element.is_optical:
                for recoil in self.fields_of_type(PhotonRecoil):
                    if recoil.tweezer is element:
                        recoil.tweezer = value
                self.fields[index] = value
                return
        self.fields.insert(0, value)

    def _feedback_rate(self, attribute):
        rates = [getattr(element, attribute) for element in self.fields_of_type(Feedback)]
        if len(rates) == 1:
            return rates[0]
        return float(sum(rates))

    def _set_feedback_rate(self, attribute, value):
        feedbacks = self.fields_of_type(Feedback)
        if len(feedbacks) > 1:
            raise ValueError(f'This Trap has {len(feedbacks)} Feedback elements: set {attribute} on one of them')
        if not feedbacks:
            feedbacks = [Feedback()]
            self.fields.append(feedbacks[0])
        setattr(feedbacks[0], attribute, float(value))

    @property
    def Γ_fbx(self):
        """
        Feedback mechanical damping rate of x-motion oscillations (sum over the Feedback elements)
        [2π·Hz]
        """
        return self._feedback_rate('Γ_fbx')

    @Γ_fbx.setter
    def Γ_fbx(self, value):
        self._set_feedback_rate('Γ_fbx', value)

    @property
    def Γ_fby(self):
        """
        Feedback mechanical damping rate of y-motion oscillations (sum over the Feedback elements)
        [2π·Hz]
        """
        return self._feedback_rate('Γ_fby')

    @Γ_fby.setter
    def Γ_fby(self, value):
        self._set_feedback_rate('Γ_fby', value)

    @property
    def Γ_fbz(self):
        """
        Feedback mechanical damping rate of z-motion oscillations (sum over the Feedback elements)
        [2π·Hz]
        """
        return self._feedback_rate('Γ_fbz')

    @Γ_fbz.setter
    def Γ_fbz(self, value):
        self._set_feedback_rate('Γ_fbz', value)

    @property
    def kicks(self):
        """The Kick elements of `fields` (step-like impulses), in order"""
        return list(self._element_plan['kicks'])

    @kicks.setter
    def kicks(self, value):
        self.fields = [element for element in self.fields if not isinstance(element, Kick)] + list(value)

    @property
    def photon_recoil(self):
        """
        Whether the scattering (photon recoil) force acts in Simulation, i.e. whether `fields` holds
        a PhotonRecoil element; setting it adds one for the trap's tweezer, or removes them
        [bool]
        """
        return bool(self._element_plan['recoils'])

    @photon_recoil.setter
    def photon_recoil(self, value):
        if value and not self.photon_recoil:
            self.fields.append(PhotonRecoil(self.tweezer))
        elif not value:
            self.fields = [element for element in self.fields if not isinstance(element, PhotonRecoil)]

    def __setstate__(self, state):
        """
        Restores a pickled Trap; a Trap pickled before the force elements (with tweezer, Γ_fbx, ...,
        photon_recoil and kicks as attributes) is converted to the equivalent fields
        """
        state = dict(state)
        if 'fields' in state:
            state['_fields'] = state.pop('fields')
        if '_fields' not in state:
            fields = [
                state.pop('tweezer'),
                Feedback(state.pop('Γ_fbx', 0.0), state.pop('Γ_fby', 0.0), state.pop('Γ_fbz', 0.0)),
            ]
            if state.pop('photon_recoil', False):
                fields.append(PhotonRecoil(fields[0]))
            fields.extend(state.pop('kicks', []))
            state['_fields'] = fields
            state.setdefault('linearisation_point', np.zeros(3))
        state['_fields'] = FieldList(state['_fields'])
        # Trap.position (laboratory frame) is newer than some pickles
        state.setdefault('position', np.zeros(3))
        state.pop('_element_plan_cache', None)
        self.__dict__.update(state)

    def __getstate__(self):
        """The instance state without the (derived) element-plan cache"""
        return {key: value for key, value in self.__dict__.items() if key != '_element_plan_cache'}

    @property
    def info(self):
        """
        Prints information on Trap instance
        """
        print(self.info_str)

    @property
    def info_str(self):
        """A multi-line summary of the trap's parameters"""
        return (
            f'Trap(id={id(self)}'
            f'\n    trap_index={self.trap_index}'
            f'\n'
            f'\n    Γfb,x = {self.Γ_fbx / kHz:.6f} kHz'
            f'\n    Γfb,y = {self.Γ_fby / kHz:.6f} kHz'
            f'\n    Γfb,z = {self.Γ_fbz / kHz:.6f} kHz'
            f'\n    β = {self.β:.5f}'
            f'\n    ψ = {self.ψ / degrees:.5f}°'
            f'\n'
            f'\n    Γth = {self.Γth / Hz:.6e} Hz'
            f'\n    Γth ~ 15Pr²/(m〈v〉) = {self.Γth_2 / Hz:.6e} Hz'
            f'\n    Γth ~ 16/(πρr〈v〉) = {self.Γth_3 / Hz:.6e} Hz'
            f'\n    fth = {self.fth / zN:.5f} zN/√Hz'
            f'\n    Sth = {self.Sth:.5e} N²/Hz'
            f'\n'
            f'\n    τx = {self.τx / ms:.6f} ms'
            f'\n    τy = {self.τy / ms:.6f} ms'
            f'\n    τz = {self.τz / ms:.6f} ms'
            f'\n'
            f'\n    A = 2 kBT / m = {self.A:.5e} m²Hz²'
            f'\n'
            f'\n    fdir = {self.fdir / zN:.5f} zN/√Hz'
            f'\n    fdir,x = {self.fdirx / zN:.5f} zN/√Hz'
            f'\n    fdir,y = {self.fdiry / zN:.5f} zN/√Hz'
            f'\n    fdir,z = {self.fdirz / zN:.5f} zN/√Hz'
            f'\n'
            f'\n    Ωx = {self.Ωx / kHz:.5f} kHz'
            f'\n    Ωy = {self.Ωy / kHz:.5f} kHz'
            f'\n    Ωz = {self.Ωz / kHz:.5f} kHz'
            f'\n'
            f'\n    Qx = {self.Qx:.5e}'
            f'\n    Qy = {self.Qy:.5e}'
            f'\n    Qz = {self.Qz:.5e}'
            f'\n'
            f'\n    kx = {self.kx:.5e} N/m'
            f'\n    ky = {self.ky:.5e} N/m'
            f'\n    kz = {self.kz:.5e} N/m'
            f'\n'
            f'{self._optical_info_str}'
            f'\n    F_gas = {self.F_gas:.5e} N'
            f'\n    mean_free_path = {self.mean_free_path:.5e} m'
            f'\n    collision_rate = {self.collision_rate:.5e} Hz'
            f'\n    collision_time = {self.collision_time:.5e} s'
            f'\n'
            f'\n    x_SQL = {self.x_SQL:.5e} m'
            f'\n    y_SQL = {self.y_SQL:.5e} m'
            f'\n    z_SQL = {self.z_SQL:.5e} m'
            f'\n'
            f'\n    Fx_SQL = {self.Fx_SQL:.5e} N'
            f'\n    Fy_SQL = {self.Fy_SQL:.5e} N'
            f'\n    Fz_SQL = {self.Fz_SQL:.5e} N'
            f'\n'
            f'{self._fields_info_str}'
            f'\n    id(bead) = {id(self.bead)}'
            f'\n    id(gas) = {id(self.gas)}'
            f'\n)'
        )

    @property
    def _optical_info_str(self):
        if not self.has_optical_field:
            return ''
        return (
            f'\n    α_eff = {self.α_eff:.5e} Fm²'
            f'\n    σ_scattering = {self.σ_scattering:.5e} m²'
            f'\n    z_eq = {self.z_eq/ μm:.5f} μm'
            f'\n'
            f'\n    U_depth = {self.U_depth / eV:.5f} eV'
            f'\n    U_depth = {self.U_depth / kB:.5e} K'
            f'\n    v_escape = {self.v_escape:.5f} m/s'
            f'\n    p_escape = {self.p_escape:.5e} kg·m/s'
            f'\n'
        )

    @property
    def _fields_info_str(self):
        # the historical summary of a shorthand trap (tweezer + feedback, recoil, kicks) lists only
        # id(tweezer); any other element is listed by name
        lines = ''
        if self.has_optical_field:
            lines += f'\n    id(tweezer) = {id(self.tweezer)}'
        others = [
            element for element in self.fields
            if not (element.is_optical or isinstance(element, (Feedback, PhotonRecoil, Kick)))
        ]
        if others or not self.has_optical_field:
            lines += f'\n    fields = [{", ".join(element.label for element in self.fields)}]'
        return lines

    ################################################################################################
    # CAVITY DRIVE (coherent scattering, used by MultiTrap)
    # ----------------------------------------------------------------------------------------------
    @property
    def drive_tweezer(self):
        """
        The `Tweezer` whose light the bead coherently scatters into a cavity: `cavity_drive_tweezer`
        if given, otherwise the trapping tweezer
        """
        return self.tweezer if self.cavity_drive_tweezer is None else self.cavity_drive_tweezer

    @property
    def cavity_drive_field(self):
        """
        Amplitude of the light the bead scatters into a cavity, at the bead:
        `cavity_drive_field_ratio` × `drive_tweezer.E0` (the trapping tweezer's E0 by default)
        [V/m]
        """
        return self.cavity_drive_field_ratio * self.drive_tweezer.E0

    ################################################################################################
    # DERIVED PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @property
    def _uses_sphere_gas_damping(self):
        """
        Whether the gas damping is the original scalar formula (a Sphere/Bead, or any particle-like
        object that is not a levisim `Particle`), rather than the particle's damping tensor
        """
        return isinstance(self.bead, Sphere) or not isinstance(self.bead, Particle)

    @cached_property
    def Γth_tensor(self):
        """
        Thermal gas damping-rate tensor of the centre-of-mass motion in the lab frame, for full
        diffuse accommodation with the surface at the gas temperature (αc = γs = 1, as `Γth`): Γth·𝟙
        for a sphere, R Γ_cm Rᵀ of the particle's current orientation otherwise
        (`Particle.lab_translational_damping_tensor`)
        [2π·Hz]
        """
        if self._uses_sphere_gas_damping:
            return self.Γth * np.eye(3)
        return np.asarray(self.bead.lab_translational_damping_tensor(self.gas), dtype=float)

    @cached_property
    def Γth_j(self):
        """
        Dict of the thermal gas damping rates of the (x, y, z) motion: Γth on every axis for a
        sphere, the diagonal of `Γth_tensor` otherwise. The off-diagonal elements of a tilted
        non-spherical particle's tensor (gas friction coupling the axes) are left out of the per-axis
        damping used by `MultiTrap` and `FreeSpaceSimulation`.
        [2π·Hz]
        """
        if self._uses_sphere_gas_damping:
            return {'x': self.Γth, 'y': self.Γth, 'z': self.Γth}
        diagonal = np.diag(self.Γth_tensor)
        return {axis: float(diagonal[i]) for i, axis in enumerate('xyz')}

    @cached_property
    def fth_j(self):
        """
        Dict of the thermal force noises of the (x, y, z) motion, √(2 m kB T Γth,j) (all equal to
        fth for a sphere)
        [N/√Hz]
        """
        return {axis: np.sqrt(2 * self.bead.m * kB * self.gas.T * Γ) for axis, Γ in self.Γth_j.items()}

    @cached_property
    def Sth_j(self):
        """
        Dict of the thermal force spectral densities of the (x, y, z) motion, fth_j² (all equal to
        Sth for a sphere)
        [N²/Hz]
        """
        return {axis: f**2 for axis, f in self.fth_j.items()}

    @cached_property
    def Γth(self):
        """
        Thermal gas mechanical damping rate (for a non-spherical particle: the mean of the diagonal of
        `Γth_tensor`, i.e. its trace/3; the per-axis rates are `Γth_j`)
        [2π·Hz]
        """
        if not self._uses_sphere_gas_damping:
            tensor = self.Γth_tensor
            if np.all(tensor == tensor[0, 0] * np.eye(3)):
                return float(tensor[0, 0])
            return float(np.trace(tensor) / 3)
        # NOTE: many publications in our field use varying different expressions...
        # my choice is derived from: http://dx.doi.org/10.1016/j.physleta.2010.06.041
        # 2010-06-19 -- Cavalleri et al. -- Gas damping force noise on a macroscopic test body in an infinite gas reservoir
        # simplified / referenced in Pontin 2020 "Ultranarrow-linewidth" paper
        # doi: 10.1103/PhysRevResearch.2.023349
        geometric_factor = (1 + π/8)
        # equivalent to 4πr²/3m, i.e. 1/3 surface area over mass
        bead_factor = 1 / (self.bead.ρ * self.bead.r)
        gas_factor = (self.gas.m * self.gas.vth * self.gas.P) / (kB * self.gas.T)
        return geometric_factor * bead_factor * gas_factor

    @cached_property
    def Γth_2(self):
        """
        (Alternative formula for)
        Thermal gas mechanical damping rate
        [2π·Hz]
        """
        return 15 * self.gas.P * self.bead.r**2 / (self.bead.m * self.gas.vth)

    @cached_property
    def Γth_3(self):
        """
        (Alternative formula for)
        Thermal gas mechanical damping rate
        [2π·Hz]
        """
        return 16 * self.gas.P / (π * self.bead.ρ * self.bead.r * self.gas.vth)


    @cached_property
    def fth(self):
        """
        Thermal force noise due to gas damping
        [N/√Hz]
        a.k.a. thermal noise floor
        uses bead mass, but gas temperature and damping rate
        NOTE: https://en.wikipedia.org/wiki/Langevin_equation
        """
        # NOTE: one-sided spectra effectively have a force noise √2 larger
        # since the *spectral* floor is doubled
        # so, often written as
        # fth = np.sqrt(4 * m * kB * T * Γ)
        return np.sqrt(2 * self.bead.m * kB * self.gas.T * self.Γth)

    @cached_property
    def Sth(self):
        """
        Thermal force spectral density due to gas damping
        [N²/Hz]
        """
        return self.fth**2

    @cached_property
    def fdir(self):
        """
        Directed force noise
        [N/√Hz]
        """
        return self.β * self.fth

    @cached_property
    def Sdir(self):
        """
        Directed force spectral density
        [N²/Hz]
        """
        return self.fdir**2

    @cached_property
    def fdirx(self):
        """
        Directed force noise in the x-axis
        [N/√Hz]
        """
        return self.fdir * np.cos(self.ψ)

    @cached_property
    def fdiry(self):
        """
        Directed force noise in the y-axis
        [N/√Hz]
        """
        return self.fdir * np.sin(self.ψ)

    @cached_property
    def fdirz(self):
        """
        Directed force noise in the z-axis
        [N/√Hz]
        """
        # not modelling any directed force in z
        return 0

    @cached_property
    def fdirj(self):
        """
        Dict of directed force noises in the (x, y, z)-axes
        [N/√Hz]
        """
        return {
            'x': self.fdirx,
            'y': self.fdiry,
            'z': self.fdirz,
        }

    @cached_property
    def Sdirx(self):
        """
        Directed force spectral density for x-axis motion
        [N²/Hz]
        """
        return self.fdirx**2

    @cached_property
    def Sdiry(self):
        """
        Directed force spectral density for y-axis motion
        [N²/Hz]
        """
        return self.fdiry**2

    @cached_property
    def Sdirz(self):
        """
        Directed force spectral density for z-axis motion
        [N²/Hz]
        """
        return self.fdirz**2

    @cached_property
    def Sdirj(self):
        """
        Dict of directed force spectral density for (x, y, z)-axis motion
        [N²/Hz]
        """
        return {
            'x': self.Sdirx,
            'y': self.Sdiry,
            'z': self.Sdirz,
        }

    @cached_property
    def Sψψ(self):
        """
        Angular force spectral density
        [N²/Hz]
        """
        return self.Sdir * np.cos(self.ψ)**2 * np.sin(self.ψ)**2

    # ----------------------------------------------------------------------------------------------
    # OTHER PER AXIS PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @property
    def _other_damping_rates(self):
        """
        Summed damping_rates of the 'damping' elements other than Feedback (e.g. EddyCurrentDamping),
        or None if there are none (so that the damping of every other trap is computed as before)
        [2π·Hz]
        """
        others = [
            element for element in self.fields_in_group('damping') if not isinstance(element, Feedback)
        ]
        if not others:
            return None
        return sum(np.asarray(element.damping_rates(self.bead), dtype=float) for element in others)

    @cached_property
    def Γx(self):
        """
        Total mechanical damping rate of (x, y, z)-motion oscillations
        [2π·Hz]
        """
        Γ = self.Γth_j['x'] + self.Γ_fbx
        extra = self._other_damping_rates
        return Γ if extra is None else Γ + extra[0]

    @cached_property
    def Γy(self):
        """
        Total mechanical damping rate of y-motion oscillations
        [2π·Hz]
        """
        Γ = self.Γth_j['y'] + self.Γ_fby
        extra = self._other_damping_rates
        return Γ if extra is None else Γ + extra[1]

    @cached_property
    def Γz(self):
        """
        Total mechanical damping rate of z-motion oscillations
        [2π·Hz]
        """
        Γ = self.Γth_j['z'] + self.Γ_fbz
        extra = self._other_damping_rates
        return Γ if extra is None else Γ + extra[2]

    @cached_property
    def Γj(self):
        """
        Dict of total mechanical damping rates for (x, y, z)-motion oscillations
        [2π·Hz]
        """
        return {
            'x': self.Γx,
            'y': self.Γy,
            'z': self.Γz,
        }

    @cached_property
    def τx(self):
        """
        Autocorrelation / thermalisation time for x-motion oscillations
        [s]
        """
        return 1 / (self.Γx / Hz)

    @cached_property
    def τy(self):
        """
        Autocorrelation / thermalisation time for y-motion oscillations
        [s]
        """
        return 1 / (self.Γy / Hz)

    @cached_property
    def τz(self):
        """
        Autocorrelation / thermalisation time for z-motion oscillations
        [s]
        """
        return 1 / (self.Γz / Hz)

    @cached_property
    def τj(self):
        """
        Dict of Autocorrelation / thermalisation times for (x, y, z)-motion oscillations
        [s]
        """
        return {
            'x': self.τx,
            'y': self.τy,
            'z': self.τz,
        }

    @cached_property
    def stiffness_contributions(self):
        """
        List of (element, stiffness tensor) for every element of `fields`, each tensor
        K_ij = -∂F_i/∂r_j evaluated at the linearisation_point
        [N/m]
        """
        return [(element, element.stiffness(self.linearisation_point, self.bead)) for element in self.fields]

    @cached_property
    def stiffness_tensor(self):
        """
        Total stiffness tensor of the trap at the linearisation_point, the sum of the elements'
        stiffness tensors (in the order of `fields`)
        [N/m]

        For the default tweezer trap it is diag(kx, ky, kz) of the tweezer at its focus. Its
        off-diagonal part (from fields whose principal axes differ) is `intra_trap_coupling_stiffness`.
        """
        total = None
        for _, K in self.stiffness_contributions:
            total = K if total is None else total + K
        return np.zeros((3, 3)) if total is None else total

    @cached_property
    def intra_trap_coupling_stiffness(self):
        """
        Off-diagonal part of `stiffness_tensor` (zero diagonal), K_jk for j ≠ k
        [N/m]
        """
        K = np.array(self.stiffness_tensor, dtype=float)
        np.fill_diagonal(K, 0.0)
        return K

    @cached_property
    def intra_trap_coupling_matrix(self):
        """
        Direct mechanical couplings between the axes of this trap's particle, in levisim's g
        convention (MultiTrapBase.mechanical_coupling_matrix): entry [j, k] = gjk, the term
        -gjk·qj·qk of H/ħ with qj = bj + bj†, so that dbj/dt ∋ +i·gjk·qk
        [2π·Hz]

        A force F_j = -K_jk x_k with xj = xj_zpf·qj gives ::

            gjk = -K_jk · xj_zpf · xk_zpf / ħ

        with the zero-point fluctuations of the trap frequencies Ωj (`j_zpf`). Axes ordered (x, y, z);
        the diagonal is zero. For a conservative trap the matrix is symmetric.
        """
        zpf = np.array([self.x_zpf, self.y_zpf, self.z_zpf])
        # (0.0 - ... rather than unary minus, so that uncoupled entries are +0.0, not -0.0)
        return 0.0 - self.intra_trap_coupling_stiffness * np.outer(zpf, zpf) / ħ

    @cached_property
    def intra_trap_couplings(self):
        """
        Dict of the intra-trap mechanical couplings {'xy': gxy, 'yx': gyx, 'xz': ..., 'zx': ...,
        'yz': ..., 'zy': ...}, key jk for row j and column k of `intra_trap_coupling_matrix`, i.e. the
        rate at which axis k drives axis j. All zero for the default tweezer trap.
        [2π·Hz]
        """
        g = self.intra_trap_coupling_matrix
        axes = 'xyz'
        return {
            axes[j] + axes[k]: g[j, k]
            for j in range(3) for k in range(3) if j != k
        }

    @cached_property
    def field_noise_psd(self):
        """
        Sum of the force noises the elements add (e.g. the tweezer's photon-recoil force noise),
        as a two-sided 3×3 spectral density matrix; it excludes the gas (Sth) and the directed noise
        (Sdir), and is not used by FreeSpaceSimulation
        [N²/Hz]
        """
        total = np.zeros((3, 3))
        for element in self.fields:
            total = total + element.noise(self.bead)
        return total

    ################################################################################################
    # ROTATION: librations about the particle's orientation (levisim.physics.rotation)
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def static_torque(self):
        """
        Total torque of the elements on the particle in its current orientation at the
        linearisation point. The librational quantities below linearise about that orientation; a
        torque that is not zero means that it is not an equilibrium (it is logged by `MultiTrap`)
        [N·m]
        """
        total = np.zeros(3)
        for element in self.fields:
            total = total + np.asarray(element.torque(self.linearisation_point, 0.0, self.bead), dtype=float)
        if self.finite_size_torque:
            from levisim.physics.rotation import finite_size_torque
            total = total + finite_size_torque(*self.finite_size_tensors)
        return total

    @cached_property
    def finite_size_tensors(self):
        """
        The pair of tensors that the finite-size torque uses: the translational stiffness tensor
        (m·Ωj² from `override_Ω_dict` when it gives x, y and z, else `stiffness_tensor`) and the
        particle's lab-frame mass-weighted second moment ⟨r rᵀ⟩ = S/m
        (`levisim.physics.rotation.second_moment_tensor`)
        [N/m], [m²]
        """
        from levisim.physics.rotation import second_moment_tensor
        if self.override_Ω_dict and all(axis in self.override_Ω_dict for axis in 'xyz'):
            k = self.bead.m * np.diag([float(self.override_Ω_dict[axis])**2 for axis in 'xyz'])
        else:
            k = np.asarray(self.stiffness_tensor, dtype=float)
        S_body = second_moment_tensor(self.bead.inertia_tensor) / self.bead.m
        return k, np.asarray(self.bead.lab_tensor(S_body), dtype=float)

    @cached_property
    def torque_stiffness_tensor(self):
        """
        Total torque stiffness K_ij = -∂τ_i/∂θ_j of the elements for small rotations θ = (θα, θβ, θγ)
        about the lab axes x, y, z, at the particle's orientation and the linearisation point
        [N·m/rad]
        """
        total = np.zeros((3, 3))
        for element in self.fields:
            total = total + element.torque_stiffness(self.bead, None, self.linearisation_point)
        if self.finite_size_torque:
            from levisim.physics.rotation import finite_size_torque_stiffness
            total = total + finite_size_torque_stiffness(*self.finite_size_tensors)
        return total

    @cached_property
    def torque_noise_psd(self):
        """
        Sum of the elements' torque noises (the tweezer's recoil torque noise), two-sided, lab frame
        [N²·m²/Hz]
        """
        total = np.zeros((3, 3))
        for element in self.fields:
            total = total + element.torque_noise(self.bead, None, self.linearisation_point)
        return total

    @cached_property
    def Ij(self):
        """
        Dict of the moments of inertia about the lab axes, {'α': I_xx, 'β': I_yy, 'γ': I_zz} of the
        lab-frame inertia tensor in the particle's orientation
        [kg·m²]
        """
        diagonal = np.diag(np.asarray(self.bead.lab_inertia_tensor, dtype=float))
        return {axis: float(diagonal[i]) for i, axis in enumerate(('α', 'β', 'γ'))}

    @cached_property
    def Ωθ(self):
        """
        Dict of the librational frequencies about the lab axes, {'α': Ωα, 'β': Ωβ, 'γ': Ωγ} with
        Ωj = √(K_jj/I_jj) from `torque_stiffness_tensor` and `Ij` (zero where K_jj ≤ 0, i.e. a free
        or unstable rotation). `override_Ω_dict` entries 'α', 'β', 'γ' replace them.
        [2π·Hz]
        """
        K = self.torque_stiffness_tensor
        frequencies = {}
        for i, axis in enumerate(('α', 'β', 'γ')):
            if self.override_Ω_dict and axis in self.override_Ω_dict:
                frequencies[axis] = float(self.override_Ω_dict[axis])
                continue
            frequencies[axis] = float(np.sqrt(K[i, i] / self.Ij[axis])) if K[i, i] > 0 else 0.0
        return frequencies

    @cached_property
    def Γrot_tensor(self):
        """
        Rotational gas damping-rate tensor in the lab frame, R Γ_rot Rᵀ (Martinetz et al., Phys. Rev. E
        97, 052112 (2018), αc = γs = 1), so that dJ/dt = -Γ J for the angular momentum J
        [2π·Hz]
        """
        return np.asarray(self.bead.lab_rotational_damping_tensor(self.gas), dtype=float)

    @cached_property
    def Γrot_j(self):
        """
        Dict of the rotational gas damping rates about the lab axes, the diagonal of `Γrot_tensor`
        [2π·Hz]
        """
        diagonal = np.diag(self.Γrot_tensor)
        return {axis: float(diagonal[i]) for i, axis in enumerate(('α', 'β', 'γ'))}

    @cached_property
    def θ_zpf(self):
        """
        Dict of the librational zero-point fluctuations √(ħ/(2 I_j Ωj)) (inf for a free rotation)
        [rad]
        """
        return {
            axis: (np.sqrt(ħ / (2 * self.Ij[axis] * Ω)) if Ω > 0 else np.inf)
            for axis, Ω in self.Ωθ.items()
        }

    @cached_property
    def libration_recoil_heating_rates(self):
        """
        Recoil heating rates of the librations, {'α': Γ_rec,α, ...} = S_τ,jj/(2 I_j ħ Ωj) from the
        diagonal of `torque_noise_psd` (zero for a free rotation)
        [2π·Hz, phonons per second]
        """
        S = np.diag(self.torque_noise_psd)
        return {
            axis: (S[i] / (2 * self.Ij[axis] * ħ * self.Ωθ[axis]) if self.Ωθ[axis] > 0 else 0.0)
            for i, axis in enumerate(('α', 'β', 'γ'))
        }

    @cached_property
    def kx(self):
        """
        Spring constant of the restoring force in the x-axis, the xx element of the total stiffness
        tensor (for a tweezer trap, Re α_eff E0²/wx0²)
        [N/m]
        """
        return self.stiffness_tensor[0, 0]

    @cached_property
    def ky(self):
        """
        Spring constant of the restoring force in the y-axis, the yy element of the total stiffness
        tensor (for a tweezer trap, Re α_eff E0²/wy0²)
        [N/m]
        """
        return self.stiffness_tensor[1, 1]

    @cached_property
    def kz(self):
        """
        Spring constant of the restoring force in the z-axis, the zz element of the total stiffness
        tensor (for a tweezer trap, Re α_eff E0²/wz0²)
        [N/m]
        """
        return self.stiffness_tensor[2, 2]

    @cached_property
    def kj(self):
        """
        Dict of spring constants of the restoring forces in the (x, y, z)-axes (the diagonal of the
        total stiffness tensor)
        [N/m]
        """
        return {
            'x': self.kx,
            'y': self.ky,
            'z': self.kz,
        }

    # NOTE: calculated as first order harmonic frequencies from the Tweezer Hamiltonian
    # using classic Hooke's Law result Ω = √(k/m)
    # (ignores photon recoil)
    # allows for manually input frequencies
    @cached_property
    def Ωx(self):
        """
        Resonant frequencies for x-axis oscillation
        [2π·Hz]
        """
        if self.override_Ω_dict:
            return self.override_Ω_dict['x']
        return np.sqrt(self.kj['x'] / self.bead.m)

    @cached_property
    def Ωy(self):
        """
        Resonant frequencies for y-axis oscillation
        [2π·Hz]
        """
        if self.override_Ω_dict:
            return self.override_Ω_dict['y']
        return np.sqrt(self.kj['y'] / self.bead.m)

    @cached_property
    def Ωz(self):
        """
        Resonant frequencies for z-axis oscillation
        [2π·Hz]
        """
        if self.override_Ω_dict:
            return self.override_Ω_dict['z']
        return np.sqrt(self.kj['z'] / self.bead.m)

    @cached_property
    def Ωj(self):
        """
        Resonant frequencies for y-axis oscillation
        [2π·Hz]
        """
        return {
            'x': self.Ωx,
            'y': self.Ωy,
            'z': self.Ωz,
        }

    @cached_property
    def Qx(self):
        """
        Quality factor of the oscillation in the x-axis
        [·]
        """
        return self.Ωx / self.Γx

    @cached_property
    def Qy(self):
        """
        Quality factor of the oscillation in the y-axis
        [·]
        """
        return self.Ωy / self.Γy

    @cached_property
    def Qz(self):
        """
        Quality factor of the oscillation in the z-axis
        [·]
        """
        return self.Ωz / self.Γz


    @cached_property
    def Q(self):
        """
        Dict of quality factors of the oscillations in (x, y, z)-axes
        [·]
        """
        return {
            'x': self.Qx,
            'y': self.Qy,
            'z': self.Qz,
        }

    # ----------------------------------------------------------------------------------------------
    # STANDARD QUANTUM LIMITS
    # ----------------------------------------------------------------------------------------------
    # NOTE: answer to "find source for this": x_SQL = √(ħ/(mΩ)) = √2 x_zpf, where x_zpf = √(ħ/(2mΩ))
    # is the ground-state (zero-point) rms displacement (see j_zpf). The spectral standard quantum
    # limit of continuous displacement detection is the minimum *added* noise, imprecision plus
    # backaction, S_xx^SQL(ω) = ħ|χ(ω)|, i.e. ħ/(mΓΩ) on resonance (Clerk et al., Rev. Mod. Phys. 82,
    # 1155 (2010); Mason et al., Nat. Phys. 15, 745 (2019), arXiv:1809.10629). It is therefore a PSD,
    # not a displacement; x_SQL² = 2 x_zpf² is the variance of zero-point motion plus an equal amount
    # of added noise, which is the sense in which √(ħ/mΩ) is often quoted. Likewise Fj_SQL² =
    # 2ħmΓΩ = 2ħ/|χ(Ω)| is a one-sided force PSD on resonance, so its units are N/√Hz (not N).
    @cached_property
    def x_zpf(self):
        """
        Zero-point fluctuation (ground-state rms displacement) of the x-motion, √(ħ/(2mΩx))
        [m]
        """
        return np.sqrt(ħ / (2 * self.bead.m * self.Ωx))

    @cached_property
    def y_zpf(self):
        """
        Zero-point fluctuation (ground-state rms displacement) of the y-motion, √(ħ/(2mΩy))
        [m]
        """
        return np.sqrt(ħ / (2 * self.bead.m * self.Ωy))

    @cached_property
    def z_zpf(self):
        """
        Zero-point fluctuation (ground-state rms displacement) of the z-motion, √(ħ/(2mΩz))
        [m]
        """
        return np.sqrt(ħ / (2 * self.bead.m * self.Ωz))

    @cached_property
    def j_zpf(self):
        """
        Dict of zero-point fluctuations of the (x, y, z)-motions
        [m]
        """
        return {'x': self.x_zpf, 'y': self.y_zpf, 'z': self.z_zpf}

    @cached_property
    def x_SQL(self):
        """
        Standard Quantum Limit displacements for x-motions
        [m]
        """
        return np.sqrt(ħ / (self.bead.m * self.Ωx))

    @cached_property
    def y_SQL(self):
        """
        Standard Quantum Limit displacements for y-motions
        [m]
        """
        return np.sqrt(ħ / (self.bead.m * self.Ωy))

    @cached_property
    def z_SQL(self):
        """
        Standard Quantum Limit displacements for z-motions
        [m]
        """
        return np.sqrt(ħ / (self.bead.m * self.Ωz))

    @cached_property
    def j_SQL(self):
        """
        Standard Quantum Limit displacements for (x, y, z)-motions
        [m]
        """
        return {
            'x': self.x_SQL,
            'y': self.y_SQL,
            'z': self.z_SQL,
        }

    @cached_property
    def Fx_SQL(self):
        """
        Standard Quantum Limit forces in x-axis
        (one-sided force noise on resonance, see the NOTE above x_zpf)
        [N/√Hz]
        """
        return np.sqrt(2 * ħ * self.bead.m * self.Γx * self.Ωx)

    @cached_property
    def Fy_SQL(self):
        """
        Standard Quantum Limit forces in y-axis
        (one-sided force noise on resonance, see the NOTE above x_zpf)
        [N/√Hz]
        """
        return np.sqrt(2 * ħ * self.bead.m * self.Γy * self.Ωy)

    @cached_property
    def Fz_SQL(self):
        """
        Standard Quantum Limit forces in z-axis
        (one-sided force noise on resonance, see the NOTE above x_zpf)
        [N/√Hz]
        """
        return np.sqrt(2 * ħ * self.bead.m * self.Γz * self.Ωz)

    @cached_property
    def Fj_SQL(self):
        """
        Standard Quantum Limit forces in (x, y, z)-axes
        [N]
        """
        return {
            'x': self.Fx_SQL,
            'y': self.Fy_SQL,
            'z': self.Fz_SQL,
        }

    # ----------------------------------------------------------------------------------------------
    # PROPERTIES OF INTERACTIONS OF THE BEAD WITH THE THERMAL GAS
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def F_gas(self):
        """
        Gas pressure multiplied by the bead's surface area
        [N]
        """
        return self.gas.P * self.bead.A

    @cached_property
    def mean_free_path(self):
        """
        The mean free path for a gas collision into the bead
        [m]
        """
        return kB * self.gas.T / (np.sqrt(2) * self.F_gas)

    @cached_property
    def collision_rate(self):
        """
        Rate of gas collisions into the bead
        [Hz]
        """
        return self.gas.vth / self.mean_free_path

    @cached_property
    def collision_time(self):
        """
        Average time between gas collisions into the bead
        [s]
        """
        return 1 / self.collision_rate

    # ----------------------------------------------------------------------------------------------
    # OTHER PROPERTIES OF THE TWEEZER TRAP POTENTIAL
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def U_depth(self):
        """
        Energy depth of the tweezer potential well, ``|Utw(0, 0, 0)| = (Re α_eff / 4) E0²``, the
        cycle-averaged dipole energy at the focus (half the value of levisim <= 0.1, which used the
        peak energy); `v_escape` and `p_escape` follow from it
        [J]
        """
        # the tweezer potential is negative, so this is an absolute depth size
        return np.abs(self.Utw(0, 0, 0))

    @cached_property
    def v_escape(self):
        """
        Velocity the bead needs to escape the potential
        [m/s]
        """
        return np.sqrt(2 * self.U_depth / self.bead.m)

    @cached_property
    def p_escape(self):
        """
        Momentum the bead needs to escape the potential
        [kg·m/s]
        """
        return self.bead.m * self.v_escape

    @cached_property
    def max_Ftwx(self):
        """
        Maximum force due to the tweezer potential (its steepest slope) in the x-axis
        [N]
        """
        zR = self.tweezer.zR
        return max(np.array([self.Ftwx(x, 0, 0) for x in np.linspace(-zR, zR, 1000)]))

    @cached_property
    def max_Ftwy(self):
        """
        Maximum force due to the tweezer potential (its steepest slope) in the y-axis
        [N]
        """
        zR = self.tweezer.zR
        return max(np.array([self.Ftwy(0, y, 0) for y in np.linspace(-zR, zR, 1000)]))

    @cached_property
    def max_Ftwz(self):
        """
        Maximum force due to the tweezer potential (its steepest slope) in the z-axis
        [N]
        """
        zR = self.tweezer.zR
        return max(np.array([self.Ftwz(0, 0, z) for z in np.linspace(-zR, zR, 1000)]))

    @property
    def bead_has_scalar_polarisability(self):
        """
        Whether the particle's polarisability is one scalar (an isotropic sphere); otherwise α_eff
        is the projection on the tweezer polarisation of its lab-frame tensor
        [bool]
        """
        from levisim.physics.tweezer import _has_scalar_polarisability
        return _has_scalar_polarisability(self.bead)

    @cached_property
    def α_eff(self):
        """
        Effective polarisability of the bead that accounts for the radiation reaction force; for a
        non-spherical or birefringent particle, ê·α_eff·ê on the tweezer polarisation ê = ŷ in the
        particle's current orientation (`Tweezer.effective_polarisability`)
        [complex Fm²]
        """
        return self.tweezer.effective_polarisability(self.bead)

    @cached_property
    def photon_recoil_heating_rates(self):
        """
        Photon-recoil heating rates of the bead's centre-of-mass motion, {'x': Γ_rec,x, ...}
        [2π·Hz, phonons per second: d〈n_j〉/dt = Γ_rec,j]

        ::

            Γ_rec,j = π/(15ħε0) · (α·E0/2π)² · k⁵ · c_j · x_zpf,j²,   (c_x, c_y, c_z) = (2, 1, 7)

        i.e. (c_j/5)·R·k²·x_zpf,j² for a rate R = P_scat/ħω of scattered tweezer photons: c_j = 1
        along the polarisation (levisim's tweezer y axis), 2 perpendicular to it in the focal plane
        (x), and 7 along the beam (z), 2 from the dipole emission pattern plus 5 from the momentum
        of the absorbed tweezer photons. This is Eq. (4) of Piotrowski et al., Nat. Phys. 19, 1009
        (2023), whose polarisation is along their x. Eq. (B33) of Gonzalez-Ballestero et al.,
        Phys. Rev. A 100, 013805 (2019) has π/30 because its Γ(r) is the rate of the dissipator
        -Γ(r)·[q, [q, ρ]], which heats at 2Γ(r) phonons per second. α is Re(α_eff), as in the trap
        stiffness, and x_zpf uses the trap frequencies Ωj (override_Ω_dict if given).

        These are used by `MultiTrap(photon_recoil=True)` (the default with physical parameters); the `photon_recoil` flag of this class
        is separate and switches on the recoil force in `Simulation`.
        """
        if self.bead_has_scalar_polarisability and self.tweezer.has_default_polarisation:
            prefactor = π / (15 * ħ * ε0) * (np.real(self.α_eff) * self.tweezer.E0 / (2*π))**2 * self.tweezer.k**5
            coefficients = {'x': 2, 'y': 1, 'z': 7}
        else:
            # anisotropic particle: the induced dipole need not lie along the polarisation (see
            # Tweezer.recoil_coefficients)
            α, c = self.tweezer.recoil_coefficients(self.bead)
            prefactor = π / (15 * ħ * ε0) * (α * self.tweezer.E0 / (2*π))**2 * self.tweezer.k**5
            coefficients = dict(zip('xyz', c))
        return {
            axis: prefactor * coefficient * ħ / (2 * self.bead.m * self.Ωj[axis])
            for axis, coefficient in coefficients.items()
        }

    @cached_property
    def σ_scattering(self):
        """
        Scattering cross section for tweezer light hitting the bead
        [m²]
        """
        return self.tweezer.k * np.imag(self.α_eff) / ε0

    @cached_property
    def z_eq(self):
        """
        Equilibrium point between optical gradient and scattering forces, linearised in z/zR
        [m]

        Balancing F_scattering (on axis, ``∝ |E|² (k - 1/zR)`` at z = 0, the 1/zR being the Gouy-phase
        gradient) against the linearised gradient force -(Re α/2) E0² z / zR² gives
        z_eq = (Im α / Re α) k (zR² - zR/k).
        This agrees with Eq. (A4) of Toroš et al., arXiv:2012.15822, z0 = (εr-1)/(εr+2) (2k⁴zR²/3) R³,
        to leading order in the particle size (``Im α ≈ k³|α|²/6πε0``), apart from the factor (1 - 1/(k zR)),
        because Eq. (A2) uses the plane-wave scattering force without the Gouy phase. Since
        z_eq/zR is not tiny for 100 nm-scale beads (≈ 0.12 for the defaults), see z_eq_exact for the
        root of the full on-axis force.
        """
        γ0 = self.tweezer.zR**2 - self.tweezer.zR / self.tweezer.k
        return (np.imag(self.α_eff) / np.real(self.α_eff)) * self.tweezer.k * γ0

    # ----------------------------------------------------------------------------------------------
    # SCALE FACTORS FOR FITTING
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def A(self):
        """
        Scaling factor for the theoretical PSD fitting expression, *without* feedback cooling
        [m²(2π·Hz)²] = [m²/s²]

        ::

            S(ω) = A Γ / ((Ω²-ω²)² + Γ²ω²)

        Derived via ::

            Sxx = |χ(ω)|² Sth
            |χ(ω)|² = 1 / (m²((Ω²-ω²)² + Γ²ω²))
            Sth = 2m kBT Γ

        Hence ::

            A = 2 kBT / m

        Conventions: S is the *two-sided* displacement PSD per unit (linear) frequency [m²/Hz],
        written as a function of angular frequencies ω, Ω, Γ [2π·Hz]; Sth = fth² = 2 m kB T Γth is
        the two-sided force PSD, 〈F(t)F(t')〉 = Sth δ(t - t'). Check: ∫ S df over all f
        = A Γ / (2 Γ Ω²) = kB T / (m Ω²), i.e. equipartition.

        - a one-sided PSD G = 2S has amplitude 2A
        - with feedback (Γ = Γth + Γfb) and the directed force, the amplitude is 2 kB T_eff / m,
          with T_eff given by self.T_eff (A itself uses the gas temperature)
        - for a fit done in Hz (f, Ω/2π, Γ/2π), as in SpectralDensityEstimate.fit_PSD, the amplitude
          is A / (2π)³, see A_Hz
        """
        return 2 * kB * self.gas.T / self.bead.m

    @cached_property
    def A_Hz(self):
        """
        PSD fit amplitude A for a fit done in linear-frequency variables (f, Ω/2π, Γ/2π)
        [m²·Hz²]

        ::

            S(f) = A_Hz (Γ/2π) / (((Ω/2π)² - f²)² + (Γ/2π)² f²)

        The Lorentzian Γ/((Ω²-ω²)²+Γ²ω²) has dimensions of time³, so changing every frequency
        from 2π·Hz to Hz scales it by (2π)³ while the PSD value [m²/Hz] itself does not change::

            A_Hz = A / (2π)³ (two-sided; double it for a one-sided PSD)
        """
        return self.A / Hz**3

    @cached_property
    def A_rad(self):
        """
        Scaling factor for the theoretical PSD fitting expression
        [m²(2π·Hz)²]

        ::

            A_rad = A / 2π = kBT / π m
        """
        # NOTE: answer to "why A/2π and not A/(2π)²?": A_rad is the number the old
        # SpectralDensityEstimate.fit_PSD reported, which converted its Hz-fit amplitude with Hz² instead
        # of Hz³ (see A_Hz): A_Hz × (2π)² = A / 2π for a two-sided PSD. The physically meaningful
        # amplitudes are A (angular variables) and A_Hz (linear variables); fit_PSD now reports Afit
        # in the same units and convention as A (×2 if one-sided). Kept for backwards compatibility.
        return kB * self.gas.T / (π * self.bead.m)

    @cached_property
    def T_eff(self):
        """
        Dict of effective (motional) temperatures of the (x, y, z)-motions, from equipartition
        [K]

        The total force PSD on axis j is Sth + Sdir,j = Sth (1 + β² cos²ψ) for x, (1 + β² sin²ψ) for y,
        (1) for z, and the total damping is Γj = Γth + Γfb,j, so ::

            T_eff,j = T Γth (1 + Sdir,j / Sth) / Γj

        which reduces to the cold-damping result T Γth / (Γth + Γfb,j) when β = 0.
        """
        return {
            axis: self.gas.T * self.Γth * (1 + self.Sdirj[axis] / self.Sth) / self.Γj[axis]
            for axis in ('x', 'y', 'z')
        }

    @cached_property
    def equipartition_factor(self):
        """
        Conversion factor of position variance into temperature
        [kg.(2π·Hz)² / (J/K)] = [K.(2π/m)]

        Equipartition theorem states ::

            kinetic energy from oscillation = thermal energy
            ½mΩ²〈x²〉 = ½kB*T
            Therefore expecting
            T = (mΩ² / kB) * 〈x²〉
        """
        return {
            axis: self.bead.m * Ωj**2 / kB
            for axis, Ωj in self.Ωj.items()
        }

    ################################################################################################
    # FUNCTIONS OF DYNAMIC VARIABLES
    # ----------------------------------------------------------------------------------------------
    def kinetic_energy(self, px, py, pz):
        """
        Energy associated with the nanoparticle Hamiltonian

        H_np = p² / 2m
        [J]
        """
        return (px**2 + py**2 + pz**2) / (2 * self.bead.m)

    def Utw(self, x, y, z):
        """
        Cycle-averaged energy of the tweezer-bead dipole interaction, the potential whose gradient is
        the tweezer force `Ftw`

        ``Utw = -(Re α_eff / 4) |E|²``
        [J]

        ``|E|`` is the amplitude of the light field, ``E(t) = |E| cos(ωt + ...)``; the dipole energy
        ``-(Re α_eff / 2) E(t)²`` oscillates at 2ω (~10¹⁵ Hz for 1064 nm light), far faster than the
        motion, and its cycle average gives the ¼. Before levisim 1.0 this returned the peak energy
        ``-(Re α_eff / 2) |E|²``, twice the potential of the force.
        """
        return -(np.real(self.α_eff) / 4) * self.tweezer.E_field_amplitude(x, y, z)**2

    def Ftwx(self, x, y, z):
        """
        Gradient force due to the tweezer potential in the x-direction

        Ftw,x = -dUtw/dx = (4x/wx²) Utw
        [N]
        """
        wx = self.tweezer.wx(z)
        return (4 * x / wx**2) * self.Utw(x, y, z)

    def Ftwy(self, x, y, z):
        """
        Gradient force due to the tweezer potential in the y-direction

        Ftw,y = -dUtw/dy = (4y/wy²) Utw
        [N]
        """
        wy = self.tweezer.wy(z)
        return (4 * y / wy**2) * self.Utw(x, y, z)

    def Ftwz(self, x, y, z):
        """
        Gradient force due to the tweezer potential in the z-direction

        Ftw,z = -dUtw/dz (derived manually)
        [N]
        """
        wz = self.tweezer.wz(z)
        ellipse = self.tweezer.ellipse(x, y, z)
        return (4 * z / wz**2) * self.Utw(x, y, z) * (1 - 2 * ellipse)

    def Ftw(self, x, y, z):
        """
        Gradient force due to the tweezer potential, (Ftw,x, Ftw,y, Ftw,z)
        Identical expressions to Ftwx, Ftwy, Ftwz, but evaluating the potential Utw (and the beam
        profiles) only once - used in the Simulation loop, where it is ~3x cheaper than three calls.
        [N]
        """
        # NOTE: these are bit-identical to the levisim <= 0.1 forces 0.5·(4x/wx²)·U with the peak
        # energy U = 2·Utw: halving by a power of two is exact in floating point
        U = self.Utw(x, y, z)
        wx = self.tweezer.wx(z)
        wy = self.tweezer.wy(z)
        wz = self.tweezer.wz(z)
        ellipse = self.tweezer.ellipse(x, y, z)
        return (
            (4 * x / wx**2) * U,
            (4 * y / wy**2) * U,
            (4 * z / wz**2) * U * (1 - 2 * ellipse),
        )

    @cached_property
    def σ_Rayleigh(self):
        """
        Rayleigh scattering cross section of a small dielectric sphere (no radiation reaction)

        σR = (8π/3) k⁴ (α/(4πε0))² = (8π/3) (π²V²/λ⁴) χ0², χ0 = 3(εr - 1)/(εr + 2)
        To leading order in the size it equals σ_scattering = k Im(α_eff)/ε0, which is what
        F_scattering uses.
        [m²]
        """
        λ = self.tweezer.λ
        V, εr = self.bead.V, self.bead.εr

        # equivalent to α / V.ε0
        χ0 = 3 * (εr - 1) / (εr + 2)
        return (8*π/3) * (π**2 * V**2 / λ**4) * χ0**2

    def F_scattering(self, x, y, z):
        """
        Scattering (photon recoil, radiation pressure) force along the beam axis

        ::

            F_scat,z = (Im α_eff / 2) |E(r)|² ∂z[kz - arctan(z/zR)]
                     = σ_scattering I(r) / c × (1 - zR/(k(zR² + z²)))

        with ``|E(r)|² = E_field_amplitude(x, y, z)²``, ``I = ε0 c |E|²/2`` the local intensity, and the
        Gouy-phase gradient reducing the effective wavenumber near the focus.
        The tweezer light is taken to propagate along +z, so the force pushes the bead towards +z,
        consistent with z_eq > 0.

        This is the dipole force's dissipative part, in the same time-averaged convention as
        Utw/Ftwj (``|E|`` is the complex amplitude, ``F_grad = (Re α/4)∇|E|²``). It equals Marko Toroš's
        ``F = (σR/σL)(2P/c)|u|²`` (σL = πw0², in the commented code it replaces) up to the Gouy factor
        and σR ≈ σ_scattering. Its transverse components (from the wavefront curvature, ∝ x/R(z))
        and the associated recoil heating (momentum diffusion) are NOT included.
        [N]
        """
        zR, k = self.tweezer.zR, self.tweezer.k
        phase_gradient = k - zR / (zR**2 + z**2)
        return 0.5 * np.imag(self.α_eff) * self.tweezer.E_field_amplitude(x, y, z)**2 * phase_gradient

    def F_photon_recoil(self, x, y, z):
        """
        Photon recoil force used by Simulation, along +z: F_scattering if photon_recoil is enabled
        (the sum over the PhotonRecoil elements), else 0
        [N]
        """
        recoils = self._element_plan['recoils']
        if not recoils:
            return 0
        if len(recoils) == 1 and recoils[0].tweezer is self.tweezer:
            return self.F_scattering(x, y, z)
        F = 0
        for recoil in recoils:
            F = F + recoil.force((x, y, z), 0.0, self.bead)[2]
        return F

    ################################################################################################
    # FORCES OF THE ELEMENTS (used by Simulation)
    # ----------------------------------------------------------------------------------------------
    def element_force(self, element, x, y, z, t=0.0):
        """
        Force of one element of `fields` on the particle at (x, y, z) and time t, as a (3,) array;
        the trap's tweezer goes through Ftw (so that a Trap subclass overriding Utw/Ftw is honoured)
        and kicks are given the trap's gas
        [N]
        """
        if element.is_optical and element is self.tweezer:
            return np.array(self.Ftw(x, y, z))
        if isinstance(element, Kick):
            return element.force((x, y, z), t, self.bead, gas=self.gas)
        return element.force((x, y, z), t, self.bead)

    def field_force(self, x, y, z, t=0.0):
        """
        Sum of the forces of the 'field' elements (tweezers, potentials; not feedback, kicks,
        recoil or gravity), as (Fx, Fy, Fz), summed in the order of `fields`

        For a trap whose only field element is its tweezer this is exactly Ftw(x, y, z).
        [N]
        """
        plan = self._element_plan
        if plan['single_tweezer']:
            return self.Ftw(x, y, z)
        elements = plan['groups'].get('field', [])
        if not elements:
            return (0.0, 0.0, 0.0)
        F = self.element_force(elements[0], x, y, z, t)
        for element in elements[1:]:
            F = F + self.element_force(element, x, y, z, t)
        return F[0], F[1], F[2]

    def kick_force(self, t):
        """
        The kick force (F_kick,x, F_kick,y) at time t; as Simulation always has, the last active kick
        wins, and its p, F0, Fx and Fy attributes are set
        [N]
        """
        F_kick_x, F_kick_y = 0, 0
        for kick in self._element_plan['kicks']:
            if kick.start <= t < kick.end:
                # K = p_kick / pth
                kick.p = kick.K * self.gas.pth

                # since
                # p_kick = F0 * duration
                # then
                # F0 = p_kick / duration
                kick.F0 = kick.p / kick.duration

                F_kick_x = kick.F0 * np.cos(kick.θ)
                F_kick_y = kick.F0 * np.sin(kick.θ)

                kick.Fx = F_kick_x
                kick.Fy = F_kick_y
        return F_kick_x, F_kick_y

    def static_force(self, r, t=0.0):
        """
        Total position-dependent force of the elements at r and time t: the field, recoil and
        gravity elements (not the feedback damping or the kicks); a time-dependent element (a
        `PaulTrap`) contributes its time-averaged `secular_force` instead of its force at t
        [N]
        """
        x, y, z = as_position(r)
        F = np.zeros(3)
        for element in self.fields:
            if element.simulation_group in ('field', 'recoil', 'gravity'):
                if element.is_time_dependent:
                    # an rf element (PaulTrap) enters through its time-averaged (secular) force
                    F = F + element.secular_force((x, y, z), self.bead)
                else:
                    F = F + self.element_force(element, x, y, z, t)
        return F

    def find_equilibrium(self, r_guess=None, t=0.0):
        """
        Equilibrium position of the particle, the root of `static_force` near r_guess (default the
        origin), e.g. the gravitational sag -g/Ωj² of a Gravity element, or z_eq_exact with recoil
        [m]
        """
        r_guess = np.zeros(3) if r_guess is None else as_position(r_guess)
        from levisim.physics.fields.magnetic import (
            MagneticTrap,
            find_static_equilibrium,
        )
        magnetic = self.fields_of_type(MagneticTrap)
        if magnetic:
            # a magnetic trap: Newton's method on the analytic stiffness, after a scan of the vertical
            # line through r_guess for the stable force balance with gravity (the origin is usually
            # not near the levitation point, and the stiffness there need not be restoring)
            return find_static_equilibrium(self, r_guess, magnetic[0].source.length_scale)
        scale = 1e-9  # solve in nm
        # residuals in units of the force of a 1 nm displacement (stiffness at the origin)
        K0 = sum(element.stiffness(None, self.bead) for element in self.fields)
        force_scale = max(float(np.max(np.abs(np.diag(K0)))) * scale, 1e-30)

        def residual(u):
            return self.static_force(u * scale, t) / force_scale

        solution, _, status, message = fsolve(residual, r_guess / scale, full_output=True, xtol=1e-13)
        if status != 1:
            raise RuntimeError(f'find_equilibrium did not converge: {message}')
        return solution * scale

    def magnetic_trap_report(self, element=None, r_guess=None, scan_range=None):
        """
        Levitation condition and margin, equilibrium (with gravity and every other static element),
        stiffness tensor, axis and normal-mode frequencies, couplings, trap depth and eddy-current
        damping of the trap's `MagneticTrap` (the first, or `element`), as a dict; see
        `levisim.physics.fields.magnetic.magnetic_trap_report`
        [dict]
        """
        from levisim.physics.fields.magnetic import magnetic_trap_report
        return magnetic_trap_report(self, element=element, r_guess=r_guess, scan_range=scan_range)

    def magnetic_trap_report_str(self, element=None, r_guess=None, scan_range=None):
        """The `magnetic_trap_report` as a printable summary"""
        from levisim.physics.fields.magnetic import magnetic_trap_report_str
        return magnetic_trap_report_str(self.magnetic_trap_report(element, r_guess, scan_range))

    def paul_trap_report(self, element=None):
        """
        The `PaulTrap.report` of the trap's Paul trap (the first, or `element`), with the stiffness of
        all the other elements at the linearisation point folded into the Mathieu a_j, so that its
        'Ω_exact' are the exact (Floquet) secular frequencies of the hybrid trap, and 'stable' its
        stability flag
        [dict]
        """
        from levisim.physics.fields.electric import PaulTrap
        if element is None:
            paul_traps = self.fields_of_type(PaulTrap)
            if not paul_traps:
                raise ValueError('This Trap has no PaulTrap in its fields')
            element = paul_traps[0]
        others = sum(
            (K for other, K in self.stiffness_contributions if other is not element),
            np.zeros((3, 3)),
        )
        return element.report(self.bead, extra_stiffness=others)

    @cached_property
    def z_eq_exact(self):
        """
        On-axis equilibrium point of the gradient + scattering forces, Ftw,z(0, 0, z) + F_scat(0, 0, z) = 0,
        found numerically in (0, zR) (no z/zR ≪ 1 assumption; compare z_eq)
        [m]
        """
        def total_Fz(z):
            return self.Ftwz(0, 0, z) + self.F_scattering(0, 0, z)
        return brentq(total_Fz, 0, self.tweezer.zR, xtol=1e-16, rtol=1e-14)

    @cached_property
    def Ωj_at_z_eq(self):
        """
        Dict of harmonic frequencies about the shifted equilibrium (0, 0, z_eq_exact), from central
        differences of the total force: Ωj² = -∂Fj/∂j / m (including ∂F_scat/∂z for z).
        The default Ωj are evaluated at the focus; at z_eq_exact the beam is wider, so Ωx, Ωy are lower.
        [2π·Hz]
        """
        z0, m = self.z_eq_exact, self.bead.m
        h = 1e-4 * self.tweezer.w0
        kx = -(self.Ftwx(h, 0, z0) - self.Ftwx(-h, 0, z0)) / (2*h)
        ky = -(self.Ftwy(0, h, z0) - self.Ftwy(0, -h, z0)) / (2*h)

        def total_Fz(z):
            return self.Ftwz(0, 0, z) + self.F_scattering(0, 0, z)
        kz = -(total_Fz(z0 + h) - total_Fz(z0 - h)) / (2*h)
        return {axis: np.sqrt(k / m) for axis, k in (('x', kx), ('y', ky), ('z', kz))}
