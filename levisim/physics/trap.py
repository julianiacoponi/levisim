"""
Combined physics of a levitated free-space tweezer trap (Trap)
"""

import numpy as np
from functools import cached_property
from scipy.optimize import brentq

from levisim.core.constants import π, ħ, c, kB, ε0
from levisim.core.units import (
    μm, pm,
    ms,
    degrees,
    Hz, kHz,
    zN,
    eV,
)

from levisim.physics.tweezer import Tweezer
from levisim.physics.bead import Bead
from levisim.physics.gas import Gas

from levisim.analysis.spectral_density_theory import SpectralDensityTheory


class Trap:
    """
    Combined physics of a levitated free-space tweezer trap
    i.e. of an experimental setup of
    - a Bead instance, levitated by
    - a Tweezer instance, floating in
    - a Gas instance

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
        self.tweezer = kwargs.get('tweezer', Tweezer())
        # the levitated particle; `particle=` is accepted as an alias of `bead=`
        if 'bead' in kwargs and 'particle' in kwargs:
            raise ValueError('Pass either bead or particle, not both')
        self.bead = kwargs['bead'] if 'bead' in kwargs else kwargs.get('particle', Bead())
        self.gas = kwargs.get('gas', Gas())
        # Kick() instances for simulating discrete impacts
        # generate this with kicks.generate_kick_list()
        self.kicks = kwargs.get('kicks', [])

        ############################################################################################
        # DEFINED PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Feedback mechanical damping rates of (x, y, z)-motion oscillations
        # [2π·Hz]
        self.Γ_fbx = float(kwargs.get('Γ_fbx', 2.1*kHz))
        self.Γ_fby = float(kwargs.get('Γ_fby', 2.5*kHz))
        self.Γ_fbz = float(kwargs.get('Γ_fbz', 0.0*kHz))

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
        # Whether or not to include the photon recoil (scattering) force along the beam axis in
        # Simulation, see F_scattering (Tania doesn't use it)
        # [bool]
        self.photon_recoil = bool(kwargs.get('photon_recoil', False))

        # ------------------------------------------------------------------------------------------
        # Override resonant frequencies (that ignore Tweezer properties)
        # useful to get SpectralDensityTheory curves to match Experiment or Simulation data
        self.override_Ω_dict = kwargs.get('override_Ω_dict', None)

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
            f'\n    α_eff = {self.α_eff:.5e} Fm²'
            f'\n    σ_scattering = {self.σ_scattering:.5e} m²'
            f'\n    z_eq = {self.z_eq/ μm:.5f} μm'
            f'\n'
            f'\n    U_depth = {self.U_depth / eV:.5f} eV'
            f'\n    U_depth = {self.U_depth / kB:.5e} K'
            f'\n    v_escape = {self.v_escape:.5f} m/s'
            f'\n    p_escape = {self.p_escape:.5e} kg·m/s'
            f'\n'
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
            f'\n    id(tweezer) = {id(self.tweezer)}'
            f'\n    id(bead) = {id(self.bead)}'
            f'\n    id(gas) = {id(self.gas)}'
            f'\n)'
        )

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
    @cached_property
    def Γth(self):
        """
        Thermal gas mechanical damping rate
        [2π·Hz]
        """
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
    @cached_property
    def Γx(self):
        """
        Total mechanical damping rate of (x, y, z)-motion oscillations
        [2π·Hz]
        """
        return self.Γth + self.Γ_fbx

    @cached_property
    def Γy(self):
        """
        Total mechanical damping rate of y-motion oscillations
        [2π·Hz]
        """
        return self.Γth + self.Γ_fby

    @cached_property
    def Γz(self):
        """
        Total mechanical damping rate of z-motion oscillations
        [2π·Hz]
        """
        return self.Γth + self.Γ_fbz

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
    def kx(self):
        """
        Spring constant of the restoring force in the x-axis due to the tweezer potential
        [N/m]
        """
        return np.real(self.α_eff) * self.tweezer.E0**2 / self.tweezer.wx0**2

    @cached_property
    def ky(self):
        """
        Spring constant of the restoring force in the y-axis due to the tweezer potential
        [N/m]
        """
        return np.real(self.α_eff) * self.tweezer.E0**2 / self.tweezer.wy0**2

    @cached_property
    def kz(self):
        """
        Spring constant of the restoring force in the z-axis due to the tweezer potential
        [N/m]
        """
        return np.real(self.α_eff) * self.tweezer.E0**2 / self.tweezer.wz0**2

    @cached_property
    def kj(self):
        """
        Dict of spring constants of the restoring forces in the (x, y, z)-axes due to the tweezer potential
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
        Energy depth of the potential well
        [J]
        """
        # due to tweezer-bead dipole interaction = -α/2 * |Etw(0)|**2
        # tweezer potential is negative, so this is an absolute depth size
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

    @cached_property
    def α_eff(self):
        """
        Effective polarisability of the bead that accounts for the radiation reaction force
        [complex Fm²]
        """
        return self.bead.α / (1 - 1j * (self.tweezer.k**3 / (6*π*ε0)) * self.bead.α)

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
        prefactor = π / (15 * ħ * ε0) * (np.real(self.α_eff) * self.tweezer.E0 / (2*π))**2 * self.tweezer.k**5
        coefficients = {'x': 2, 'y': 1, 'z': 7}
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
        Energy of the tweezer-bead interaction Hamiltonian

        ``H_tw = -α/2 * |Etw|²``
        [J]
        i.e. energy of tweezer-bead dipole interaction
        """
        # NOTE: 1064nm tweezer light oscillates at about 282 THz >> kHz oscillations of the bead.
        # Since this is too quick to matter for the dynamics, the cos²(θ) oscillation is averaged to ½.
        # This factor is thus *artificially* included for the final tweezer force functions Ftwx, Ftwy, Ftwz
        return -(np.real(self.α_eff) / 2) * self.tweezer.E_field_amplitude(x, y, z)**2

    def Ftwx(self, x, y, z):
        """
        Gradient force due to the tweezer potential in the x-direction

        Ftw,x = -dUtw/dx (derived manually)
        [N]
        """
        wx = self.tweezer.wx(z)
        # NOTE: *artificially* include this ½ to account for the averaging of the very fast cos²(θ) of the light fields oscillations
        return 0.5 * (4 * x / wx**2) * self.Utw(x, y, z)

    def Ftwy(self, x, y, z):
        """
        Gradient force due to the tweezer potential in the y-direction

        Ftw,y = -dUtw/dy (derived manually)
        [N]
        """
        wy = self.tweezer.wy(z)
        # NOTE: *artificially* include this ½ to account for the averaging of the very fast cos²(θ) of the light fields oscillations
        return 0.5 * (4 * y / wy**2) * self.Utw(x, y, z)

    def Ftwz(self, x, y, z):
        """
        Gradient force due to the tweezer potential in the z-direction

        Ftw,z = -dUtw/dz (derived manually)
        [N]
        """
        wz = self.tweezer.wz(z)
        ellipse = self.tweezer.ellipse(x, y, z)
        # NOTE: *artificially* include this ½ to account for the averaging of the very fast cos²(θ) of the light fields oscillations
        return 0.5 * (4 * z / wz**2) * self.Utw(x, y, z) * (1 - 2 * ellipse)

    def Ftw(self, x, y, z):
        """
        Gradient force due to the tweezer potential, (Ftw,x, Ftw,y, Ftw,z)
        Identical expressions to Ftwx, Ftwy, Ftwz, but evaluating the potential Utw (and the beam
        profiles) only once - used in the Simulation loop, where it is ~3x cheaper than three calls.
        [N]
        """
        U = self.Utw(x, y, z)
        wx = self.tweezer.wx(z)
        wy = self.tweezer.wy(z)
        wz = self.tweezer.wz(z)
        ellipse = self.tweezer.ellipse(x, y, z)
        return (
            0.5 * (4 * x / wx**2) * U,
            0.5 * (4 * y / wy**2) * U,
            0.5 * (4 * z / wz**2) * U * (1 - 2 * ellipse),
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
        Photon recoil force used by Simulation: F_scattering if photon_recoil is enabled, else 0
        [N]
        """
        if not self.photon_recoil:
            return 0
        return self.F_scattering(x, y, z)

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
