import numpy as np
from functools import cached_property

from levisim.core.constants import π, ħ, c, kB
from levisim.core.units import (
    pm,
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
        self.bead = kwargs.get('bead', Bead())
        self.gas = kwargs.get('gas', Gas())
        # Kick() instances for simulating discrete impacts
        # generate this with kicks.generate_kick_list()
        self.kicks = kwargs.get('kicks', [])

        ############################################################################################
        # DEFINED PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Feedback mechanical damping rates of (x,y,z)-motion oscillations
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
        # Whether or not to use photon recoil force
        # (Tania doesn't use it)
        self.photon_recoil = kwargs.get('photon_recoil', False)

        # ------------------------------------------------------------------------------------------
        # Override resonant frequencies (that ignore Tweezer properties)
        # useful to get SpectralDensityTheory curves to match Experiment or Simulation data
        self.override_Ω_dict = kwargs.get('override_Ω_dict', None)

        # to calculate theoretical spectral density curves
        self.theory = SpectralDensityTheory(
            self.bead.m,
            self.Ωx, self.Ωy, self.Ωz,
            self.Γ_fbx, self.Γ_fby, self.Γ_fbz,
            self.fth, self.β, self.ψ,
        )

        ############################################################################################
        # MULTI-TRAP PROPERTIES
        # ------------------------------------------------------------------------------------------
        self.trap_index = kwargs.get('trap_index')

    @property
    def info(self):
        """
        Prints information on Trap instance
        """
        print(self.info_str)

    @property
    def info_str(self):
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
            f'\n    kx = {self.kx:.5e} N/m'
            f'\n    ky = {self.ky:.5e} N/m'
            f'\n    kz = {self.kz:.5e} N/m'
            f'\n'
            f'\n    U_depth = {self.U_depth / eV:.5f} eV'
            f'\n    U_depth = {self.U_depth / kB:.5e} K'
            f'\n    v_escape = {self.v_escape:.5f} m/s'
            f'\n    p_escape = {self.p_escape:.5e} kg.m/s'
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
        Dict of directed force noises in the (x,y,z)-axes
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
        Dict of directed force spectral density for (x,y,z)-axis motion
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
        Total mechanical damping rate of (x,y,z)-motion oscillations
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
        Dict of total mechanical damping rates for (x,y,z)-motion oscillations
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
        Dict of Autocorrelation / thermalisation times for (x,y,z)-motion oscillations
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
        return self.bead.α * self.tweezer.εtw**2 / self.tweezer.wx0**2

    @cached_property
    def ky(self):
        """
        Spring constant of the restoring force in the y-axis due to the tweezer potential
        [N/m]
        """
        return self.bead.α * self.tweezer.εtw**2 / self.tweezer.wy0**2

    @cached_property
    def kz(self):
        """
        Spring constant of the restoring force in the z-axis due to the tweezer potential
        [N/m]
        """
        return self.bead.α * self.tweezer.εtw**2 / self.tweezer.wz0**2

    @cached_property
    def kj(self):
        """
        Dict of spring constants of the restoring forces in the (x,y,z)-axes due to the tweezer potential
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

    # ----------------------------------------------------------------------------------------------
    # STANDARD QUANTUM LIMITS
    # ----------------------------------------------------------------------------------------------
    # TODO: find source for this (cavity optomechanics review?)
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
        Standard Quantum Limit displacements for (x,y,z)-motions
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
        [N]
        """
        return np.sqrt(2 * ħ * self.bead.m * self.Γx * self.Ωx)

    @cached_property
    def Fy_SQL(self):
        """
        Standard Quantum Limit forces in y-axis
        [N]
        """
        return np.sqrt(2 * ħ * self.bead.m * self.Γy * self.Ωy)

    @cached_property
    def Fz_SQL(self):
        """
        Standard Quantum Limit forces in z-axis
        [N]
        """
        return np.sqrt(2 * ħ * self.bead.m * self.Γz * self.Ωz)

    @cached_property
    def Fj_SQL(self):
        """
        Standard Quantum Limit forces in (x,y,z)-axes
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
        return np.abs(-(self.bead.α / 2) * np.abs(self.tweezer.E(0, 0, 0))**2)

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
        [kg.m/s]
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

    # ----------------------------------------------------------------------------------------------
    # SCALE FACTORS FOR FITTING
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def A(self):
        """
        Scaling factor for the theoretical PSD fitting expression
        [m²Hz²]
        PSD = A Γ / ((Ω²-ω²)² + Γ²ω²)

        Derived via
        Sxx =  |χ(ω)|² Sth
        |χ(ω)|² = 1 / (m²((Ω²-ω²)² + Γ²ω²))
        Sth = 2m kBT Γ

        Hence
        A = 2 kBT / m
        """
        return 2 * kB * self.gas.T / self.bead.m

    @cached_property
    def A_rad(self):
        """
        Scaling factor for the theoretical PSD fitting expression
        [m²(2π·Hz)²]
        A_rad = A / 2π = kBT / π m
        """
        # TODO: figure out why A/2π and not A/(2π)² ?
        return kB * self.gas.T / (π * self.bead.m)

    @cached_property
    def equipartition_factor(self):
        """
        Conversion factor of position variance into temperature
        [kg.(2π·Hz)² / (J/K)] = [K.(2π/m)]
        Equipartition theorem states
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
        Energy associated with the nanoparticle Hamiltonian [J]
        H_np = p**2 / 2m
        """
        return (px**2 + py**2 + pz**2) / (2 * self.bead.m)

    def Utw(self, x, y, z):
        """
        Energy of the tweezer-bead interaction Hamiltonian [J]
        H_tw = -α/2 * |Etw|**2
        i.e. energy of tweezer-bead dipole interaction
        """
        # NOTE: 1064nm tweezer light oscillates at about 2.8 THz >> kHz oscillations of the bead.
        # Since this is too quick to matter for the dynamics, the cos²(θ) oscillation is averaged to ½.
        # This factor is thus *artificially* included for the final tweezer force functions Ftwx, Ftwy, Ftwz
        return -(self.bead.α / 2) * np.abs(self.tweezer.E(x, y, z))**2

    def Ftwx(self, x, y, z):
        """ Force due to the tweezer in the x-direction: Ftw,x = -dUtw/dx (derived manually) [N] """
        wx = self.tweezer.wx(z)
        # NOTE: *artificially* include this ½ to account for the averaging of the very fast cos²(θ) of the light fields oscillations
        return 0.5 * (4 * x / wx**2) * self.Utw(x, y, z)

    def Ftwy(self, x, y, z):
        """ Force due to the tweezer in the y-direction: Ftw,y = -dUtw/dy (derived manually) [N] """
        wy = self.tweezer.wy(z)
        # NOTE: *artificially* include this ½ to account for the averaging of the very fast cos²(θ) of the light fields oscillations
        return 0.5 * (4 * y / wy**2) * self.Utw(x, y, z)

    def Ftwz(self, x, y, z):
        """ Force due to the tweezer in the z-direction: Ftw,z = -dUtw/dz (derived manually) [N] """
        wz = self.tweezer.wz(z)
        ellipse = self.tweezer.ellipse(x, y, z)
        # NOTE: *artificially* include this ½ to account for the averaging of the very fast cos²(θ) of the light fields oscillations
        return 0.5 * (4 * z / wz**2) * self.Utw(x, y, z) * (1 - 2 * ellipse)

    # TODO: fix photon recoil!
    # @cached_property
    # def σR(self):
    #     """ Effective Rayleigh scattering cross section """
    #     λ = self.tweezer.λ
    #     V, εR = self.bead.V, self.bead.εR

    #     # equivalent to α / V.ε0
    #     χ0 = 3 * (εR - 1) / (εR + 2)
    #     return (8*π/3) * (π**2 * V**2 / λ**4) * χ0**2

    # def F_photon_recoil(self, x, y, z):
    #     """
    #     Force due to photon recoil
    #     Taken from Marko's ro-translational manuscript, Eq. 37 etc.
    #     This is the effect of the interaction between the tweezer field and the free field
    #     i.e. -α.Etw.Efree
    #     """
    #     if not self.photon_recoil:
    #         return 0
    #     P, w0 = self.tweezer.Power, self.tweezer.w0
    #     σL = π * w0**2  # effective laser cross section - could use wx0*wy0 here?
    #     return (self.σR / σL) * (2*P / c) * np.abs(self.tweezer.u(x, y, z))**2

    # TODO: find the real z equilibrium point (when recoil is included)... Some function of the scattering areas!
    # Appendix A of https://arxiv.org/pdf/2012.15822.pdf (not in https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.023071 ) !!!
    # Derive these for z0 ≠ 0 i.e. the real equilibrium point after photon recoil
