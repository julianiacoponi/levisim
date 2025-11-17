import numpy as np
from functools import cached_property
from scipy.special import jv as Jn
import scipy.integrate as integrate

from levisim.core.constants import π, c, ε0, μ0
from levisim.core.units import (
    degrees,
    mm, μm, nm,
    THz,
    mW,
    V,
)


class FocussingOptics:
    """
    Defines the optics for an incoming Gaussian into a strongly focussed spot
    """
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialises a FocussingOptics instance
        """
        ############################################################################################
        # DEFINED PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Incoming laser power
        # [W]
        self.Power = float(kwargs.get('Power', 500*mW))

        # ------------------------------------------------------------------------------------------
        # Incoming laser wavelength
        # [m]
        self.λ = float(kwargs.get('λ', 1064*nm))

        # ------------------------------------------------------------------------------------------
        # Incoming Gaussian beam waist
        # [m]
        # NOTE: Jon Gosling's PhD says in section 3.1.1
        #   "A telescope consisting of two lenses, typically in a 1 to 4 ratio,
        #   is used to magnify the size of the laser beam to approximately 5 mm
        #   so that the light overfills the lens"
        # https://discovery.ucl.ac.uk/id/eprint/10192690/2/Jonathan_Gosling_Thesis.pdf
        self.w0 = float(kwargs.get('w0', 5*mm))

        # --------------------------------------------- ---------------------------------------------
        # Focal length of focussing lens
        # [m]
        # NOTE: Since the electric field this model ultimately predicts has an amplitude that
        # oscillates with -i exp(-ikf)
        # when f = 1/4 mod λ, -i exp(-ikf) = -1 = -Re
        # when f = 1/2 mod λ, -i exp(-ikf) = +i = +Im
        # when f = 3/4 mod λ, -i exp(-ikf) = +1 = +Re
        # when f = 0,1 mod λ, -i exp(-ikf) = -i = -Im
        # therefore, default to integer + 0.75 of a wavelength, to get a purely positve real amplitude

        # NOTE: Hayden Fu's PhD
        # https://discovery.ucl.ac.uk/id/eprint/10213440/1/Fu_Thesis.pdf
        # quotes using this lens, LightPath 355330
        # https://www.edmundoptics.com/p/077-na-310mm-fl-600-1050nm-ar-coated-molded-aspheric-lens/26748/
        # with EFL (effective focal length) of 3.1mm for 830nm light
        # therefore there is some leeway as to how 1064nm light would behave

        # NOTE: ThorLabs has similar lenses e.g. 354330-C/C330TMD-C (mounted/unmounted)
        # with EFL = 3.1mm for 830nm light, but a slightly worse NA=0.70
        # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=3812
        # This lens quotes a focal shift for 1064nm light of about +0.023mm or +0.0234mm
        # https://www.thorlabs.com/images/TabImages/354330_Focal_Shift_780.gif
        # Therefore, 3.123mm / 1064nm ≈ 2935.15... so round up since the shift is maybe a bit more
        self.focal_length = float(kwargs.get('focal_length', 2935.75 * self.λ))

        # ------------------------------------------------------------------------------------------
        # Numerical aperture of focussing lens
        # [·]
        # https://www.edmundoptics.com/p/077-na-310mm-fl-600-1050nm-ar-coated-molded-aspheric-lens/26748/
        self.NA = float(kwargs.get('NA', 0.77))

        # ------------------------------------------------------------------------------------------
        # Refractive index of surrounding medium
        # [·]
        # defaults to air at atmospheric pressure
        self.n = float(kwargs.get('n', 1.000273))

        # ------------------------------------------------------------------------------------------
        # Relative (magnetic) permeability of surrounding medium
        # [·]2
        self.μr = float(kwargs.get('μr', 1))

        # ------------------------------------------------------------------------------------------
        # NOTE: slightly clunky (but effective) way of allowing numpy array inputs and outputs to
        # this class's methods, since the decorator @np.vectorize doesn't handle `self` well
        # https://stackoverflow.com/questions/48981501/is-it-possible-to-numpy-vectorize-an-instance-method
        self.fw = np.vectorize(self._fw)

        self.I0n_integrand = np.vectorize(self._I0n_integrand)
        self.I00 = np.vectorize(self._I00)
        self.I01 = np.vectorize(self._I01)
        self.I02 = np.vectorize(self._I02)

        self.IE_00 = np.vectorize(self._IE_00)
        self.E0_focus = np.vectorize(self._E0_focus)
        self.Ex_focus = np.vectorize(self._Ex_focus)
        self.Ey_focus = np.vectorize(self._Ey_focus)
        self.Ez_focus = np.vectorize(self._Ez_focus)
        self.magnitude_E_focus = np.vectorize(self._magnitude_E_focus)
        self.E_focus = np.vectorize(self._E_focus)

        self.I_focus = np.vectorize(self._I_focus)

        self.H_focus = np.vectorize(self._H_focus)
        self.H0_focus = np.vectorize(self._H0_focus)
        self.Hx_focus = np.vectorize(self._Hx_focus)
        self.Hy_focus = np.vectorize(self._Hy_focus)
        self.Hz_focus = np.vectorize(self._Hz_focus)
        self.magnitude_H_focus = np.vectorize(self._magnitude_H_focus)

        self.S_focus = np.vectorize(self._S_focus)

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info(self):
        """
        Prints information on the FocussingOptics instance
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        String describing the FocussingOptics instance
        """
        return (
            f'FocussingOptics(id={id(self)}'
            f'\n    INCOMING GAUSSIAN LASER BEAM'
            f'\n    Defined'
            f'\n        Power = {self.Power / mW:.1f} mW'
            f'\n        λ = {self.λ / nm:.1f} nm'
            f'\n        w0 = {self.w0 / mm:.3f} mm'
            f'\n'
            f'\n    Derived'
            f'\n        k = {self.k:.5e} 2π·m⁻¹'
            f'\n        ω/2π = {self.ω / THz:.3f} THz'
            f'\n        T = {self.T:.5e} s'
            f'\n        E0 = {self.E0:.5e} V/m'
            f'\n        I0 = {self.I0:.5e} W/m²'
            f'\n'
            f'\n    FOCUSSING LENS'
            f'\n    Defined'
            f'\n        focal length, f = {self.focal_length / self.λ:.2f} λ = {self.focal_length / mm:.4f} mm'
            f'\n        NA = {self.NA:.2f}'
            f'\n'
            f'\n    SURROUNDING MEDIUM'
            f'\n    Defined'
            f'\n        n = {self.n:.6f}'
            f'\n        μr = {self.μr:.3f}'
            f'\n'
            f'\n    Derived'
            f'\n        εr = {self.εr:.6f}'
            f'\n        impedance, Z = {self.Z_impedance:.3f} Ω'
            f'\n        admittance, Y = {self.Y_admittance:.3f} Ω⁻¹'
            f'\n'
            f'\n    COMBINED'
            f'\n        θ_max = sin⁻¹(NA/n) = {self.θ_max / degrees:.2f}°'
            f'\n        aperture radius, a = f sin(θ_max) = {self.aperture_radius / mm:.3f} mm'
            f'\n        filling factor, f0 = w0/a = {self.f0:.3f}'
            f'\n        Fresnel number, F = a²/fλ = {self.Fresnel_number:.3f}'
            f'\n        Σ = {self.Σ / (V / μm):.3e} V/m'
            f'\n        Σ / Z_imp = {self.Σ / self.Z_impedance:.3e} A/m'
            f'\n)'
        )

    ################################################################################################
    # DERIVED PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def k(self):
        """
        Angular wavenumber of the incoming laser light
        [2π/m]
        """
        return 2*π / self.λ

    @cached_property
    def ω(self):
        """
        Angular frequency of the incoming laser light
        [2π·Hz]
        """
        return c * self.k

    @cached_property
    def T(self):
        """
        Period of the incoming laser light waves
        [s]
        """
        return 2*π / self.ω

    @cached_property
    def E0(self):
        """
        Peak electric field value of incoming Gaussian laser beam
        [V/m]
        """
        return np.sqrt(4 * self.Power / (π * self.w0**2 * ε0 * c))

    @cached_property
    def I0(self):
        """
        Peak electric field intensity of incoming Gaussian laser beam
        [W/m²]
        """
        return 2 * self.Power / (π * self.w0**2)

    @cached_property
    def εr(self):
        """
        Relative permittivity of the surrounding medium
        [·]
        """
        return self.n**2 / self.μr

    @cached_property
    def Z_impedance(self):
        """
        Impedance of the surrounding medium
        [Ω]
        """
        return np.sqrt((μ0 * self.μr) / (ε0 * self.εr))

    @cached_property
    def Y_admittance(self):
        """
        Admittance of the surrounding medium
        [Ω⁻¹]
        """
        return 1 / self.Z_impedance

    @cached_property
    def θ_max(self):
        """
        Maximum angle of incoming light ray that will be focussed by the focussing lens
        [rad]
        """
        return np.arcsin(self.NA / self.n)

    @cached_property
    def aperture_radius(self):
        """
        Aperture radius of the focussing lens
        [m]
        """
        return self.focal_length * np.sin(self.θ_max)

    @cached_property
    def f0(self):
        """
        Filling factor of the focussing lens
        [·]
        """
        return self.w0 / self.aperture_radius

    @cached_property
    def Fresnel_number(self):
        """
        Fresnel number of focussing lens for the given wavelength of incoming laser light
        [·]
        """
        return self.aperture_radius**2 / (self.focal_length * self.λ)

    @cached_property
    def Σ(self):
        """
        Scaling factor in front of the focussed EM fields
        [complex V/m]
        """
        return (
            -(1j * self.k * self.focal_length / 2)
            *
            self.E0
            *
            np.exp(-1j * self.k * self.focal_length)
        )

    def _fw(self, θ):
        """
        Apodisation function (equivalent to a pupil filter)
        [·]
        """
        return np.exp(
            -(1 / self.f0**2)
            *
            np.sin(θ)**2 / np.sin(self.θ_max)
        )

    # ----------------------------------------------------------------------------------------------
    # BESSEL INTEGRALS
    # ----------------------------------------------------------------------------------------------
    def _I0n_integrand(
        self,
        n,
        ρ, z, θ,
    ):
        """
        Integrand of the I0n integrals
        [·]
        """
        return (
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html#scipy.special.jv
            Jn(n, self.k * ρ * np.sin(θ))
            *
            self._fw(θ)
            *
            np.sqrt(np.cos(θ))
            *
            np.exp(1j * self.k * z * np.cos(θ))
            *
            # NOTE: Fresnel transmission coefficients can scale these parts as follows
            # (1 ± cosθ) --> (τs ± τp * cosθ)
            # sin²θ --> τp·sin²θ
            # https://www.spiedigitallibrary.org/conference-proceedings-of-spie/2655/0000/Confocal-imaging-through-a-dielectric-interface-aberration-modeling/10.1117/12.237486.pdf
            [
                np.sin(θ) * (1 + np.cos(θ)),
                np.sin(θ)**2,
                np.sin(θ) * (1 - np.cos(θ)),
            ][n]
        )

    # NOTE: integrate.quad() docs state
    # https://docs.scipy.org/doc/scipy/tutorial/integrate.html
    #   "The return value is a tuple,
    #   with the first element holding the estimated value of the integral
    #   and the second element holding an estimate of the absolute integration error."
    # NOTE: integration over complex values can be done linearly
    # i.e. integrate real and imaginary parts separately, sum to give the complex answer
    # https://stackoverflow.com/questions/5965583/use-scipy-integrate-quad-to-integrate-complex-numbers
    def _I00(self, ρ, z):
        """
        Integral for Gaussian (0,0) mode with J0 Bessel function
        [·]
        """
        return (
            integrate.quad(lambda θ: np.real(self._I0n_integrand(0, ρ, z, θ)), 0, self.θ_max)[0]
            +
            1j * integrate.quad(lambda θ: np.imag(self._I0n_integrand(0, ρ, z, θ)), 0, self.θ_max)[0]
        )

    def _I01(self, ρ, z):
        """
        Integral for Gaussian (0,0) mode with J1 Bessel function
        [·]
        """
        return (
            integrate.quad(lambda θ: np.real(self._I0n_integrand(1, ρ, z, θ)), 0, self.θ_max)[0]
            +
            1j * integrate.quad(lambda θ: np.imag(self._I0n_integrand(1, ρ, z, θ)), 0, self.θ_max)[0]
        )

    def _I02(self, ρ, z):
        """
        Integral for Gaussian (0,0) mode with J2 Bessel function
        [·]
        """
        return (
            integrate.quad(lambda θ: np.real(self._I0n_integrand(2, ρ, z, θ)), 0, self.θ_max)[0]
            +
            1j * integrate.quad(lambda θ: np.imag(self._I0n_integrand(2, ρ, z, θ)), 0, self.θ_max)[0]
        )

    # ----------------------------------------------------------------------------------------------
    # FOCUSSED ELECTRIC FIELD
    # ----------------------------------------------------------------------------------------------
    def _IE_00(self, component, x, y, z):
        """
        Given component of Gaussian (0,0) mode's focussing integrals for the electric field
        [complex]
        """
        # NOTE: np.sin(±π) and np.cos(±π/2) are not exactly zero as they should be
        ρ, φ = np.sqrt(x**2 + y**2), np.arctan2(y, x)
        if component == 'x':
            if x == y and x != 0 and y != 0:
                # NOTE: for x = y ≠ 0, φ=π/4,-3π/4 so cos(2φ) = 0
                return self._I00(ρ, z)
            return self._I00(ρ, z) + self._I02(ρ, z) * np.cos(2*φ)
        if component == 'y':
            # NOTE: for the y-axis (x=0), φ=±π/2 so sin(2φ) = 0
            if x == 0.0:
                return 0.0
            return self._I02(ρ, z) * np.sin(2*φ)
        if component == 'z':
            # NOTE: for the y-axis (x=0), φ=±π/2 so cos(φ) = 0
            if x == 0.0:
                return 0.0
            return -2j * self._I01(ρ, z) * np.cos(φ)

    def _E0_focus(self, component, x, y, z):
        """
        Given component of focussed electric field's complex amplitude vector
        [complex V/m]
        """
        return self.Σ * self._IE_00(component, x, y, z)

    def _Ex_focus(self, x, y, z, t):
        """
        Focussed electric field vector's x-component
        [V/m]
        """
        return np.real(self._E0_focus('x', x, y, z) * np.exp(-1j * self.ω * t))

    def _Ey_focus(self, x, y, z, t):
        """
        Focussed electric field vector's y-component
        [V/m]
        """
        return np.real(self._E0_focus('y', x, y, z) * np.exp(-1j * self.ω * t))

    def _Ez_focus(self, x, y, z, t):
        """
        Focussed electric field vector's z-component
        [V/m]
        """
        return np.real(self._E0_focus('z', x, y, z) * np.exp(-1j * self.ω * t))

    def _magnitude_E_focus(self, x, y, z, t):
        """
        Magnitude of the focussed electric field vector
        [V/m]
        """
        return np.sqrt(
            self._E_focus('x', x, y, z, t)**2
            +
            self._E_focus('y', x, y, z, t)**2
            +
            self._E_focus('z', x, y, z, t)**2
        )

    def _E_focus(self, component, x, y, z, t):
        """
        Given component of the focussed electric field vector
        [V/m]
        """
        if component == 'x':
            return self._Ex_focus(x, y, z, t)
        if component == 'y':
            return self._Ey_focus(x, y, z, t)
        if component == 'z':
            return self._Ez_focus(x, y, z, t)
        if component == 'magnitude':
            return self._magnitude_E_focus(x, y, z, t)

    def _I_focus(self, x, y, z, t):
        """
        Intensity of the focussed electric field
        [W/m²]
        """
        return 0.5 * ε0 * self._magnitude_E_focus(x, y, z, t)**2 * c

    # ----------------------------------------------------------------------------------------------
    # FOCUSSED MAGNETISING FIELD
    # ----------------------------------------------------------------------------------------------
    def _IH_00(self, component, x, y, z):
        """
        Given component of Gaussian (0,0) mode's focussing integrals for the magnetsing field
        [complex]
        """
        ρ, φ = np.sqrt(x**2 + y**2), np.arctan2(y, x)
        if component == 'x':
            return self._I02(ρ, z) * np.sin(2*φ)
        if component == 'y':
            return self._I00(ρ, z) - self._I02(ρ, z) * np.cos(2*φ)
        if component == 'z':
            return -2j * self._I01(ρ, z) * np.sin(φ)

    def _H0_focus(self, component, x, y, z):
        """
        Given component of focussed magnetising field's complex amplitude vector
        [complex A/m]
        """
        return (self.Σ / self.Z_impedance) * self._IH_00(component, x, y, z)

    def _Hx_focus(self, x, y, z, t):
        """
        Focussed magnetising field vector's x-component
        [A/m]
        """
        return np.real(self._H0_focus('x', x, y, z) * np.exp(-1j * self.ω * t))

    def _Hy_focus(self, x, y, z, t):
        """
        Focussed magnetising field vector's y-component
        [A/m]
        """
        return np.real(self._H0_focus('y', x, y, z) * np.exp(-1j * self.ω * t))

    def _Hz_focus(self, x, y, z, t):
        """
        Focussed magnetising field vector's z-component
        [A/m]
        """
        return np.real(self._H0_focus('z', x, y, z) * np.exp(-1j * self.ω * t))

    def _magnitude_H_focus(self, x, y, z, t):
        """
        Magnitude of the focussed magnetising field vector
        [A/m]
        """
        return np.sqrt(
            self._H_focus('x', x, y, z, t)**2
            +
            self._H_focus('y', x, y, z, t)**2
            +
            self._H_focus('z', x, y, z, t)**2
        )

    def _H_focus(self, component, x, y, z, t):
        """
        Given component of the focussed magnetising field vector
        [A/m]
        """
        if component == 'x':
            return self._Hx_focus(x, y, z, t)
        if component == 'y':
            return self._Hy_focus(x, y, z, t)
        if component == 'z':
            return self._Hz_focus(x, y, z, t)
        if component == 'magnitude':
            return self._magnitude_H_focus(x, y, z, t)

    # ----------------------------------------------------------------------------------------------
    # FOCUSSED (TIME-AVERAGED) POYNTING VECTOR
    # ----------------------------------------------------------------------------------------------
    def _S_focus(self, x, y, z, t):
        """
        The (time-averaged) Poynting vector of the focussed EM fields
        """
        return 0.5 * np.real(np.cross(
            np.array([
                self._E0_focus('x', x, y, z, t),
                self._E0_focus('y', x, y, z, t),
                self._E0_focus('z', x, y, z, t),
            ]),
            np.conjugate(np.array([
                self._H0_focus('x', x, y, z, t),
                self._H0_focus('y', x, y, z, t),
                self._H0_focus('z', x, y, z, t),
            ]))
        ))

    def F_gradient(self, I):
        """
        Gradient force due to intensity of the focussed electric field on a nanosphere
        with a given dispersive polarisability
        [W/m²]
        """
        Re_α = 5e-33
        return (Re_α / (2 * ε0 * c))  * np.gradient(I)

    def F_scattering(self, S):
        """
        Scattering force due to radiation pressure of the focussed electric field on a nanosphere
        with a given dissipitave polarisability
        [W/m²]
        """
        Im_α = 4e-34
        σ_scattering = k * Im_α / ε0
        return (σ_scattering / c) * S
