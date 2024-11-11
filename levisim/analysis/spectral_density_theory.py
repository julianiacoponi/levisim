import numpy as np


class SpectralDensityTheory:
    """
    Spectral density definition:
    Sab(ω) =〈[a(ω)]†.b(ω)〉

    where [O(ω)]† = O†(-ω) for operators
    but [f(ω)]† = [f(ω)]* = f*(ω) for functions

    Displacement responds to a force via a response function:
    x(ω) = χx(ω) Fx(ω)
    y(ω) = χy(ω) Fy(ω)
    χ(ω) = 1 / (m(Ω²-ω² + iωΓ))

    Notably, for the harmonic oscillator's susceptibility (a.k.a. response) functions:
    χ*(-ω) = χ(ω)

    There are two force noises (units, N/√Hz), thermal and directed, whose magntiude we relate by β:
    Fx = fthx + fdx = fth(1 + β cos(ψ))
    Fy = fthy + fdy = fth(1 + β sin(ψ))
    fthx = fthy = fth
    fd = β fth


    --- AUTO-CORRELATIONS ---
    The stochastic forces (such as the thermal and directed ones considered here) are 'delta-correlated':
    〈fthj(t).fthj(t')〉= Sth.δ(t - t') - equal in x, y
    〈fd(t).fd(t')〉= Sdir.δ(t - t') - different components in x, y
    The x, y components are:
    〈fdx(t).fdx(t')〉= 〈fd(t)cos(ψ).fd(t')cos(ψ)〉= Sdir.cos²(ψ).δ(t - t') = Sdirx.δ(t - t')
    〈fdy(t).fdy(t')〉= 〈fd(t)sin(ψ).fd(t')sin(ψ)〉= Sdir.sin²(ψ).δ(t - t') = Sdiry.δ(t - t')

    This means their force spectral densities (units, N²/Hz) are constant:
    〈fthj²〉= Sthj(ω) = Sth, equal in both x, y
    〈fd²〉= Sdir(ω) = Sdir, different components in x, y
    The x, y components are:
    〈fdx²〉= Sdirx = Sdir.cos²(ψ)
    〈fdy²〉= Sdiry = Sdir.sin²(ψ)
    Sdir = β² Sth

    # TODO: clarify the difference between forces and force noises, to clarify how 〈.〉turns a force / force noise, into a force spectrum.
    This probably requires a good description Wiener-Khinchin theorem, and Langevin forces from Brownian motion


    --- CROSS-CORRELATIONS ---
    By definition, the thermal noises in x, y are uncorrelated:
    〈fthx(t).fthy(t')〉= 0

    By definition, the directed and thermal noises are uncorrelated:
    〈fth(t).fd(t')〉= 0

    So the spectral densities associated with their combination are zero.
    (NOTE: The above assumption is what is broken if one does not use two separate Wiener processes to model the two stochastic forces.)

    NOTE: Correlation functions (time-domain) and spectral densities (frequency-domain) are related via the
    Wiener-Khinchin theorem: https://en.wikipedia.org/wiki/Wiener–Khinchin_theorem


    --- POWER SPECTRAL DENSITIES (PSDs) ---
    The power spectral density is Sab(ω) for a=b
    They are real-valued, always.
    Displacement PSDs are the product of:
    - the absolute square of the response function, and
    - the force spectral densities

    PSD of x-motion:
    Sxx(ω) =〈[x(ω)]†.x(ω)〉
           = |χx(ω)|².〈(fthx + fdx)(fthx + fdx)〉
           = |χx(ω)|².〈fthx² + 2.fthx.fdx + fdx²〉
           = |χx(ω)|².(Sth + 0 + Sdirx)
           = |χx(ω)|².Sth.(1 + β² cos²(ψ))

    PSD of y-motion:
    Syy(ω) =〈[y(ω)]†.y(ω)〉
           = |χy(ω)|².〈(fthy + fdy)(fthy + fdy)〉
           = |χy(ω)|².〈fthy² + 2.fth.fdy + fdy²〉
           = |χy(ω)|².(Sth + 0 + Sdiry)
           = |χy(ω)|².Sth.(1 + β² sin²(ψ))

    |χ(ω)|² = 1 / (m²((Ω²-ω²)² + Γ²ω²))

    NOTE: This means that the force noise deduced from the PSD Fx = √(Sxx(ω))/|χx(ω)| = √(Sth.(1 + β² sin²(ψ)))
    does *not* equal the linear sum of fthx + fdx, but rather the quadrature sum √(fthx² + fdx²).


    --- CROSS-SPECTRAL DENSITIES (CSDs) ---
    The cross-correlation spectral density is defined as Sab(ω) for a≠b
    It is complex-valued, in general, with Sba(ω) = Sab(ω)*

    CSD of x-, y-motions:
    Sxy(ω) =〈[x(ω)]†.y(ω)〉
           = χxy(ω).〈(fthx + fdx)(fthy + fdy)〉
           = χxy(ω).〈fthx.fthy + fth.fdy + fdx.fth + fdx.fdy〉
           = χxy(ω).〈0 + 0 + 0 + fd².cos(ψ)sin(ψ)〉
           = χxy(ω).Sdir.cos(ψ)sin(ψ)
           = χxy(ω).Sfψ

    Displacement CSDs are the product of the cross-susceptibility and the angular force spectrum (defined below).

    "cross-susceptibility"
    χxy(ω) = χx*(ω).χy(ω)

    "angular force spectrum"
    Sfψ = Sdir.cos(ψ)sin(ψ) = ½.Sdir.sin(2ψ)
    """

    def __init__(
        self,
        m,
        Ωx, Ωy, Ωz,
        Γx, Γy, Γz,
        fth, β, ψ,
        **kwargs,
    ):
        self.m = float(m)

        self.Ωx = float(Ωx)
        self.Ωy = float(Ωy)
        self.Ωz = float(Ωz)

        self.Γx = float(Γx)
        self.Γy = float(Γy)
        self.Γz = float(Γz)

        self.fth = float(fth)  # N/√Hz
        self.β = float(β)
        self.ψ = float(ψ)

        # all spectral densities are doubled if one-sided
        self.one_sided = bool(kwargs.get('one_sided', True))

        # number of averages taken to get to a spectral density estimate
        self.nd = int(kwargs.get('nd', 1))

    @property
    def fd(self):
        """ Size of the directed force """
        return self.β * self.fth

    @property
    def fdx(self):
        """ Size of the directed force in the x-axis """
        return self.fd * np.cos(self.ψ)

    @property
    def fdy(self):
        """ Size of the directed force in the y-axis """
        return self.fd * np.sin(self.ψ)

    @property
    def fdz(self):
        """ Size of the directed force in the z-axis """
        # not modelling any directed force in z
        return 0

    @property
    def Sth(self):
        """ The thermal spectral force density """
        if self.one_sided:
            return 2 * self.fth**2
        return self.fth**2

    @property
    def Sdir(self):
        """ The directed spectral force density """
        if self.one_sided:
            return 2 * self.fd**2
        return self.fd**2

    @property
    def Sdirx(self):
        """ The directed spectral force density in the x-axis """
        if self.one_sided:
            return 2 * self.fdx**2
        return self.fdx**2

    @property
    def Sdiry(self):
        """ The directed spectral force density in the y-axis """
        if self.one_sided:
            return 2 * self.fdy**2
        return self.fdy**2

    @property
    def Sdirz(self):
        """ The directed spectral force density in the z-axis """
        if self.one_sided:
            return 2 * self.fdz**2
        return self.fdz**2

    @property
    def Sfψ(self):
        """ The angular spectral force density """
        return self.Sdir * np.cos(self.ψ) * np.sin(self.ψ)

    @property
    def Sffx(self):
        """ The total spectral force density in the x-axis """
        return self.Sth + self.Sdirx

    @property
    def Sffy(self):
        """ The total spectral force density in the y-axis """
        return self.Sth + self.Sdiry

    @property
    def Sffz(self):
        """ The total spectral force density in the z-axis """
        return self.Sth + self.Sdirz

    @property
    def γxy(self):
        """ The coherence of the x(t), y(t) signals """
        X = (self.β**2 * np.cos(self.ψ) * np.sin(self.ψ))**2
        B = 1 + self.β**2
        # coherence of x, y signals
        return np.sqrt(X / (X + B))

    # SUSCEPTIBILITES
    def χ(self, Ωj, Γj, ω):
        """ Susceptibility function for a given harmonic frequency """
        # NOTE: this means, the Fourier convention used has b = -1
        # which is common in pure maths, systems engineering and signal processing
        # https://github.com/julianiacoponi/Fourier-conventions/tree/main
        # https://mathworld.wolfram.com/FourierTransform.html
        return (self.m*(Ωj**2 - ω**2 + 1j*Γj*ω))**(-1)

    def χx(self, ω):
        """ Susceptibility function for x-motion """
        return self.χ(self.Ωx, self.Γx, ω)

    def χy(self, ω):
        """ Susceptibility function for y-motion """
        return self.χ(self.Ωy, self.Γy, ω)

    def χz(self, ω):
        """ Susceptibility function for z-motion """
        return self.χ(self.Ωz, self.Γz, ω)

    def χxy(self, ω):
        """ Cross-susceptibility function for x, y motions """
        return np.conj(self.χx(ω)) * self.χy(ω)

    # POWER SPECTRAL DENSITY FUNCTIONS
    def Sxx(self, ω):
        """ Power spectral density of the x-motion """
        return np.abs(self.χx(ω))**2 * self.Sffx

    def Syy(self, ω):
        """ Power spectral density of the y-motion """
        return np.abs(self.χy(ω))**2 * self.Sffy

    def Szz(self, ω):
        """ Power spectral density of the z-motion """
        return np.abs(self.χz(ω))**2 * self.Sffz

    # CROSS-CORRELATION SPECTRAL DENSITY FUNCTIONS
    def Sxy(self, ω):
        """ Cross-correlation spectral density of x, y motions """
        return self.χxy(ω) * self.Sdir * np.cos(self.ψ) * np.sin(self.ψ)

    def Cxy(self, ω):
        """
        Coincident, in-phase spectral density (co-spectrum) of x, y motions
        Real part of the cross-correlation
        """
        return np.real(self.Sxy(ω))

    def Qxy(self, ω):
        """
        Quadrature, out-of-phase spectral density (quad-spectrum) of x, y motions
        Negative imaginary part of the cross-correlation
        """
        return -np.imag(self.Sxy(ω))

    def θxy(self, ω):
        """ The phase angle of the cross-correlation """
        return np.angle(self.Sxy(ω))

    # STATISTICAL ERROR FUNCTIONS
    # NOTE: See Bendat + Piersol, Random Data textbook, chapter 9
    def σSxx(self, ω):
        """ Standard deviation of Sxx """
        return self.Sxx(ω) / np.sqrt(self.nd)

    def σSyy(self, ω):
        """ Standard deviation of Syy """
        return self.Syy(ω) / np.sqrt(self.nd)

    def σSxy(self, ω):
        """ Standard deviation of |Sxy| """
        return np.sqrt(self.Sxx(ω) * self.Syy(ω) / self.nd)

    def σCxy(self, ω):
        """ Standard deviation of Cxy """
        return np.sqrt(0.5 * (self.Sxx(ω) * self.Syy(ω) + self.Cxy(ω)**2 - self.Qxy(ω)**2) / self.nd)

    def σQxy(self, ω):
        """ Standard deviation of Qxy """
        return np.sqrt(0.5 * (self.Sxx(ω) * self.Syy(ω) + self.Qxy(ω)**2 - self.Cxy(ω)**2) / self.nd)
