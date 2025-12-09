from enum import Enum, auto

class QuantumLinearTheoryMethod(Enum):
    """
    An enumeration class for the choice of method to solve the Heisenberg-Langevin equations of motion

    >>> a single ∑ indicates summing over Cartesian axis labels jn,
    >>> a second ∑ indicates summing over the 1 or 3 possible 'on-bead' pairs
        e.g.
            for N-particle 2D case: just xy
            for N-particle 3D case: xy, xz, yz

    From the linear Hamiltonian
        H = H_free + H_interaction

        H_free = -ħΔ·a†a + ∑ ħΩjn·bjn†bjn = energy of photons + energy of phonons
            - a, a† ≡ cavity photon annihilation, creation operator
            - bjn, bjn† ≡ jn-phonon annihilation, creation operator of j-axis motion of bead n (in trap n)

        H_interaction = - ∑ Qjn·Ljn - ∑∑ gjk·Qjn·Qkn = light-mech interactions + mech-mech interactions
            - Qjn ≡ qjn / qzpf = bjn† + bjn = rescaled position operator
            - Ljn ≡ Gjn·a† + Gjn*·a = light-mech coupling operator
            - Gjn ≡ rate of exchange between jn-phonons and cavity photons
            - gjk ≡ rate of exchange between jn-phonons and kn-axis phonons

        Overall:
            H/ħ = -Δ·a†a + ∑ Ωjn·bjn†bjn - ∑ Qjn·Ljn - ∑∑ gjk·Qjn·Qkn

    Equations of motion are derived:
        Heisenberg (commutator with Hamiltonian)
        using the Canonical Commutation Relations (CCRs)
        [a, a†] = [bjn, bjn†] = 1
        +
        Langevin (minus dissipation, plus input noise)
        = (
            da/dt  = i[H/ħ, a]  - (κ/2)a  + √κ·a_in
            da†/dt = i[H/ħ, a†] - (κ/2)a† + √κ·a_in†
            ...,
            dbjn/dt  = i[H/ħ, bjn]  - (Γjn/2)bjn  + √Γjn·bjn_in
            dbjn†/dt = i[H/ħ, bjn†] - (Γjn/2)bjn† + √Γjn·bjn_in†
            ...,
        )
        = (
            da/dt  = (+iΔ - κ/2)a  + i ∑ Gjn (bjn† + bjn)  + √κ·a_in
            da†/dt = (-iΔ - κ/2)a† - i ∑ Gjn*(bjn† + bjn)  + √κ·a_in†
            ...,
            dbjn/dt  = (-iΩjn - Γjn/2)bjn  + iLjn + i ∑ gjk(bkn† + bkn) + √Γjn·bjn_in
            dbjn†/dt = (+iΩjn - Γjn/2)bjn† - iLjn - i ∑ gjk(bkn† + bkn) + √Γjn·bjn_in†
            ...,
        )

    Leading to a collective matrix equation of motion:
        dc/dt = A·c + c_in
        c ≡ vector of annihilation and creation operators
        = (
            a,
            a†
            ...
            bjn
            bjn†
            ...
        )
        c_in ≡ vector of input noises times the square root of their damping rates
        = (
            √κ·a_in
            √κ·a_in†
            ...
            √Γjn·bjn_in
            √Γjn·bjn_in†
            ...
        )
        A ≡ 'drift' matrix of couplings between them

    Solved through the Fourier transform's time-derivative relation:
        -iωc(ω) = A @ c(ω) + c_in(ω)
        c(ω) = (-iωI - A)⁻¹ @ c_in(ω)
        c(ω) = T(ω) @ c_in(ω)
        where T(ω) is the 'transformation' matrix from operator input noises to operator solutions

    >>> The first three methods rely on the maths above only.

    LINEAR_AMPLIFIER_MODEL
        T(ω) ≡ (-iωI - A)⁻¹
        Sets up the drift matrix A, and numerically inverts to get the transformation matrix T(ω)
            - Light operator solutions are then the first two entries of:
                a(ω), a†(ω) = T(ω) @ c_in(ω)[:2]
            - The mechanical operator solutions are the rest of the entries:
                ..., bjn(ω), bjn†(ω), ..., = T(ω) @ c_in(ω)[2:]

    HYBRID_LINEAR_AMPLIFIER_MODEL
        Numerically inverts for T(ω), as above, but:
            - uses only the mechanical operator solutions
            - to then solve for the light operators via their Fourier transform solution:
                a(ω)  = χc(ω)  [+ i ∑ Gjn (bjn†(ω) + bjn(ω))  + √κ·a_in(ω)]
                a†(ω) = χc*(-ω)[- i ∑ Gjn*(bjn†(ω) + bjn(ω))  + √κ·a_in†(ω)]
                where
                χc(ω) ≡ [-i(ω + Δ) + κ/2]⁻¹ ≡ optical susceptibility/response function

        Therefore, the calculation pathway compared to LINEAR_AMPLIFIER_MODEL is:
            - equivalent for all mechanical spectra (PSDs, CSDs)
            - different for optical spectra (homodyne, heterodyne, via input-output theory).

    TNT_LINEAR_AMPLIFIER_MODEL
        >>> NOTE: 2020-05-26 -- Monteiro, Toros -- Quantum sensing and cooling in three-dimensional levitated cavity optomechanics -- PRR2
        >>> Appendix A of https://doi.org/10.1103/PhysRevResearch.2.023228

        Numerically inverts for T(ω), as above, but:
            - goes directly to the 'matrix of all spectra', Scc
                Scc† = 〈c(ω)·[c(ω)]†〉= 〈T(ω) @ c_in(ω)·[T(ω) @ c_in(ω)]†〉
                Scc† = T(ω) @ N @ T(ω)†
            where
                N = 〈c_in(ω)·[c_in(ω)]†〉= diagonal matrix of quanta
                = diag(
                    κ〈a_in(ω)·[a_in(ω)]†〉= κ(np + 1)
                    κ〈a_in†(ω)·[a_in†(ω)]†〉= κ·np
                    ...,
                    Γjn〈bjn_in(ω)·[bjn_in(ω)]†〉= Γjn(njn + 1)
                    Γjn〈bjn_in†(ω)·[bjn_in†(ω)]†〉= Γjn·njn
                    ...
                )
            - np = number of cavity photons, np = 0 is quantum 'shot noise'
            - njn = number of jn-phonons, ~ kBT / ħΩjn where T ~ 300K is the thermal bath temperature

        NOTE: this does not account for input-output theory, so does not give accurate optical spectra, only PSDs

    >>> The next 2 methods rely on
        - 'tracing out' the light operators
        - relating all the mechanical operators through a 'hybridisation' matrix
        - in particular, solving for X(ω), in terms of X1D(ω)

            X(ω) = R(ω) @ X(ω) + X1D(ω)
            (I - R(ω)) @ X(ω) = X1D(ω)
            D(ω) @ X(ω) = X1D(ω)
            X(ω) = D(ω)⁻¹ @ X1D(ω)
            X(ω) = H(ω) @ X1D(ω)

        - X(ω) ≡ column vector of (rescaled) position operators
        - X1D(ω) ≡ column vector of (rescaled) 1D position operators, as noise vectors
        - R(ω) ≡ matrix of hybridisation functions, (0s on diagonal, +Rjk(ω) otherwise)
        - D(ω) ≡ I - R(ω) = derangement matrix (1s on diagonal, -Rjk(ω) otherwise)
        - I ≡ identity matrix (1s on diagonal, 0 otherwise)
        - H(ω) ≡ D(ω)⁻¹ ≡ hybridisation matrix, the linear transformation of 1D solution into full hybridisation

    NUMERICAL_HYBRIDISATION
        H(ω) ≡ D(ω)⁻¹
        This populates D(ω) with the values of -Rjk(ω), then numerically inverts it to get H(ω)

    POWER_SERIES_HYBRIDISATION
        H(ω) ≡ D(ω)⁻¹ ≡ adj(D)/det(D)
        This sets up the power series
            - to 'write out' the symbolic inversion of D(ω) i.e. the symbolic form of H(ω)
            - in terms of the hybridisation functions Rjk(ω)

        The power series are as follows
            - the deranged determinant has terms of deranged (non-fixed points) cycles, ν

                det(D) = 1 - ∑ R²ν - ∑ R³ν + ∑(-1)**c R⁴ν ... + ∑(-1)**c R**α_ν

                ν ≡ a collection of deranged (non-fixed point) cycles
                α ≡ #hybridisation functions in the term = 'order' of the term
                c ≡ number of cycles in ν

            - the deranged adjugate matrix diagonal entries are det(D) but with row j, column j removed

                adj(D)jj = det(D\j)

            - the deranged adjugate matrix off-diagonal entries of 'path-cycles' ρ ≡ ρjkν'

                adj(D)jk = Rjk + ∑ R²ρ + ∑(-1)**c' R³ρ   ... + ∑(-1)**c' R**α_ρ

                ρjk = a path from dimension j to dimension k
                ν' = deranged (non-fixed point) cycles using dimensions not in ρjk
                c' = number of cycles in ν'

    >>> The last 2 methods rely on solving the Hamiltonian with the tweezer phase dependence
        moved from the couplings to the cavity mode operators a, a†
        i.e. going from what the majority of the multitrap assumes
            (H_CS)jn = qjn(Gjn*·a + Gjn·a†)
            Gjn ≡ gjn·exp(iφn)
        to a different but equivalent form used to derive the analytic phase-dependent results
            (H_CS)jn = gjn·qjn(a†·exp(iφn) + a·exp(-iφn)) = gjn·qjn·An
            An ≡ a†·exp(iφn) + a·exp(-iφn) ≡ an† + an

        NOTE: if the tweezer phases are defined with the opposite sign convention, the only changes are
            - Gjn ≡ gjn·exp(-iφn) (for my code derivation)
            - F(±2) → F(∓ 2) (for the analytic phase dependent derivation)

        NOTE: this analytic derivation only works for x and y degrees of freedom i.e. those coupled
        to the amplitude of the cavity light under coherent scattering (not z, which is coupled to the phase)

        The shared cavity mode operators are written in terms of the phase-dependent collective modes Q, Q†
            a  = χC(ω)  [ i·gbar·Q  + √κ·a_in]
            a† = χC*(-ω)[-i·gbar·Q† + √κ·a_in†]

        The phase-dependent collective modes are defined
            Q  ≡ ∑(gjn/gbar)·exp(iφn)· Qjn
            Q† ≡ ∑(gjn/gbar)·exp(-iφn)·Qjn†

            Qjn ≡ bjn† + bjn
            gbar  ≡ √(∑ gjn²) = √N·g_RMS
            g_RMS ≡ √(∑ gjn²/N)
            N ≡ number of dimensions

            Q  = M_TOT⁻¹[Fa·a_in + Fa_dag·a_in† + FQ·Q_TOT + FQ_dag·Q_TOT†]
            Q† = M_TOT⁻¹[FFa·a_in + FFa_dag·a_in† + FFQ·Q_TOT + FFQ_dag·Q_TOT†]

            Q_TOT  ≡ (1/gbar) ∑ √Γjn·gjn·exp(iφn) ·Q_THERM
            Q_TOT† ≡ (1/gbar) ∑ √Γjn·gjn·exp(-iφn)·Q_THERM

            Q_THERM ≡ χjn(ω)·bjn_in + χjn*(-ω)·bjn_in†
            χjn(ω) ≡ [-i(ω - Ωjn) + Γjn/2]⁻¹ ≡ mechanical susceptibility/response function

            M_TOT ≡ 1 + ηm·F(0) + P = collective normalisation function
            F(n) ≡ ∑ gjn·μm(ω)·exp(iφn)
            [F(n)]† = -F(-n) since [μm(ω)]† = = μm*(-ω) = -μm(ω)
            P ≡ ξC(ω) [F(-2)·F(2) - F(0)²]
            ξC(ω) ≡ χC(ω)·χC*(-ω)

        Transduction functions from light and thermal noises to the Q, Q† modes
            e.g. from cavity input noises a_in, a_in†
                Fa      ≡ (i√κ/gbar) [χC(ω)   F(0) + P]
                FFa_dag ≡ (i√κ/gbar) [χC*(-ω) F(0) - P]

                Fa_dag ≡ (i√κ/gbar) χC*(-ω) F(2)
                FFa    ≡ (i√κ/gbar) χC(ω)   F(-2)

            e.g. from collective thermal noises Q_TOT, Q_TOT†
                FQ      ≡ 1 - χC*(-ω) F(0)
                FFQ_dag ≡ 1 + χC(ω)   F(0)

                FQ_dag ≡  χC*(-ω) F(2)
                FFQ    ≡ -χC(ω)   F(-2)

        Since [Q]† = Q†
            [Fa(ω)]† = Fa*(-ω) = FFa_dag(ω)
            [Fa_dag(ω)]† = Fa_dag*(-ω) = FFa(ω)
            [FQ(ω)]† = FQ*(-ω) = FFQ_dag(ω)
            [FQ_dag(ω)]† = FQ_dag*(-ω) = FFQ(ω)

        Therefore the absolute values of these transduction functions are related
            |Fa(ω)|² ≡ Fa(ω)·Fa*(ω) = Fa(ω)·FFa_dag(-ω)
            |FFa_dag(ω)|² ≡ FFa_dag(ω)·FFa_dag*(ω) = FFa_dag(ω)·Fa(-ω)

            |Fa_dag(ω)|² ≡ Fa_dag(ω)·Fa_dag*(ω) = Fa_dag(ω)·FFa(-ω)
            |FFa(ω)|² ≡ FFa(ω)·FFa*(ω) = FFa(ω)·Fa_dag(-ω)

            |FQ(ω)|² ≡ FQ(ω)·FQ*(ω) = FQ(ω)·FFQ_dag(-ω)
            |FFQ_dag(ω)|² ≡ FFQ_dag(ω)·FFQ_dag*(ω) = FFQ_dag(ω)·FQ(-ω)

            |FQ_dag(ω)|² ≡ FQ_dag(ω)·FQ_dag*(ω) = FQ_dag(ω)·FFQ(-ω)
            |FFQ(ω)|² ≡ FFQ(ω)·FFQ*(ω) = FFQ(ω)·FQ_dag(-ω)

    ANALYTIC_VIA_INSIDE_CAVITY
        This uses analytic expressions of the mode operators a, a† of the shared light mode
        inside the cavity as noise vectors derived in terms of the transduction functions for the
        phase-dependent collective modes Q, Q†

        The input-output operators are then applied by the code as with the other solutions to give
        heterodyne and homodyne spectra.

    ANALYTIC_DIRECT_TO_CAVITY_OUTPUT
        This uses analytic expressions of the output mode operators a_out, a_out†
        of the cavity light leaving the cavity as noise vectors
        i.e. with input-output relations already taken into account

    """

    LINEAR_AMPLIFIER_MODEL = auto()
    HYBRID_LINEAR_AMPLIFIER_MODEL = auto()
    TNT_LINEAR_AMPLIFIER_MODEL = auto()

    NUMERICAL_HYBRIDISATION = auto()
    POWER_SERIES_HYBRIDISATION = auto()

    ANALYTIC_VIA_INSIDE_CAVITY = auto()
    ANALYTIC_DIRECT_TO_CAVITY_OUTPUT = auto()

    def __repr__(self):
        """
        Return an appropriate string when printed
        """

        if self == QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL:
            return 'LAM'
        if self == QuantumLinearTheoryMethod.HYBRID_LINEAR_AMPLIFIER_MODEL:
            return 'Hybrid LAM'
        if self == QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL:
            return 'TNT LAM'

        if self == QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION:
            return 'Numerical H'
        if self == QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION:
            return 'Power Series H'

        if self == QuantumLinearTheoryMethod.ANALYTIC_VIA_INSIDE_CAVITY:
            return 'Analytic (via inside)'
        if self == QuantumLinearTheoryMethod.ANALYTIC_DIRECT_TO_CAVITY_OUTPUT:
            return 'Analytic (direct to output)'

    @property
    def is_analytic(self):
        """
        Boolean for whether the QLT method is analytic (via inside cavity operators or direct to cavity output)
        """
        return self in [SpectrumType.ANALYTIC_VIA_INSIDE_CAVITY, SpectrumType.ANALYTIC_DIRECT_TO_CAVITY_OUTPUT]
