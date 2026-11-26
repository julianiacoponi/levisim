"""
Analytic methods of `MultiTrap`: ANALYTIC_INSIDE_CAVITY and ANALYTIC_CAVITY_OUTPUT (methods 6-7 of
`QuantumLinearTheoryMethod`)

Both eliminate the cavity field in closed form, so that the light only ever sees the collective
modes Q, Q† through the response sums D(0), D(±2) and the normalisation M_norm (see
`AnalyticMixin.analytic_kernel`), and never invert an N×N matrix. They share the hybridised positions
and differ only in which end of the input-output relations is solved first: ANALYTIC_INSIDE_CAVITY
gives the intracavity light from the 𝔽 transduction functions, ANALYTIC_CAVITY_OUTPUT the output light.
"""

import numpy as np
import itertools

from functools import cached_property

from levisim.core.logger import logger


__all__ = [
    'AnalyticMixin',
]


class AnalyticMixin:
    """
    Collective coupling gbar, the closed-form response sums D(n), P and M_norm, the 𝔽 / 𝔽̄
    transduction functions, and the analytic noise vectors (methods 6-7)

    A mixin for `levisim.multiphysics.multitrap.MultiTrap`: it reads the dimensions, cavity and
    couplings set up by `MultiTrapBase`.
    """
    ################################################################################################
    # COLLECTIVE PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def gbar(self):
        """
        The total coupling rate for the phase-dependent collective modes Q,Q†
        [2π·Hz]
        """
        return np.sqrt(self.num_dimensions) * self.g_RMS

    @cached_property
    def coupling_phases(self):
        """
        Phase θjn of each dimension's complex light-mech coupling, Gjn = gjn·exp(iθjn)

        θjn = Φjn_CS + φn, where Φjn_CS = 0 for x, y (amplitude quadrature) and π/2 for z (phase
        quadrature), and φn is the phase of the tweezer trapping bead n. This does not depend on
        `tweezer_phase_sign_convention`: that sign multiplies `Dimension.φ_coupling` and multiplies
        it again inside Gjn = gjn·exp(sign·i·φ_coupling), so it cancels (see `Dimension.φ_coupling`).
        Ordered as `self.labels`.
        [rad]
        """
        return np.array([
            self.tweezer_phase_sign_convention * dim.φ_coupling
            for dim in self.dimensions.values()
        ])

    @cached_property
    def coupling_quadrature_overlap_matrix(self):
        """
        Real antisymmetric matrix I_lp = Im(Gl*·Gp) = gl·gp·sin(θp - θl) over all pairs of dimensions

        It measures how far apart (in quadrature angle) two dimensions couple to the cavity light,
        and is the only way the coupling phases enter the cavity-mediated terms that can cancel
        (P and the thermal transduction coefficients). It is evaluated from the phase difference
        so that it is EXACTLY zero for pairs that couple to the same quadrature (equal tweezer
        phases, or x and y on the same bead) - which is what keeps those terms free of
        catastrophic cancellation when one dimension is driven on resonance.
        Ordered as `self.labels` on both axes.
        [(2π·Hz)²]
        """
        θ = self.coupling_phases
        g = np.array([dim.g for dim in self.dimensions.values()])
        return np.outer(g, g) * np.sin(θ[np.newaxis, :] - θ[:, np.newaxis])

    @cached_property
    def trap_dimension_indices(self):
        """
        For each trap (in order), the indices into `self.labels` of the dimensions of its bead
        [list of int arrays]
        """
        trap_indices = [dim.trap_index for dim in self.dimensions.values()]
        return [
            np.flatnonzero(np.array(trap_indices) == trap_index)
            for trap_index in sorted(set(trap_indices))
        ]

    @cached_property
    def intra_bead_coupling_matrix(self):
        """
        Symmetric matrix of the direct mechanical couplings between dimensions of the same bead

        This is `mechanical_coupling_matrix`, which the drift matrix also uses: entry (j, k) is
        gxy, gxz or gyz when dimensions j and k belong to the same bead, and zero otherwise.
        Ordered as `self.labels` on both axes. The closed form below is derived for couplings
        within one bead that are reciprocal (gjk = gkj), so a matrix with a non-zero entry
        between different beads, or an asymmetric one, raises NotImplementedError rather than
        being silently mis-handled.
        [2π·Hz]
        """
        couplings = np.asarray(self.mechanical_coupling_matrix, dtype=float)
        trap_indices = np.array([self.dimensions[label].trap_index for label in self.labels])
        off_bead = trap_indices[:, None] != trap_indices[None, :]
        if np.any(couplings[off_bead] != 0):
            jj, kk = np.argwhere(off_bead & (couplings != 0))[0]
            msg = (
                f'mechanical_coupling_matrix couples {self.labels[jj]} to {self.labels[kk]}, which belong to'
                ' different beads: the analytic methods do not yet support inter-bead mechanical couplings'
                ' (use the LINEAR_AMPLIFIER_MODEL method)'
            )
            logger.error(msg)
            raise NotImplementedError(msg)
        if not np.array_equal(couplings, couplings.T):
            msg = (
                'mechanical_coupling_matrix is not symmetric: the analytic methods assume reciprocal'
                ' couplings gjk = gkj (use the LINEAR_AMPLIFIER_MODEL method)'
            )
            logger.error(msg)
            raise NotImplementedError(msg)
        return couplings.copy()

    ################################################################################################
    # FUNCTIONS OF ONLY FREQUENCY
    # ----------------------------------------------------------------------------------------------
    def analytic_kernel(self, ω):
        """
        Frequency-dependent building blocks shared by every analytic-method function at ω

        Eliminating the cavity field from the quantum Langevin equations ::

            a(ω)   = χC(ω)·[ i·Σ Gjn·qjn(ω) + √κ·a_in]
            a†(ω)  = χC*(-ω)·[-i·Σ Gjn*·qjn(ω) + √κ·a_in†]
            qjn(ω) = μjn(ω)·[i(Gjn*·a + Gjn·a†) + ...] + √Γjn·Q_THERMjn(ω)

        leaves, for the positions q (ordered as `self.labels`), ::

            (W⁻¹ - iC)·q = sources,   C = iχC·G*Gᵀ - iχC*·GG†  (rank 2: the two light channels)

        where W = (diag(1/μ) - i·g_intra)⁻¹ is the mechanical response matrix, μjn = χjn(ω) - χjn*(-ω),
        and g_intra holds the direct mechanical couplings gxy, gxz, gyz between the dimensions of the
        same bead (`intra_bead_coupling_matrix`). W is block diagonal per bead (W = diag(μ) when a
        bead's dimensions are uncoupled), symmetric, and only needs one (axes × axes) inverse per
        bead - the closed form never inverts an N×N matrix. Each thermal drive enters through
        B⁻¹ = W·diag(1/μ).
        Because C has rank 2, the cavity field only ever sees three collective response sums ::

            D(0) = Gᵀ·W·G*,   D(2) = Gᵀ·W·G,   D(-2) = G*ᵀ·W·G*

        and the determinant of the remaining 2×2 light-only problem is the normalisation ::

            M_norm = 1 + ηm·D(0) - ξC·P,   P = D(0)² - D(2)·D(-2).

        NOTE: P is NOT evaluated as D(0)² - D(2)·D(-2). When the frequency grid hits a mechanical
        resonance exactly, ``|gjn²·μjn|`` ~ 1e13 rad/s for the toy parameters, and that subtraction
        loses ``~ε·|D(0)|²``, i.e. ~1e-9 of P. With unequal tweezer phases this error was then
        amplified ``~|Mjn| ~ 1e7`` in the resonant dimension's position (where the thermal drive and
        the cavity backaction nearly cancel), which was the 0.2-0.6% disagreement of the analytic
        position PSDs with LAM. The identity ::

            P = 2i·hᵀ·I·h̃,   h = W·G,   h̃ = W·G*,   I = `coupling_quadrature_overlap_matrix`

        involves no such subtraction, since I is exactly zero between dimensions coupling to the
        same quadrature. The same identity gives the brackets multiplying each dimension's
        thermal noise in a and a† without cancellation::

            F_Q·G + F_Q†·G*     → B⁻ᵀ·[G  - 2i·χC*(-ω)·Iᵀ·h ]
            FF_Q·G + FF_Q†·G*   → B⁻ᵀ·[G* - 2i·χC(ω)·Iᵀ·h̃]

        (with g_intra = 0, B⁻¹ = 1 and these are exactly the brackets of PRR 8).

        The result for the last ω requested is cached, since the analytic noise vectors call
        D, P and M_norm several times per frequency.
        [dict of complex arrays and scalars]
        """
        cached = getattr(self, '_analytic_kernel_cache', None)
        if cached is not None and cached[0] == ω:
            return cached[1]

        χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
        G = self.G_array
        G_star = np.conjugate(G)
        μ = np.array([dim.μm(ω) for dim in self.dimensions.values()])
        I = self.coupling_quadrature_overlap_matrix

        # W⁻¹ = diag(1/μ) - i·g_intra per bead, and its inverse W. W is built by inverting W⁻¹
        # rather than as (I - i·diag(μ)·g_intra)⁻¹·diag(μ): on a resonance the latter matrix has a
        # condition number ~5e6 (3×2 OFF_NODE toy system), which cost ~4e-10 in W. Beads without
        # intra-bead coupling keep W = diag(μ) exactly.
        g_intra = self.intra_bead_coupling_matrix
        W = np.diag(μ)
        # B⁻¹ = W·diag(1/μ) maps each dimension's thermal drive to the bead's uncoupled response
        B_inv = np.eye(self.num_dimensions, dtype=np.complex128)
        W_inv_blocks = []
        for indices in self.trap_dimension_indices:
            block = np.ix_(indices, indices)
            W_inv_n = np.diag(1 / μ[indices]) - 1j * g_intra[block]
            W_inv_blocks.append(W_inv_n)
            if np.any(g_intra[block]):
                W[block] = np.linalg.inv(W_inv_n)
                B_inv[block] = W[block] / μ[np.newaxis, indices]

        h, h_tilde = W @ G, W @ G_star
        D0 = G @ h_tilde
        D2 = G @ h
        Dm2 = G_star @ h_tilde
        P = 2j * h @ I @ h_tilde

        kernel = dict(
            μ=μ,
            W=W,
            W_inv_blocks=W_inv_blocks,
            h=h,
            h_tilde=h_tilde,
            D={0: D0, 2: D2, -2: Dm2},
            P=P,
            M_norm=1 + self.cavity.ηm(ω) * D0 - self.cavity.ξC(ω) * P,
            # brackets multiplying each dimension's thermal noise in a and in a† (B⁻ᵀ is the identity
            # for beads without intra-bead coupling)
            thermal_a=B_inv.T @ (G - 2j * χC_star * (I.T @ h)),
            thermal_a_dag=B_inv.T @ (G_star - 2j * χC * (I.T @ h_tilde)),
        )
        self._analytic_kernel_cache = (ω, kernel)
        return kernel

    def D(self, n, ω):
        """
        Phase-dependent collective coupling and mechanical susceptibility rate

        ::

            D(0) = Gᵀ·W·G*,   D(2) = Gᵀ·W·G,   D(-2) = G*ᵀ·W·G*   (see `analytic_kernel`)

        which without intra-bead coupling (W = diag(μ)) are D(0) = Σ gjn²·μjn and
        D(±2) = Σ gjn²·μjn·exp(±2iθjn). Only n = 0, ±2 appear in the closed form.
        [complex 2π·Hz]
        """
        if n not in (0, 2, -2):
            raise ValueError(f'D(n) is only defined for n = 0, ±2, not {n=}')
        return self.analytic_kernel(ω)['D'][n]

    def P(self, ω):
        """
        Phase-dependent function that equals zero when all phases are equal

        P = D(0)² - D(2)·D(-2), evaluated in the cancellation-free form 2i·hᵀ·I·h̃ (see
        `analytic_kernel`). Without intra-bead coupling this equals
        4·Σ_(j<k) gj²·μj·gk²·μk·sin²(θj - θk), i.e. `P_via_Pjk` and `P_via_Pj`; those
        alternative forms (and M_norm_2 ... M_norm_N_2, and the K/H/U/N functions of `Dimension`)
        do NOT include intra-bead coupling.
        [complex (2π·Hz)²]
        """
        return self.analytic_kernel(ω)['P']

    def P_via_Pj(self, ω):
        """
        Alternative definition of P as a sum of contributions from each dimension
        [complex (2π·Hz)²]
        """
        return np.sum([
            dim.P(ω)
            for dim in self.dimensions.values()
        ])

    def P_via_Pj_2(self, ω):
        """
        Alternative definition of P as a sum of contributions from each dimension
        [complex (2π·Hz)²]
        """
        return np.sum([
            dim.P_2(ω)
            for dim in self.dimensions.values()
        ])

    def P_via_Pjk(self, ω):
        """
        Alternative definition of P as a sum of contributions from each pair of dimensions

        Only pairs coupling to different quadratures contribute: pairs with equal coupling
        phases θ (x and y on the same bead, or any two x/y dimensions in tweezers of equal phase)
        give sin²(0) = 0 exactly and are skipped. Pairs on the SAME bead still contribute when
        one of them is z, since z couples to the phase quadrature (θz - θx = π/2).
        [complex (2π·Hz)²]
        """
        return np.sum([
            self.Pjk(dim_pair, ω)
            for dim_pair in itertools.combinations(self.dimensions.values(), 2)
            if dim_pair[0].φ_coupling != dim_pair[1].φ_coupling
        ])

    def Pjk(self, dim_pair, ω):
        """
        Component of P contributed by a given pair of dimensions
        [complex (2π·Hz)²]
        """
        dim_jn, dim_km = dim_pair[0], dim_pair[1]
        return 4 * (
            dim_jn.g**2
            * dim_jn.μm(ω)
            * dim_km.g**2
            * dim_km.μm(ω)
            * np.sin(dim_jn.φ_coupling - dim_km.φ_coupling)**2
        )

    def M_norm(self, ω):
        """
        Normalisation function for transducing all input noises into the collective modes Q, Q†

        M_norm = 1 + ηm·D(0) - ξC·P, i.e. the determinant of the 2×2 light-only problem.
        NOTE: this used to show a larger relative difference than the alternative forms below
        because P was evaluated as D(0)² - D(2)·D(-2), which cancels catastrophically on a
        mechanical resonance (e.g. 7.8e-10 relative error at ω = Ωy2 for the 2×2
        PHASE_DEPENDENT toy system). The alternative forms sum K(ω), which is already built from
        sin² of phase differences. With P from `analytic_kernel` the error is at machine
        precision.
        [complex]
        """
        return self.analytic_kernel(ω)['M_norm']

    def M_norm_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) - self.cavity.ξC(ω) * dim.K(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_3(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) + self.cavity.χC(ω) * dim.H(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_4(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) - np.conjugate(self.cavity.χC(-ω)) * dim.H_dag(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_5(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.χC(ω) * dim.U(ω) - np.conjugate(self.cavity.χC(-ω)))
            for dim in self.dimensions.values()
        ])

    def M_norm_6(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.χC(ω) - np.conjugate(self.cavity.χC(-ω)) * dim.U_dag(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_7(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + 2j * self.cavity.ξC(ω) * np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.Δ + 1j * dim.K(ω) / 2)
            for dim in self.dimensions.values()
        ])

    def M_norm_2_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) - self.cavity.ξC(ω) * dim.K_2(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_3_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) + self.cavity.χC(ω) * dim.H_2(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_4_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) - np.conjugate(self.cavity.χC(-ω)) * dim.H_dag_2(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_5_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.χC(ω) * dim.U_2(ω) - np.conjugate(self.cavity.χC(-ω)))
            for dim in self.dimensions.values()
        ])

    def M_norm_6_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.χC(ω) - np.conjugate(self.cavity.χC(-ω)) * dim.U_dag_2(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_7_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + 2j * self.cavity.ξC(ω) * np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.Δ + 1j * dim.K_2(ω) / 2)
            for dim in self.dimensions.values()
        ])

    def M_norm_N(self, option, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * dim.N(option, ω)
            for dim in self.dimensions.values()
        ])

    def M_norm_N_2(self, option, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * dim.N_2(option, ω)
            for dim in self.dimensions.values()
        ])

    def Fa(self, ω):
        """
        Transduction function to collective mode Q
        from annihilation operator of cavity input noise photons a_in

        Q ∝ Fa·a_in
        [complex 1/√(2π·Hz)]
        """
        iκg = 1j * np.sqrt(self.cavity.κ) / self.gbar
        return iκg * (self.cavity.χC(ω) * self.D(0, ω) - self.cavity.ξC(ω) * self.P(ω))

    def FFa_dag(self, ω):
        """
        Transduction function to collective mode Q†
        from creation operator of cavity input noise photons a_in†

        Q† ∝ FFa_dag·a_in†
        [complex 1/√(2π·Hz)]
        """
        iκg = 1j * np.sqrt(self.cavity.κ) / self.gbar
        return iκg * (np.conjugate(self.cavity.χC(-ω)) * self.D(0, ω) + self.cavity.ξC(ω) * self.P(ω))

    def Fa_dag(self, ω):
        """
        Transduction function to collective mode Q
        from creation operator of cavity input noise photons a_in†

        Q ∝ Fa_dag·a_in†
        [complex 1/√(2π·Hz)]
        """
        iκg = 1j * np.sqrt(self.cavity.κ) / self.gbar
        return iκg * np.conjugate(self.cavity.χC(-ω)) * self.D(2, ω)

    def FFa(self, ω):
        """
        Transduction function to collective mode Q†

        Q† ∝ FFa·a_in
        i.e. from annihilation operator of cavity input noise photons a_in
        [complex 1/√(2π·Hz)]
        """
        iκg = 1j * np.sqrt(self.cavity.κ) / self.gbar
        return iκg * self.cavity.χC(ω) * self.D(-2, ω)

    def FQ(self, ω):
        """
        Transduction function to collective mode Q
        from annihilation operator of collective thermal noise mode Q_TOT

        Q ∝ FQ·Q_TOT
        [complex]
        """
        return 1 - np.conjugate(self.cavity.χC(-ω)) * self.D(0, ω)

    def FQ_dag(self, ω):
        """
        Transduction function to collective mode Q
        from creation operator of collective thermal noise mode Q_TOT†

        Q ∝ FQ_dag·Q_TOT†
        [complex]
        """
        return np.conjugate(self.cavity.χC(-ω)) * self.D(2, ω)

    def FFQ(self, ω):
        """
        Transduction function to collective mode Q†
        from annihilation operator of collective thermal noise mode Q_TOT

        Q† ∝ FFQ·Q_TOT
        [complex]
        """
        return -self.cavity.χC(ω) * self.D(-2, ω)

    def FFQ_dag(self, ω):
        """
        Transduction function to collective mode Q†
        from creation operator of collective thermal noise mode Q_TOT†

        Q† ∝ FFQ_dag·Q_TOT†
        [complex]
        """
        return 1 + self.cavity.χC(ω) * self.D(0, ω)

    def analytic_thermal_noise_vector(self, coefficients, ω):
        """
        Noise vector Σ cjn·√Γjn·(χjn(ω)·bjn_in + χjn*(-ω)·bjn_in†) for per-dimension coefficients cjn

        `coefficients` is ordered as `self.labels`; the light-noise entries (a_in, a_in†) are
        left at zero for the caller to fill.
        [complex 1/√(2π·Hz) × units of the coefficients]
        """
        noise_vector = np.zeros(self.num_operators, dtype=np.complex128)
        for coefficient, dim in zip(coefficients, self.dimensions.values()):
            index = dim.noise_index
            noise_vector[index] = coefficient * np.sqrt(dim.Γ) * dim.χ(ω)
            noise_vector[index + 1] = coefficient * np.sqrt(dim.Γ) * np.conjugate(dim.χ(-ω))
        return noise_vector

    ################################################################################################
    # LIGHT AND POSITION NOISE VECTORS
    # ----------------------------------------------------------------------------------------------
    def analytic_inside_cavity_light_noise_vectors(self, ω):
        """
        The intracavity light noise vectors a(ω), a†(ω) of ANALYTIC_INSIDE_CAVITY, from the
        transduction functions `Fa`, `Fa_dag`, `FFa`, `FFa_dag` and the thermal brackets of
        `analytic_kernel`
        """
        gbar, κ = self.gbar, self.cavity.κ
        χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
        M_norm_m1 = self.M_norm(ω)**(-1)
        kernel = self.analytic_kernel(ω)

        # a ∝ χC·[F_Q·Gjn + F_Q†·Gjn*]·√Γjn·(χjn·bjn_in + χjn*·bjn_in†), etc. - with the bracket
        # evaluated in the cancellation-free form of `analytic_kernel`
        a_hat = self.analytic_thermal_noise_vector(1j * χC * M_norm_m1 * kernel['thermal_a'], ω)
        a_hat[0] = χC * (1j * gbar * M_norm_m1 * self.Fa(ω) + np.sqrt(κ))
        a_hat[1] = χC * 1j * gbar * M_norm_m1 * self.Fa_dag(ω)

        a_dag = self.analytic_thermal_noise_vector(-1j * χC_star * M_norm_m1 * kernel['thermal_a_dag'], ω)
        a_dag[0] = χC_star * -1j * gbar * M_norm_m1 * self.FFa(ω)
        a_dag[1] = χC_star * (-1j * gbar * M_norm_m1 * self.FFa_dag(ω) + np.sqrt(κ))
        return a_hat, a_dag

    def analytic_cavity_output_light_noise_vectors(self, ω):
        """
        The intracavity light noise vectors a(ω), a†(ω) of ANALYTIC_CAVITY_OUTPUT, by reversing the
        input-output relations a = (a_out - a_in)/(-√κ) on `analytic_output_light_noise_vectors`
        """
        # reverse the input-output relations!
        a_out_hat, a_out_dag = self.analytic_output_light_noise_vectors(ω)
        a_out_hat[0] = a_out_hat[0] - 1
        a_out_dag[1] = a_out_dag[1] - 1
        κ = self.cavity.κ
        a_hat = a_out_hat / -np.sqrt(κ)
        a_dag = a_out_dag / -np.sqrt(κ)
        return a_hat, a_dag

    def analytic_output_light_noise_vectors(self, ω):
        """
        The output light noise vectors a_out(ω), a_out†(ω) of ANALYTIC_CAVITY_OUTPUT, in closed form
        """
        gbar, κ = self.gbar, self.cavity.κ
        χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
        M_norm_m1 = self.M_norm(ω)**(-1)
        kernel = self.analytic_kernel(ω)

        a_out_hat = self.analytic_thermal_noise_vector(
            -1j * np.sqrt(κ) * χC * M_norm_m1 * kernel['thermal_a'], ω
        )
        a_out_hat[0] = -1j * gbar * np.sqrt(κ) * χC * M_norm_m1 * self.Fa(ω) + 1 - κ * χC
        a_out_hat[1] = -1j * gbar * np.sqrt(κ) * χC * M_norm_m1 * self.Fa_dag(ω)

        a_out_dag = self.analytic_thermal_noise_vector(
            1j * np.sqrt(κ) * χC_star * M_norm_m1 * kernel['thermal_a_dag'], ω
        )
        a_out_dag[0] = 1j * gbar * np.sqrt(κ) * χC_star * M_norm_m1 * self.FFa(ω)
        a_out_dag[1] = 1j * gbar * np.sqrt(κ) * χC_star * M_norm_m1 * self.FFa_dag(ω) + 1 - κ * χC_star
        return a_out_hat, a_out_dag

    def analytic_response_matrix(self, ω):
        """
        Closed-form mechanical response matrix H ≡ (W⁻¹ - iC)⁻¹ of all dimensions, including the cavity

        (see `analytic_kernel` for W and C). The off-bead blocks follow from the Woodbury identity ::

            H = W + W·Φ·W / M_norm,   Φ = iC - 4ξC·I·W·I

        but on a bead's own block that sum cancels almost completely when one of its dimensions
        is driven on resonance (``|W| ~ |μ|`` is then up to ~1e7 times larger than the result). The
        on-bead blocks are therefore taken from the Schur complement instead, ::

            H_nn = (W_n⁻¹ - Φ_nn^(¬n) / M_norm^(¬n))⁻¹

        where ^(¬n) means the same quantity built from all the OTHER beads only, with
        Φ_nn^(¬n) = iC_nn - 4ξC·I_n,¬n·W_¬n·I_¬n,n. This is one small (axes × axes) inverse per
        bead, so the closed form never inverts an N×N matrix. For a single uncoupled dimension it
        reduces to H_jj = μj·M_norm^(¬j)/M_norm: the dimension's susceptibility, renormalised by
        the cavity with and without it.
        Ordered as `self.labels` on both axes.
        [complex (2π·Hz)⁻¹]
        """
        kernel = self.analytic_kernel(ω)
        χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
        ηm, ξC = self.cavity.ηm(ω), self.cavity.ξC(ω)
        G, G_star = self.G_array, np.conjugate(self.G_array)
        I = self.coupling_quadrature_overlap_matrix
        W, M_norm = kernel['W'], kernel['M_norm']
        h, h_tilde = kernel['h'], kernel['h_tilde']

        C = 1j * χC * np.outer(G_star, G) - 1j * χC_star * np.outer(G, G_star)
        Φ = 1j * C - 4 * ξC * (I @ W @ I)
        H = W + W @ Φ @ W / M_norm

        traps = self.trap_dimension_indices
        D0_per_trap = np.array([G[n] @ h_tilde[n] for n in traps])
        P_per_trap_pair = np.array([[2j * h[n] @ I[np.ix_(n, m)] @ h_tilde[m] for m in traps] for n in traps])
        for ii, n in enumerate(traps):
            other_traps = [jj for jj in range(len(traps)) if jj != ii]
            others = np.concatenate([np.zeros(0, dtype=int)] + [traps[jj] for jj in other_traps])
            M_norm_without_n = (
                1
                + ηm * np.sum(D0_per_trap[other_traps])
                - ξC * np.sum(P_per_trap_pair[np.ix_(other_traps, other_traps)])
            )
            Φ_nn_without_n = (
                1j * C[np.ix_(n, n)]
                - 4 * ξC * I[np.ix_(n, others)] @ W[np.ix_(others, others)] @ I[np.ix_(others, n)]
            )
            H[np.ix_(n, n)] = np.linalg.inv(kernel['W_inv_blocks'][ii] - Φ_nn_without_n / M_norm_without_n)
        return H

    def analytic_hybridisation_matrix(self, ω):
        """
        The hybridisation matrix D⁻¹ (inverse of `derangement_matrix`) from the closed form

        The derangement matrix is D = diag(μjn/Mjn)·(W⁻¹ - iC), so D⁻¹ = H·diag(Mjn/μjn) with H from
        `analytic_response_matrix`; it equals np.linalg.inv(self.derangement_matrix(ω)) without
        inverting an N×N matrix.
        Ordered as `self.labels` on both axes.
        [complex]
        """
        M_per_dimension = np.array([dim.M(ω) for dim in self.dimensions.values()])
        return self.analytic_response_matrix(ω) * (M_per_dimension / self.analytic_kernel(ω)['μ'])[np.newaxis, :]

    def analytic_hybridised_positions(self, method, ω):
        """
        All hybridised position noise vectors from the analytic closed form

        Each dimension's thermal noise reaches the positions through H·diag(1/μ) (H from
        `analytic_response_matrix`), and the cavity input noise through the light force ::

            q = W·i(G*·a + G·a†)   per unit a_in, a_in†

        written so that the resonant terms of D(0) and D(±2) do not have to cancel numerically.
        The positions are the same for both analytic methods (ANALYTIC_INSIDE_CAVITY and
        ANALYTIC_CAVITY_OUTPUT share this closed form). Rows are ordered as `self.labels`.
        [complex 1/√(2π·Hz)]
        """
        kernel = self.analytic_kernel(ω)
        κ = self.cavity.κ
        χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
        G, G_star = self.G_array, np.conjugate(self.G_array)
        I = self.coupling_quadrature_overlap_matrix
        μ, W, M_norm = kernel['μ'], kernel['W'], kernel['M_norm']

        positions = np.zeros((self.num_dimensions, self.num_operators), dtype=np.complex128)
        # cavity input noise
        positions[:, 0] = W @ (1j * χC * np.sqrt(κ) / M_norm * (G_star + 2j * χC_star * (I.T @ kernel['h_tilde'])))
        positions[:, 1] = W @ (1j * χC_star * np.sqrt(κ) / M_norm * (G + 2j * χC * (I.T @ kernel['h'])))
        # thermal noise
        thermal_response = self.analytic_response_matrix(ω) / μ[np.newaxis, :]
        for kk, dim in enumerate(self.dimensions.values()):
            index = dim.noise_index
            positions[:, index] = thermal_response[:, kk] * np.sqrt(dim.Γ) * dim.χ(ω)
            positions[:, index + 1] = thermal_response[:, kk] * np.sqrt(dim.Γ) * np.conjugate(dim.χ(-ω))
        return positions
