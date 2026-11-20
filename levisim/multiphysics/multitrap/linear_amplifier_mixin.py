"""
Linear Amplifier Model methods of `MultiTrap`: LINEAR_AMPLIFIER_MODEL, HYBRID_LINEAR_AMPLIFIER_MODEL
and TNT_LINEAR_AMPLIFIER_MODEL (methods 1-3 of `QuantumLinearTheoryMethod`)

All three solve the linearised quantum Langevin equations dc/dt = A·c + c_in for the full vector of
operators c = (a, a†, ..., bjn, bjn†, ...) by inverting (-iω·I - A) at each frequency. They take the
positions (and, for the LAM, the light operators) by slicing c; the TNT LAM builds every spectrum at
once as T·N·T†.
"""

import numpy as np

from functools import cached_property

from levisim.core.logger import logger
from levisim.core.wrappers import timer, profiler, progress_bar
from levisim.multiphysics.multitrap.base import TIMER, PROFILER, PROGRESS_BAR


__all__ = [
    'LinearAmplifierMixin',
]


class LinearAmplifierMixin:
    """
    Drift matrix A, the transfer matrix T(ω) = (-iω·I - A)⁻¹, and the noise vectors and TNT spectra
    built from them (methods 1-3)

    A mixin for `levisim.multiphysics.multitrap.MultiTrap`: it reads the dimensions, cavity, labels
    and noise correlations set up by `MultiTrapBase`.
    """
    ################################################################################################
    # SETUP
    # ----------------------------------------------------------------------------------------------
    def setup_linear_amplifier_model_matrices(self):
        """
        Setup the drift matrix (A) and input noise vector (c_in)
        to help solve the quantum Langevin equations directly
        via a Linear Amplifier Model (LAM) approach

        ::

            dc(t)/dt = A @ c(t) + c_in(t)

            Heisenberg (commutator with Hamiltonian)
            +
            Langevin (minus dissipation, plus input noise)
            = (
                da/dt  = i[H/ħ, a]  - (κ/2)a  + √κ·a_in
                da†/dt = i[H/ħ, a†] - (κ/2)a† + √κ·a_in†
                ...
                dbjn/dt  = i[H/ħ, bjn]  - (Γjn/2)bjn  + √Γjn·bjn_in
                dbjn†/dt = i[H/ħ, bjn†] - (Γjn/2)bjn† + √Γjn·bjn†_in
                ...
            )

            c ≡ vector of annihilation and creation operators
            c ≡ (a, a†, ..., bjn, bjn†, ...)^⊤

            c_in ≡ noise column vector times the square root of the damping rate
            c_in ≡ (√κ·a_in, √κ·a_in†, ..., √Γjn·bjn_in, √Γjn·bjn_in†, ...)^⊤

            A ≡ 'drift' matrix of couplings between them
        """
        self.identity_matrix = np.eye(self.num_operators)
        self.drift_matrix = np.zeros((self.num_operators, self.num_operators), dtype=np.complex128)

        Δ, κ = self.cavity.Δ, self.cavity.κ

        # the overall dampings for the nosies in the PSDs
        self.γ = np.zeros(self.num_operators)
        self.γ[0] = κ
        self.γ[1] = κ

        self.drift_matrix[0, 0] = 1j*Δ - κ/2
        self.drift_matrix[1, 1] = -1j*Δ - κ/2

        for ii in range(2, self.num_operators):
            label_jn = self.noise_index_to_label[ii]
            j, n = label_jn[0], label_jn[1:]

            dim_jn = self.dimensions[label_jn]
            Γj = dim_jn.Γ
            Ωj = dim_jn.Ω
            self.γ[ii] = Γj

            Gj = dim_jn.G
            Gj_star = np.conjugate(Gj)
            iG = 1j*Gj
            iG_star = 1j*Gj_star

            # fill out the 2 rows for the 2 light operators
            # with their couplings to the mechanical operators
            self.drift_matrix[0, ii] = iG  # a couples to bjn†, bjn
            self.drift_matrix[1, ii] = -iG_star  # a† couples to bjn†, bjn

            # annihilation operators will have an even index
            # creation operators will have an odd index
            # e.g.
            # ii = 2 corresponds to bx1 (annihilation)
            # ii = 3 corresponds to bx1† (creation)
            # ii % 2 = 0 for even numbers (False), 1 for odd numbers (True)
            ii_is_odd = bool(ii % 2)
            is_annihilation_operator = not ii_is_odd

            if is_annihilation_operator:
                self.drift_matrix[ii, 0] = iG_star
                self.drift_matrix[ii, 1] = iG
                self.drift_matrix[ii, ii] = -1j*Ωj - Γj/2
            else:
                self.drift_matrix[ii, 0] = -iG_star
                self.drift_matrix[ii, 1] = -iG
                self.drift_matrix[ii, ii] = 1j*Ωj - Γj/2

            # add the mech-to-mech couplings
            igxy = 1j*dim_jn.gxy
            igxz = 1j*dim_jn.gxz
            igyz = 1j*dim_jn.gyz
            if self.Cartesian_axes_per_trap == 2:
                if j == 'x':
                    # k = y
                    if is_annihilation_operator:
                        # takes bxn to byn,byn†
                        self.drift_matrix[ii, ii + 2] = igxy
                        self.drift_matrix[ii, ii + 3] = igxy
                    else:
                        # takes bxn† to byn,byn†
                        self.drift_matrix[ii, ii + 1] = -igxy
                        self.drift_matrix[ii, ii + 2] = -igxy
                elif j == 'y':
                    # k = x
                    if is_annihilation_operator:
                        # takes byn to bxn,bxn†
                        self.drift_matrix[ii, ii - 2] = igxy
                        self.drift_matrix[ii, ii - 1] = igxy
                    else:
                        # takes byn† to bxn,bxn†
                        self.drift_matrix[ii, ii - 3] = -igxy
                        self.drift_matrix[ii, ii - 2] = -igxy

            if self.Cartesian_axes_per_trap == 3:
                if j == 'x':
                    # k, l = y, z
                    if is_annihilation_operator:
                        # takes bxn to byn,byn†
                        self.drift_matrix[ii, ii + 2] = igxy
                        self.drift_matrix[ii, ii + 3] = igxy
                        # takes bxn to bzn,bzn†
                        self.drift_matrix[ii, ii + 4] = igxz
                        self.drift_matrix[ii, ii + 5] = igxz
                    else:
                        # takes bxn† to byn,byn†
                        self.drift_matrix[ii, ii + 1] = -igxy
                        self.drift_matrix[ii, ii + 2] = -igxy
                        # takes bxn† to bzn,bzn†
                        self.drift_matrix[ii, ii + 3] = -igxz
                        self.drift_matrix[ii, ii + 4] = -igxz
                elif j == 'y':
                    # k, l = x, z
                    if is_annihilation_operator:
                        # takes byn to bxn,bxn†
                        self.drift_matrix[ii, ii - 2] = igxy
                        self.drift_matrix[ii, ii - 1] = igxy
                        # takes byn to bzn,bzn†
                        self.drift_matrix[ii, ii + 2] = igyz
                        self.drift_matrix[ii, ii + 3] = igyz
                    else:
                        # takes byn† to bxn,bxn†
                        self.drift_matrix[ii, ii - 3] = -igxy
                        self.drift_matrix[ii, ii - 2] = -igxy
                        # takes byn† to bzn,bzn†
                        self.drift_matrix[ii, ii + 1] = -igyz
                        self.drift_matrix[ii, ii + 2] = -igyz
                elif j == 'z':
                    # k, l = x, y
                    if is_annihilation_operator:
                        # takes bzn to bxn,bxn†
                        self.drift_matrix[ii, ii - 4] = igxz
                        self.drift_matrix[ii, ii - 3] = igxz
                        # takes bzn to bxn,bxn†
                        self.drift_matrix[ii, ii - 2] = igyz
                        self.drift_matrix[ii, ii - 1] = igyz
                    else:
                        # takes bzn† to bxn,bxn†
                        self.drift_matrix[ii, ii - 5] = -igxz
                        self.drift_matrix[ii, ii - 4] = -igxz
                        # takes bzn† to byn,byn†
                        self.drift_matrix[ii, ii - 3] = -igyz
                        self.drift_matrix[ii, ii - 2] = -igyz

        # c_in ≡ (
        #   √κ·a_in
        #   √κ·a_in†
        #   ...
        #   √Γjn·bjn_in
        #   √Γjn·bjn_in†
        #   ...
        # )
        self.c_in = np.sqrt(self.γ)

        # Noise matrix
        # N ≡ diag(
        #       κ〈a_in(ω)[a_in(ω)]†
        #       κ〈a_in†(ω)[a_in†(ω)]†〉
        #       ...
        #       Γjn〈bjn_in(ω)[bjn_in(ω)]†〉
        #       Γjn〈bjn_in†(ω)[bjn_in†(ω)]†〉
        #       ...
        #   )
        #   = diag(
        #       κ(nc + 1)
        #       κ·nc
        #       ...
        #       Γjn(njn + 1)
        #       Γjn·njn
        #       ...
        #   )
        # nc ≡ photons due to optical noise inside cavity = 0 for shot noise
        # njn ≡ phonons due to jn degree of freedom ~ kB·T/ħ·Ωjn (Bose-Einstein distribution)
        self.noise_matrix = np.diag(self.γ * self.quantum_factor_per_noise_operator)

        self._check_linear_stability()

    ################################################################################################
    # FUNCTIONS OF ONLY FREQUENCY
    # ----------------------------------------------------------------------------------------------
    def T_matrix(self, ω):
        """
        The transformation matrix of input noises into operator values

        ::

            c(ω) ≡ T(ω) @ c_in(ω)
            with
            T(ω) ≡ (-iω·I - A(ω))⁻¹
            (where I is the identity matrix)

        Derived from the Fourier-transformed drift matrix relation ::

            dc(t)/dt = A(t) @ c(t) + c_in(t)
            -iω.c(ω) = A(ω) @ c(ω) + c_in(ω)
            c(ω) = (-iω·I - A(ω))⁻¹ @ c_in(ω)
        """
        return np.linalg.inv(-1j*ω * self.identity_matrix - self.drift_matrix)

    def c_noise_vector(self, ω):
        """
        Calculates the vector of operators, without summing all the noises together

        ::

            c ≡ (a, a†, ..., bjn, bjn†, ...)^⊤
            c(ω) = T(ω) @ c_in(ω)

        i.e. where each operator is a noise vector of all the input noises
        """
        c_noise = np.zeros((self.num_operators, self.num_operators), dtype=np.complex128)
        T = self.T_matrix(ω)
        for ii in range(self.num_operators):
            T_row = T[ii, :]
            # https://stackoverflow.com/questions/3337301/numpy-matrix-to-array#3338368
            T_row = np.squeeze(np.asarray(T_row))
            c_noise[ii] = T_row * self.c_in
        return c_noise

    def linear_amplifier_light_noise_vectors(self, ω):
        """
        The intracavity light noise vectors a(ω), a†(ω) of the LINEAR_AMPLIFIER_MODEL

        Goes straight to the LAM solution for the light field operators, which already accounts
        for the noises a_in, a_in† (the first two rows of c(ω) = T(ω)·c_in(ω)).
        """
        c = self.c_noise_vector(ω)
        a_hat, a_dag = c[0], c[1]
        return a_hat, a_dag

    def linear_amplifier_hybridised_positions(self, ω):
        """
        All hybridised position noise vectors qjn(ω) = bjn(ω)† + bjn(ω) from the LAM solution c(ω)

        Used by LINEAR_AMPLIFIER_MODEL and HYBRID_LINEAR_AMPLIFIER_MODEL. Rows are ordered as
        `self.labels`.
        """
        hybrid_positions = np.zeros((self.num_dimensions, self.num_operators), dtype=np.complex128)
        c = self.c_noise_vector(ω)
        for ii, label in enumerate(self.labels):
            # annihlation operator index
            ai = self.label_to_noise_index[label]
            # creation operator index
            ci = ai + 1
            # qjn(ω) = bjn(ω)† + bjn(ω)
            hybrid_positions[ii] = c[ci] + c[ai]
        return hybrid_positions

    ################################################################################################
    # TNT LINEAR AMPLIFIER MODEL: ALL SPECTRA AT ONCE
    # ----------------------------------------------------------------------------------------------
    @cached_property
    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def quantum_operator_spectral_density_array(self):
        """
        Calculate all spectra 'at once' using 'TNT' Linear Amplifier Model (TNT LAM)

        ::

            Scc†(ω) = T(ω) @ N @ T(ω)†
        """
        logger.debug('🧮 Calculating Scc†(ω) = T(ω)·N·T(ω)† via the TNT method over {} ω points', len(self.ω_range))

        N_ops = self.num_operators
        Scc_array = np.zeros((len(self.ω_range), N_ops, N_ops), dtype=np.complex128)
        for ii, ω in enumerate(self.ω_range):
            T = self.T_matrix(ω)
            Scc_array[ii] = T @ self.noise_matrix @ T.conj().T
        return Scc_array

    @cached_property
    def position_power_spectra_array(self):
        """
        Calculates an array of the PSDs of all dimensions' zpf-scaled position operators
        (see position_cross_spectra_array for all the cross spectra)
        """
        logger.debug('Extracting the zpf-scaled position power spectra from Scc†(ω)')

        Sxx_array = np.zeros((len(self.ω_range), self.num_dimensions))
        Scc_array = self.quantum_operator_spectral_density_array
        for dd, label in enumerate(self.labels):
            # annihlation operator index
            ai = self.label_to_noise_index[label]
            # creation operator index
            ci = ai + 1
            # q² ~ (b + b†)² = (b + b†)(b + b†) = b·b + b†·b† + b·b† + b†·b
            Sxx_array[:, dd] = np.real(
                Scc_array[:, ci, ci]
                + Scc_array[:, ai, ai]
                # NOTE: make sure to add the cross b·b† and b†·b contributions!
                # this fixes https://github.com/julianiacoponi/levisim/issues/1
                + Scc_array[:, ai, ci]
                + Scc_array[:, ci, ai]
            )
        return Sxx_array

    @cached_property
    def position_cross_spectra_array(self):
        """
        Calculates an array of the cross spectra of all pairs of dimensions' zpf-scaled position
        operators, from the same TNT LAM array as position_power_spectra_array

        S_qjqk(ω)[ii, jj, kk] = Σ Scc†(ω) over the 2 × 2 annihilation/creation index pairs of
        dimensions jj and kk, since qjn = bjn + bjn†. The diagonal is position_power_spectra_array
        (real), and S_qkqj = S_qjqk*. Same convention as position_cross_spectrum(label_A, label_B),
        i.e. 〈qA(ω)·qB(ω)†〉.
        [complex ·], shape (len(ω_range), num_dimensions, num_dimensions)
        """
        logger.debug('Extracting the zpf-scaled position cross spectra from Scc†(ω)')

        Scc_array = self.quantum_operator_spectral_density_array
        indices = [(self.label_to_noise_index[label], self.label_to_noise_index[label] + 1) for label in self.labels]
        Sqq_array = np.zeros((len(self.ω_range), self.num_dimensions, self.num_dimensions), dtype=np.complex128)
        for jj, (aj, cj) in enumerate(indices):
            for kk, (ak, ck) in enumerate(indices):
                Sqq_array[:, jj, kk] = (
                    Scc_array[:, aj, ak]
                    + Scc_array[:, aj, ck]
                    + Scc_array[:, cj, ak]
                    + Scc_array[:, cj, ck]
                )
        return Sqq_array
