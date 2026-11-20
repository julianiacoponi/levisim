"""
Hybridisation methods of `MultiTrap`: NUMERICAL_HYBRIDISATION and POWER_SERIES_HYBRIDISATION
(methods 4-5 of `QuantumLinearTheoryMethod`)

Both solve for the hybridised positions x from the one-dimensional solutions x1D through the
derangement matrix, x = R·x + x1D, D·x = x1D, D = I - R. They differ only in how D⁻¹ is obtained:
NUMERICAL_HYBRIDISATION inverts the (N×N) matrix D at each frequency, POWER_SERIES_HYBRIDISATION
evaluates D⁻¹ = adj(D)/det(D) from the power series of deranged cycles and path-cycles
(`levisim.multiphysics.hybridisation`), loaded from a pickle or generated at construction.
"""

import numpy as np

import os
from pathlib import Path

from levisim.core.logger import logger
from levisim.core.units import kHz, degrees
from levisim.core.utils import load_variable_from_pickle

from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.hybridisation import (
    HybridisationTerm,
    generate_all_hybridisation_functions,
    generate_determinant_and_adjugate_by_order,
    _log_power_series_term_counts,
)


__all__ = [
    'POWER_SERIES_DATA_DIR',
    'POWER_SERIES_PICKLE_DIR',
    'find_power_series_file',
    'HybridisationMixin',
]


# NOTE: the power-series pickle files are made by scripts/generate_power_series_pickles.py. The 2D,
# 4D and 6D series ship with the package, in levisim/multiphysics/power_series_data/ (this file sits
# in levisim/multiphysics/multitrap/); larger ones can be generated into a folder given by
# power_series_dir= or by the LEVISIM_POWER_SERIES_DIR environment variable
POWER_SERIES_DATA_DIR = Path(__file__).resolve().parents[1] / 'power_series_data'
# the name this folder had before the series were shipped with the package
POWER_SERIES_PICKLE_DIR = POWER_SERIES_DATA_DIR


def find_power_series_file(filename, power_series_dir=None):
    """
    The path of a power-series pickle, looked for in order in `power_series_dir`, in
    $LEVISIM_POWER_SERIES_DIR, and in the folder shipped with levisim (POWER_SERIES_DATA_DIR)

    Raises FileNotFoundError naming every folder tried, and how to generate the file.
    """
    folders = [Path(d) for d in (power_series_dir, os.environ.get('LEVISIM_POWER_SERIES_DIR')) if d]
    folders.append(POWER_SERIES_DATA_DIR)
    for folder in folders:
        candidate = folder / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f'No power-series pickle {filename!r} in {[str(f) for f in folders]}. Generate it with '
        f'`python scripts/generate_power_series_pickles.py` (from a clone of the repository), or '
        f'build the series in place with MultiTrap(generate_power_series=True).'
    )


class HybridisationMixin:
    """
    Derangement matrix D = I - R, the hybridisation functions Rjk, and D⁻¹ numerically or from the
    power series of its determinant and adjugate (methods 4-5)

    A mixin for `levisim.multiphysics.multitrap.MultiTrap`: it reads the dimensions, cavity and
    labels set up by `MultiTrapBase`.
    """
    def setup_hybridisation(self, **kwargs):
        """
        Sets up the hybridisation functions and, if asked for, the power series of the deranged
        determinant and adjugate

        Keyword arguments (from `MultiTrap`):

            - max_order: highest order of the power series (default and maximum 9, and at most the
              number of dimensions)
            - evaluation_order: order at which the power series is evaluated (default max_order)
            - load_in_power_series: load the power series found by `find_power_series_file`
              (`power_series_dir=`, then $LEVISIM_POWER_SERIES_DIR, then the shipped `POWER_SERIES_DATA_DIR`)
            - generate_power_series: generate the power series symbolically (slow above 6 dimensions)
            - compare_Rjk_calculations: check each hybridisation function against its alternative forms
        """
        self.all_Rjk = generate_all_hybridisation_functions(self.labels)

        # don't allow a larger max order than 9
        self.max_order = min(kwargs.get('max_order', 9), self.num_dimensions)
        self.evaluation_order = kwargs.get('evaluation_order', self.max_order)

        self.load_in_power_series = kwargs.get('load_in_power_series', False)
        self.generate_power_series = kwargs.get('generate_power_series', False)

        if self.load_in_power_series:
            N, d = self.num_traps, self.Cartesian_axes_per_trap
            n , max_order = N * d, self.max_order
            filename = f'{n}D_{N}x{d}_up_to_{max_order}.pickle'
            power_series_file_path = find_power_series_file(filename, kwargs.get('power_series_dir'))
            logger.info('Loading the {}D power series up to order {} from {}', n, max_order, filename)
            power_series_data = load_variable_from_pickle(power_series_file_path)
            self.deranged_determinant_by_order, self.deranged_adjugate_by_order = power_series_data
            _log_power_series_term_counts(
                self.deranged_determinant_by_order, self.deranged_adjugate_by_order, self.labels,
            )

        elif self.generate_power_series:
            # generate_determinant_and_adjugate_by_order logs its own start and finish at INFO
            self.deranged_determinant_by_order, self.deranged_adjugate_by_order = generate_determinant_and_adjugate_by_order(
                n=self.num_dimensions,
                labels=self.labels,
                max_order=self.max_order
            )

        self.compare_Rjk_calculations = kwargs.get('compare_Rjk_calculations', False)

    ################################################################################################
    # FUNCTIONS OF ONLY FREQUENCY
    # ----------------------------------------------------------------------------------------------
    def x1D_noise_vector(self, ω):
        """
        Column vector of the 1D solutions to the QLT equations of motion for each dimension, as a noise vector
        """
        x1D = np.zeros((self.num_dimensions, self.num_operators), dtype=np.complex128)
        for ii, dimension in enumerate(self.dimensions.values()):
            x1D[ii] = dimension.q1D(ω)
        return x1D

    def derangement_matrix(self, ω):
        """
        The matrix that transforms x to x1D

        ::

            x = R @ x + x1D
            D @ x = x1D
            D = I - R

        Invert it to solve the fully hybridised position vector using the 1D noise vectors ::

            x = D⁻¹ @ x1D
        """
        D_matrix = np.eye(self.num_dimensions, dtype=np.complex128)
        for jj, label_jn in enumerate(self.labels):
            for kk, label_km in enumerate(self.labels):
                if jj == kk:
                    continue
                term = HybridisationTerm(sign=-1, pairs=[(label_jn, label_km)])
                D_matrix[jj, kk] = self.hybridisation_term_value(term, ω)
        return D_matrix

    def deranged_determinant(self, ω):
        """
        The value of the deranged determinant at a given frequency
        """
        R_values = self.hybridisation_function_values(ω)
        value = 0
        for term in self.deranged_determinant_by_order[self.evaluation_order]:
            value += self.hybridisation_term_value(term, ω, R_values=R_values)
        return value

    def deranged_adjugate_matrix_entry(self, ij, ω):
        """
        The value of given adjugate entry of the derangement matrix at a given frequency
        """
        R_values = self.hybridisation_function_values(ω)
        value = 0
        for term in self.deranged_adjugate_by_order[self.evaluation_order][ij]:
            value += self.hybridisation_term_value(term, ω, R_values=R_values)
        return value

    # calculation step functions - these are calculated for a given frequency
    def hybridisation_function_values(self, ω):
        """
        All N² - N hybridisation functions at a given frequency, as {(label_jn, label_km): Rjk}

        NOTE: the power series evaluate the same N² - N functions in every one of their terms
        (n!·n factors for the determinant of order n), so they are computed once per frequency
        here and looked up by hybridisation_term_value. Each value comes from the same
        hybridisation_function_value call as before, so the sums are bit-identical.
        [complex ·]
        """
        return {
            (label_jn, label_km): self.hybridisation_function_value(label_jn, label_km, ω)
            for label_jn in self.labels
            for label_km in self.labels
            if label_jn != label_km
        }

    def hybridisation_term_value(self, term, ω, R_values=None):
        """
        The value of a hybridisation term (a signed product of hybridisation functions) at a
        given frequency

        `R_values` is an optional table from hybridisation_function_values(ω); without it each
        hybridisation function is evaluated directly.
        [complex ·]
        """
        value = complex(1)
        for pair in term.pairs:
            if R_values is None:
                Rjk = self.hybridisation_function_value(pair[0], pair[1], ω)
            else:
                Rjk = R_values[pair]
            value *= Rjk

        return term.sign * value

    def hybridisation_function_value(self, label_jn, label_km, ω):
        """
        The hybridisation function Rjn,km(ω): the response of dimension jn to dimension km, via
        the cavity (on- and off-bead) and the direct mechanical coupling gjk = `mechanical_coupling_matrix`[jn, km]
        (on-bead only)
        [complex ·]
        """

        dim_jn = self.dimensions[label_jn]
        dim_km = self.dimensions[label_km]

        j, n = label_jn[0], label_jn[1:]
        k, m = label_km[0], label_km[1:]

        Gjn = dim_jn.G
        Gkm_star = np.conjugate(dim_km.G)

        μj = dim_jn.μm(ω)
        Mj = dim_jn.M(ω)

        ηC_m = self.cavity.ηm(ω)
        ηC_p = self.cavity.ηp(ω)

        jk = j + k

        if self.compare_Rjk_calculations:
            gjn = dim_jn.g
            gkm = dim_km.g

        # the direct mechanical coupling: the rate at which km drives jn
        # (MultiTrapBase.mechanical_coupling_matrix)
        gjk_matrix = self.mechanical_coupling_matrix[self.label_to_index[label_jn], self.label_to_index[label_km]]

        # on-bead, intra-particle hybridisation
        if m == n:
            gjk = gjk_matrix

            if self.compare_Rjk_calculations:
                if jk in ['xy', 'yx']:
                    Rjk_A = (1j * μj / Mj) * (1j * ηC_m * gjn * gkm + gjk)

                elif jk in ['xz', 'yz']:
                    Rjk_A = (1j * μj / Mj) * (-ηC_p * gjn * gkm + gjk)

                elif jk in ['zx', 'zy']:
                    Rjk_A = (1j * μj / Mj) * (ηC_p * gjn * gkm + gjk)

        # off-bead, inter-particle hybridisation
        else:
            # NOTE: the power-series (and analytic) hybridisation is derived for direct mechanical
            # couplings within one bead only; inter-bead couplings are a later generalisation
            if gjk_matrix != 0:
                msg = (
                    f'mechanical_coupling_matrix couples {label_jn} to {label_km}, which belong to different'
                    ' beads: the hybridisation functions do not yet support inter-bead mechanical'
                    ' couplings (use the LINEAR_AMPLIFIER_MODEL method)'
                )
                logger.error(msg)
                raise NotImplementedError(msg)
            gjk = 0

            if self.compare_Rjk_calculations:
                φnm = dim_jn.φ - dim_km.φ

                # amplitude-amplitude coupling or phase-phase coupling
                if jk in ['xy', 'yx', 'xx', 'yy'] or jk in ['zz']:
                    Rjk_B1 = (1j * μj / Mj) * (1j * ηC_m * np.cos(φnm) + ηC_p * np.sin(φnm)) * gjn * gkm
                    ηC_φ_P = self.cavity.ηφ_P(φnm, ω)
                    Rjk_B2 = (1j * μj / Mj) * (-ηC_φ_P) * gjn * gkm

                    # NOTE: Tania's form is related to the phase form
                    # ηC_φ_T = iηC_φ_P
                    # iηC_φ_T = -ηC_φ_P
                    # ηC_φ_T = self.cavity.ηφ_T(φnm, ω)
                    # Rjk_B2 = (1j * μj / Mj) * (1j * ηC_φ_T) * gjn * gkm

                # amplitude-phase coupling
                elif jk in ['xz', 'yz']:
                    Rjk_B1 = (1j * μj / Mj) * (1j * ηC_m * np.sin(φnm) - ηC_p * np.cos(φnm)) * gjn * gkm
                    ηC_φ_A = self.cavity.ηφ_A(φnm, ω)
                    Rjk_B2 = (1j * μj / Mj) * (-ηC_φ_A) * gjn * gkm

                # phase-amplitude coupling
                elif jk in ['zx', 'zy']:
                    Rjk_B1 = (1j * μj / Mj) * (-1j * ηC_m * np.sin(φnm) + ηC_p * np.cos(φnm)) * gjn * gkm
                    ηC_φ_A = self.cavity.ηφ_A(φnm, ω)
                    Rjk_B2 = (1j * μj / Mj) * (ηC_φ_A) * gjn * gkm

        # NOTE: This expression works regardless of on-bead/off-bead
        # so long as the gjk = 0 if off-bead
        Rjk = (1j * μj / Mj) * (1j * ηC_m * np.real(Gjn * Gkm_star) + ηC_p * np.imag(Gjn * Gkm_star) + gjk)

        if self.compare_Rjk_calculations:
            # 1e-16 fails the float comparision trivially
            rtol = 1e-15
            atol = 1e-15
            debug_str = (
                f'\n{ω / kHz} kHz'
                f'\n{label_jn=}, {label_km=}'
                f'\ng{label_jn} = {dim_jn.g / kHz} kHz'
                f'\ng{label_km} = {dim_km.g / kHz} kHz'
                f'\nφ{label_jn[1:]} = {dim_jn.φ / degrees}°'
                f'\nφ{label_km[1:]} = {dim_km.φ / degrees}°'
                f'\nG{label_jn} = {Gjn / kHz} kHz'
                f'\nG{label_km}* = {Gkm_star / kHz} kHz'
            )
            if m == n:
                debug_str += f'\n{Rjk_A=}, {Rjk=}'
                assert np.isclose(Rjk_A, Rjk, rtol=rtol, atol=atol), debug_str
            else:
                debug_str_1 = debug_str + f'\n{Rjk_B1=}, {Rjk=}'
                debug_str_2 = debug_str + f'\n{Rjk_B2=}, {Rjk=}'
                debug_str_12 = debug_str + f'\n{Rjk_B1=}, {Rjk_B2=}'
                try:
                    # assert np.isclose(Rjk_B1, Rjk_B2, rtol=rtol, atol=atol), debug_str_12
                    # assert np.isclose(Rjk_B1, Rjk, rtol=rtol, atol=atol), debug_str_1
                    # assert np.isclose(Rjk_B2, Rjk, rtol=rtol, atol=atol), debug_str_2
                    assert np.isclose(Rjk_B1, Rjk, rtol=rtol, atol=atol)
                    assert np.isclose(Rjk_B2, Rjk, rtol=rtol, atol=atol)
                    assert np.isclose(Rjk_B1, Rjk_B2, rtol=rtol, atol=atol)
                except AssertionError:
                    logger.error('Hybridisation function comparison failed:{}', debug_str + f'\n{Rjk=}, {Rjk_B1=}, {Rjk_B2=}')
                    raise

        return Rjk

    ################################################################################################
    # FUNCTIONS OF METHOD AND FREQUENCY
    # ----------------------------------------------------------------------------------------------
    def hybridisation_matrix(self, method, ω):
        """
        The matrix that transforms the column vector of 1D position operators (as noise vectors)
        to the column vector of fully hybridised position operators (as noise vectors)
        i.e. the inverse of the derangement matrix ::

            x = D⁻¹ @ x1D
            via
            x = R @ x + x1D
            D @ x = x1D
            D = I - R
        """
        if method == QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION:
            # this method inverts D directly/numerically
            return np.linalg.inv(self.derangement_matrix(ω))

        if method == QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION:
            # this method inverts D symbolically
            # D⁻¹ = adj(D)/det(D)

            # det(D) = determinant of D = power series of deranged cycles ν
            # det(D) = 1 - ∑ R²ν - ∑ R³ν + ...

            # adj(D) = adjugate matrix of D = transpose of the matrix of cofactors
            # cofactor = determinant of a minor of D
            # minor = matrix left when one row and one column have been removed
            # diagonal entry adj(D)ii = det(D\i)
            # i.e. det(D) but with row i and column i removed

            # off-diagonal entry adj(D)ij = power series of 'path-cycles' ρ
            # adj(D)ij = Rij + ∑ R²ρ + ∑ R³ρ + ...

            deranged_adjugate_matrix = np.eye(self.num_dimensions, dtype=np.complex128)
            for jj, label_jn in enumerate(self.labels):
                for kk, label_km in enumerate(self.labels):
                    deranged_adjugate_matrix[jj, kk] = self.deranged_adjugate_matrix_entry(label_jn + label_km, ω)
            return deranged_adjugate_matrix / self.deranged_determinant(ω)

    def hybridisation_hybridised_positions(self, method, ω):
        """
        All hybridised position noise vectors x = D⁻¹·x1D, with D⁻¹ from `hybridisation_matrix`

        Used by NUMERICAL_HYBRIDISATION and POWER_SERIES_HYBRIDISATION. Rows are ordered as
        `self.labels`.
        """
        hybrid_positions = self.hybridisation_matrix(method, ω) @ self.x1D_noise_vector(ω)
        # https://stackoverflow.com/questions/3337301/numpy-matrix-to-array#3338368
        return np.squeeze(np.asarray(hybrid_positions))

    def power_series_position_noise_vector(self, label, ω):
        """
        The noise vector of one hybridised position from the power series, x = adj(D)/det(D)·x1D

        Only the row of the adjugate for `label` is evaluated, rather than the whole matrix.
        """
        # NOTE: only fill out the adjugate row desired, ignoring the rest of the matrix
        deranged_adjugate_matrix_row = np.zeros(self.num_dimensions, dtype=np.complex128)
        for ii, column_label in enumerate(self.labels):
            ij = label + column_label
            deranged_adjugate_matrix_row[ii] = self.deranged_adjugate_matrix_entry(ij, ω)

        # x = adj(D)/det(D) * x1D
        # https://stackoverflow.com/questions/3337301/numpy-matrix-to-array#3338368
        return np.squeeze(np.asarray((deranged_adjugate_matrix_row @ self.x1D_noise_vector(ω)) / self.deranged_determinant(ω)))
