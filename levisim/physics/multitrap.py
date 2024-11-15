import sympy
import numpy as np

from enum import Enum, auto
from copy import copy
from functools import cached_property, partial
from collections import namedtuple
from sympy.combinatorics.named_groups import SymmetricGroup

from levisim.core.constants import (
    π, ħ, kB,
    m_gas,
    shot_noise,
    node, halfnode, antinode,
)
from levisim.core.units import (
    nm,
    degrees,
    Hz, kHz,
    mbar,
)
from levisim.physics.tweezer import Tweezer
from levisim.physics.bead import Bead
from levisim.physics.gas import Gas
from levisim.physics.cavity import Cavity
from levisim.physics.trap import Trap

alphabet = 'abcdefghijklmnopqrstuvwxyz'
HybridisationTerm = namedtuple('HybridisationTerm', ['sign', 'pairs'])


def generate_subscripts_and_labels(num_traps, Cartesian_axes_per_trap):
    """
    Generates:
    - the unique subscripts for the hybridisation functions
        e.g. 'abcd' for 4 total dimensions
    - the unique labels of the form
        e.g. ['x1' 'y1', 'x2', 'y2'] for 2 particles xy
    """

    # 10 = 10 particles, x (10x1)
    # 10 = 5 particles, xy (5x2)

    # 9 = 9 particles, x (9x1)
    # 9 = 3 particles, xyz (3x3)

    # 8 = 8 particles, x (8x1)
    # 8 = 4 particles, xy (4x2)

    # 7 = 7 particles, x (7x1)

    # 6 = 6 particles, x (6x1)
    # 6 = 3 particles, xy (3x2)
    # 6 = 2 particles, xyz (2x3)

    # 5 = 5 particles, x (5x1)

    # 4 = 4 particles, x (4x1)
    # 4 = 2 particles, xy (2x2)

    # 3 = 3 particles, x (3x1)
    # 3 = 1 particle, xyz (1x3)

    # 2 = 2 particles, x (2x1)
    # 2 = 1 particle, xy (1x2)

    # 1 = 1 particle, x (1x1)

    num_total_dims = Cartesian_axes_per_trap * num_traps

    # use the alphabet for subscripts
    subscripts = alphabet[:num_total_dims]

    labels = []
    for trap_index in range(1, num_traps + 1):
        for Cartesian_index in range(Cartesian_axes_per_trap):
            label = str('xyz'[Cartesian_index]) + str(trap_index)
            labels.append(label)

    return subscripts, labels


def term_string(term):
    if not term.pairs:
        return '+1'
    if term.sign == 1:
        t_str = '+'
    else:
        t_str = '-'

    for pair in term.pairs:
        pair0_str = ''.join(pair[0])
        pair1_str = ''.join(pair[1])
        t_str = f'{t_str}R{pair0_str}{pair1_str}.'
    return t_str


def generate_deranged_terms(
    n,
    i=None, j=None,
    dimension_labels=None,
    debug=False,
):
    """
    n = overall order of system
    i,j = labels of adjugate entry
    """
    if dimension_labels is None:
        dimension_labels = list('abcdefghijklmnopqrstuvwyxz')
    dimension_labels = dimension_labels[:n]

    # determinant
    if i is None and j is None:
        det, diag = True, False
        perm_generator = SymmetricGroup(n).generate_schreier_sims()
        if debug:
            print(f'doing detD{n}')

    # adjugate entry
    if i is not None and j is not None:
        assert 0 <= i <= (n - 1)
        assert 0 <= j <= (n - 1)
        det, diag = False, i == j
        perm_generator = SymmetricGroup(n - 1).generate_schreier_sims()

        if diag:
            dim = dimension_labels[i]
            adj_ii_labels = copy(dimension_labels)
            adj_ii_labels.remove(dim)

            if debug:
                print(f'doing diagonal adjD{n}_{dim}{dim}')
                print(f'old labels {dimension_labels}')
                print(f'new labels {adj_ii_labels}')

        else:
            path_origin, path_destination = dimension_labels[i], dimension_labels[j]
            adj_ij_labels = copy(dimension_labels)
            adj_ij_labels.remove(path_origin)
            adj_ij_labels.remove(path_destination)
            adj_ij_labels.insert(0, path_origin + path_destination)

            if debug:
                print(f'doing off-diagonal adjD{n}_{path_origin}{path_destination}')
                print(f'old labels {dimension_labels}')
                print(f'new labels {adj_ij_labels}')

    all_terms = []
    while True:
        try:
            perm = next(perm_generator)
            pairs = []
            for position, label in enumerate(perm):

                # determinant and diagonal adjugate entry have deranged cycles
                if det or diag:
                    cycs = len(perm.cyclic_form)
                    sign = (-1)**cycs

                    # excludes fixed points
                    if position == label:
                        continue

                    if det:
                        labels = dimension_labels
                    elif diag:
                        labels = adj_ii_labels

                    origin = labels[position]
                    destination = labels[label]
                    pairs.append((origin, destination))

                    if debug:
                        print(f'edge {position}→{label} == {origin}→{destination}')

                # off-diagonal adjugate entry has path-cycles
                else:
                    cycs_without_0 = 0
                    for cyc in perm.cyclic_form:
                        if 0 not in cyc:
                            cycs_without_0 += 1
                    sign = (-1)**cycs_without_0

                    if position == 0 and label == 0:
                        pairs.append((path_origin, path_destination))
                        if debug:
                            print(f'edge {position}→{label} == {path_origin}→{path_destination}')

                    elif position == 0:
                        destination = adj_ij_labels[label]
                        pairs.append((path_origin, destination))
                        if debug:
                            print(f'edge {position}→{label} == {path_origin}→{destination}')

                    elif label == 0:
                        origin = adj_ij_labels[position]
                        pairs.append((origin, path_destination))
                        if debug:
                            print(f'edge {position}→{label} == {origin}→{path_destination}')

                    else:
                        # excludes fixed points
                        if position == label:
                            continue

                        origin = adj_ij_labels[position]
                        destination = adj_ij_labels[label]
                        pairs.append((origin, destination))
                        if debug:
                            print(f'edge {position}→{label} == {origin}→{destination}')

            if debug:
                print(perm.full_cyclic_form, term_string(sign, pairs))

            # NOTE: sorted could help with unit testing, but not needed for now
            # doesn't seem to affect performance that much
            # could use set() instead of list()
            # all_terms.append(HybridisationTerm(sign=sign, pairs=sorted(pairs)))

            all_terms.append(HybridisationTerm(sign=sign, pairs=pairs))

        except StopIteration:
            break

    return all_terms

class QuantumLinearTheoryMethod(Enum):
    """
    An enumeration class for the choice of method to solve the Heisenberg-Langevin equations of motion

    From the linear Hamiltonian (summing over Cartesian axis labels jn):

        H = H_free + H_interaction

        H_free = -ħΔa†.a + ∑ ħΩjn.bjn†.bjn = energy of photons + energy of phonons
            - a, a† = cavity photon annihilation, creation operator
            - bjn, bjn† = jn-phonon annihilation, creation operator of j-axis motion of bead n (in trap n)

        H_interaction = - ∑ Qjn.Ljn - ∑ gjk.Qjn.Qkn = light-mech interactions + mech-mech interactions
            - Qjn = qjn / qzpf = bjn† + bjn = rescaled position operator
            - Ljn = Gjn.a† + Gjn*.a = light-mech coupling operator
            - Gjn = rate of exchange between jn-phonons and cavity photons
            - gjk = rate of exchange between jn-phonons and kn-axis phonons

        Overall:
            H/ħ = -Δa†.a + ∑ Ωjn.bjn†.bjn - ∑ Qjn.Ljn - ∑ gjk.Qjn.Qkn

    Equations of motion are derived:
        Heisenberg (commutator with Hamiltonian)
        using the Canonical Commutation Relations (CCRs)
        [a, a†] = [bjn, bjn†] = 1
        +
        Langevin (minus dissipation, plus input noise)
        = (
            da/dt  = i[H/ħ, a]  - (κ/2)a  + √κ.a_in
            da†/dt = i[H/ħ, a†] - (κ/2)a† + √κ.a_in†
            ...,
            dbjn/dt  = i[H/ħ, bjn]  - (Γjn/2)bjn  + √Γjn.bjn_in
            dbjn†/dt = i[H/ħ, bjn†] - (Γjn/2)bjn† + √Γjn.bjn_in†
            ...,
        )
        = (
            da/dt  = (+iΔ - κ/2)a  + i ∑ Gjn (bjn† + bjn)  + √κ.a_in
            da†/dt = (-iΔ - κ/2)a† - i ∑ Gjn*(bjn† + bjn)  + √κ.a_in†
            ...,
            dbjn/dt  = (-iΩjn - Γjn/2)bjn  + iLjn + i ∑ gjk(bkn† + bkn) + √Γjn.bjn_in
            dbjn†/dt = (+iΩjn - Γjn/2)bjn† - iLjn - i ∑ gjk(bkn† + bkn) + √Γjn.bjn_in†
            ...,
        )

    Leading to a collective matrix equation of motion:
        dc/dt = A.c + c_in
        - c = vector of annihilation and creation operators
        = (
            a,
            a†,
            ...,
            bjn,
            bjn†,
            ...,
        )
        - c_in = vector of input noises times the square root of their damping rates
        = (
            √κ.a_in,
            √κ.a_in†,
            ...,
            √Γjn.bjn_in,
            √Γjn.bjn_in†,
            ...,
        )
        - A = 'drift' matrix of couplings between them

    Solved through the Fourier transform's time-derivative relation:
        -iωc(ω) = A @ c(ω) + c_in(ω)
        c(ω) = (-iωI - A)⁻¹ @ c_in(ω)
        c(ω) = T(ω) @ c_in(ω)
        where T(ω) is the 'transformation' matrix from operator input noises to operator solutions

    >>> The first three methods rely on the maths above only.

    LINEAR_AMPLIFIER_MODEL
        T(ω) = (-iωI - A)⁻¹
        Sets up the drift matrix A, and numerically inverts to get the transformation matrix T(ω)
            - Light operator solutions are then the first two entries of:
                a(ω), a†(ω) = T(ω) @ c_in(ω)[:2]
            - The mechanical operator solutions are the rest of the entries:
                ..., bjn(ω), bjn†(ω), ..., = T(ω) @ c_in(ω)[2:]

    HYBRID_LINEAR_AMPLIFIER_MODEL
        Numerically inverts for T(ω), as above, but:
            - uses only the mechanical operator solutions
            - to then solve for the light operators via their Fourier transform solution:
                a(ω)  = χc(ω)  [+ i ∑ Gjn (bjn†(ω) + bjn(ω))  + √κ.a_in(ω)]
                a†(ω) = χc*(-ω)[- i ∑ Gjn*(bjn†(ω) + bjn(ω))  + √κ.a_in†(ω)]
                where
                χc(ω) = [-i(ω + Δ) - κ/2]⁻¹ = optical susceptibility/response function

        Therefore, the calculation pathway compared to LINEAR_AMPLIFIER_MODEL is:
            - equivalent for all mechanical spectra (PSDs, CSDs)
            - different for optical spectra (homodyne, heterodyne, via input-output theory).

    TNT_LINEAR_AMPLIFIER_MODEL
        Numerically inverts for T(ω), as above, but:
            - goes directly to the 'matrix of all spectra', Scc†
                Scc† = 〈c(ω).c†(ω)〉= 〈T(ω) @ c_in(ω).[T(ω) @ c_in(ω)]†
                Scc† = T(ω) @ N @ T(ω)†
            where
                N = 〈c_in(ω).c_in†(ω)〉= diagonal matrix of quanta
                = diag(
                    κ〈a_in(ω).a_in†(ω)〉= κ.np
                    κ〈a_in†(ω).a_in†(ω)〉= κ(np + 1)
                    ...,
                    Γjn〈bjn_in(ω).bjn_in†(ω)〉= Γjn.njn
                    Γjn〈bjn_in†(ω).bjn_in†(ω)〉= Γjn(njn + 1)
                    ...
                )
            - np = number of cavity photons, np = 0 is quantum 'shot noise'
            - njn = number of jn-phonons, ~ kB.T / ħ.Ωjn where T ~ 300K is the thermal bath temperature

        NOTE: this does not account for input-output theory, so does not give accurate optical spectra, only PSDs

    >>> The last 2 methods rely on
            - 'tracing out' the light operators
            - relating all the mechanical operators through a 'hybridisation' matrix
            - in particular, solving for X(ω), in terms of X1D(ω)

                X(ω) = R(ω) @ X(ω) + X1D(ω)
                (I - R(ω)) @ X(ω) = X1D(ω)
                D(ω) @ X(ω) = X1D(ω)
                X(ω) = D(ω)⁻¹ @ X1D(ω)
                X(ω) = H(ω) @ X1D(ω)

            - X(ω) = column vector of (rescaled) position operators
            - X1D(ω) = column vector of (rescaled) 1D position operators, as noise vectors
            - R(ω) = matrix of hybridisation functions, (0s on diagonal, +Rjk(ω) otherwise)
            - D(ω) = I - R(ω) = derangement matrix (1s on diagonal, -Rjk(ω) otherwise)
            - I = identity matrix (1s on diagonal, 0 otherwise)
            - H(ω) = D(ω)⁻¹ = hybridisation matrix, the linear transformation of 1D solution into full hybridisation

        NUMERICAL_HYBRIDISATION
            H(ω) = D(ω)⁻¹
            This populates D with the values of -Rjk(ω), then numerically inverts it to get H(ω)

        POWER_SERIES_HYBRIDISATION
            H(ω) = D(ω)⁻¹ = adj(D)/det(D)
            This sets up the power series
                - to 'write out' the symbolic inversion of D(ω) i.e. the symbolic form of H(ω)
                - in terms of the hybridisation functions Rjk(ω)

            The power series are as follows
                - the deranged determinant has terms of deranged (non-fixed points) cycles, ν

                    det(D) = 1 - ∑ R²ν - ∑ R³ν + ∑(-1)**c R⁴ν ... + ∑(-1)**c R**α_ν

                    ν = a collection of deranged (non-fixed point) cycles
                    α = #hybridisation functions in the term = 'order' of the term
                    c = #cycles in ν

                - the deranged adjugate matrix diagonal entries are det(D) but with row j, column j removed

                    adj(D)jj = det(D\j)

                - the deranged adjugate matrix off-diagonal entries of 'path-cycles' ρ = ρjkν'

                    adj(D)jk = Rjk + ∑ R²ρ + ∑(-1)**c' R³ρ   ... + ∑(-1)**c' R**α_ρ

                    ρjk = a path from dimension j to dimension k
                    ν' = deranged (non-fixed point) cycles using dimensions not in ρjk
                    c' = #cycles in ν'
    """

    LINEAR_AMPLIFIER_MODEL = auto()
    HYBRID_LINEAR_AMPLIFIER_MODEL = auto()
    TNT_LINEAR_AMPLIFIER_MODEL = auto()

    NUMERICAL_HYBRIDISATION = auto()
    POWER_SERIES_HYBRIDISATION = auto()

    def __repr__(self):
        """ Return an appropriate string when printed """
        if self == QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION:
            return 'Numerical H'
        if self == QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION:
            return 'Power Series H'
        if self == QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL:
            return 'LAM'
        if self == QuantumLinearTheoryMethod.HYBRID_LINEAR_AMPLIFIER_MODEL:
            return 'Hybrid LAM'
        if self == QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL:
            return 'TNT LAM'


class MultiTrap:
    """
    Class to calculate multi-particle Quantum Linear Theory
    """
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialises a MultiTrap instance
        """
        self.num_traps = kwargs.get('num_traps', 2)
        self.Cartesian_axes_per_trap = kwargs.get('Cartesian_axes_per_trap', 2)
        self.ω_range = kwargs.get('ω_range', np.arange(-400, 400, 0.5) * kHz)

        self.num_total_dims = self.num_traps * self.Cartesian_axes_per_trap
        # we assume there is one light degree of freedom, with just 2 in the noise vector: a_in, a_in^dagger
        # for each dimension we have 2 entries in the noise vector: bjn,in, bjn,in^dagger
        self.num_operators = 2 * self.num_total_dims + 2

        # TODO: no need for subscripts really?
        self.subscripts, self.labels = generate_subscripts_and_labels(
            self.num_traps,
            self.Cartesian_axes_per_trap,
        )
        # e.g. 'c': 'z1' 1 particle, xyz
        self.sub_to_label = {sub: label for sub, label in zip(self.subscripts, self.labels)}
        # e.g. 'y2': 'd' if 2 particle, xy
        self.label_to_sub = {label: sub for sub, label in self.sub_to_label.items()}

        # + 2 to account for the two conjugate light field operators
        # e.g 'x2' will be the 7th and 8th entries in the noise vector for 2 particle, xy
        self.label_to_noise_index = {label: 2*ii + 2 for ii, label in enumerate(self.labels)}
        self.noise_index_to_label = {noise_index: label for label, noise_index in self.label_to_noise_index.items()}

        # ensure the odd indices also get assigned to the correct label
        noise_indices = list(self.noise_index_to_label.keys())
        for noise_index in noise_indices:
            self.noise_index_to_label[noise_index + 1] = self.noise_index_to_label[noise_index]
        self.noise_index_to_label[0] = 'A'
        self.noise_index_to_label[1] = 'A'

        gas_kwargs = kwargs.get('gas_kwargs', {})
        self.gas = Gas(**gas_kwargs)
        cavity_kwargs = kwargs.get('cavity_kwargs', {})
        self.cavity = Cavity(**cavity_kwargs)
        self.toy_params = kwargs.get('toy_params')
        if self.toy_params:
            self.setup_toy_parameters(**self.toy_params)
        else:
            self.setup_levitated_physics(**kwargs)
            self.setup_linearised_coherent_scattering_physics(**kwargs)
            self.dimensions = {label: Dimension(self, label) for label in self.labels}

        self.G = {label: dim.G for label, dim in self.dimensions.items()}
        self.G_array = np.array(list(self.G.values()))

        self.quantum_factor_per_noise_operator = [
            # photon annihilation operator, <a_in(ω).a_in†(ω)> = np + 1
            shot_noise + 1,
            # photon creation operator, <a_in(ω)†.a_in(ω)> = np
            shot_noise,
        ]

        for label in self.labels:
            # NOTE: this is to match Tania's original fortran's values
            # njn = 1.4e-23 * 300 / (1.05e-34 * self.dimensions[label].Ω)

            # NOTE: there is basically no difference between approx. and exact calcs
            # since kBT >> ħΩ for T = 300K
            # njn = self.dimensions[label].noise_phonons_approx
            njn = self.dimensions[label].noise_phonons_exact

            # phonon annihilation operators, <bjn_in(ω).bjn_in†(ω)> = njn + 1
            self.quantum_factor_per_noise_operator.append(njn + 1)
            # phonon creation operators, <bjn_in†(ω).bjn_in(ω)> = njn
            self.quantum_factor_per_noise_operator.append(njn)

        self.setup_linear_amplifier_model_matrices()

        self.setup_hybrid_ps = kwargs.get('setup_hybridisation_power_series', False)
        if self.setup_hybrid_ps:
            self.setup_hybridisation_power_series()

        self.compare_Rjk_calculations = kwargs.get('compare_Rjk_calculations', False)

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info(self):
        print(
            f'MultiTrap('
            f'\n    num_traps = {self.num_traps}'
            f'\n    Cartesian_axes_per_trap = {self.Cartesian_axes_per_trap}'
            f'\n    ω_range = [{self.ω_range.min() / kHz}, {self.ω_range.max() / kHz}] kHz'
            f'\n    num_ω_points = {len(self.ω_range)}'
            f'\n'
            f'\n    setup_hybridisation_power_series = {self.setup_hybrid_ps}'
            f'\n    compare_Rjk_calculations = {self.compare_Rjk_calculations}'
            f'\n'
            f'\n    id(gas) = {id(self.gas)}'
            f'\n    id(cavity) = {id(self.cavity)}'
            f'\n)'
        )
        if self.toy_params:
            self.toy_params_info

    @property
    def info_per_dimension(self):
        for dimension in self.dimensions.values():
            dimension.info

    @property
    def info_per_trap(self):
        if not self.toy_params:
            for trap in self.traps:
                trap.info

    @property
    def toy_params_info(self):
        if not self.toy_params:
            return

        if self.use_Γ_formula:
            Γ_str = (
                f'\n    Γ_formula = {self.toy_Γ_formula / Hz:.5e} Hz'
                f'\n    Γ_formula_from_Antonio_paper = (1 + π/8) m_gas〈v〉P / (kBT) / (ρr) = {self.toy_Γ_formula_from_Antonio_paper / Hz:.5e} Hz'
                f'\n    Γ_formula_from_Tania_paper = 15Pr²/(m〈v〉) = {self.toy_Γ_formula_from_Tania_paper / Hz:.5e} Hz'
                f'\n    Γ_formula_from_Tania_fortran = 16P/(πρr〈v〉) = {self.toy_Γ_formula_from_Tania_fortran / Hz:.5e} Hz'
            )
        else:
            Γ_str = (
                f'\n    Γ = {self.toy_Γ / kHz:.5e} kHz'
            )
        print(
            'toy_params = {'
            f'\n    type = {self.toy_param_model}'
            f'{self.toy_model_description}'
            f'\n    Ωz_to_Ωx_multiplier = {self.Ωz_to_Ωx_multiplier:.1f}'
            f'\n'
            f'\n    ρ = {self.toy_ρ:.1f} kg/m³'
            f'\n    r = {self.toy_r / nm:.1f} nm'
            f'\n    m = {self.toy_m:.5e} kg'
            f'\n'
            f'\n    P = {self.toy_P / mbar:.1f} mbar'
            f'\n    T = {self.toy_T:.1f} K'
            f'\n    〈v〉 = {self.toy_vth:.1f} m/s'
            f'\n'
            f'\n    all Ωx = {self.toy_Ωxs / kHz} kHz'
            f'\n    all Ωy = {self.toy_Ωys / kHz} kHz'
            f'\n    all Ωz = {self.toy_Ωzs / kHz} kHz'
            f'\n'
            f'{Γ_str}'
            f'\n'
            f'\n    gx = {self.toy_gx / kHz:.5f} kHz'
            f'\n    gy = {self.toy_gy / kHz:.5f} kHz'
            f'\n    gz = {self.toy_gz / kHz:.5f} kHz'
            f'\n'
            f'\n    gxy = {self.toy_gxy / kHz:.5f} kHz'
            f'\n    gxz = {self.toy_gxz / kHz:.5f} kHz'
            f'\n    gyz = {self.toy_gyz / kHz:.5f} kHz'
            f'\n'
            '\n}'
        )

    ################################################################################################
    # SETUP FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    def setup_levitated_physics(self, **kwargs):
        """
        Setup the physics of a levitated bead
        """
        # roughly 42 degrees
        θtw = 0.731991
        θtw = 45*degrees
        self.tweezers = [
            Tweezer(θ=θtw),
            Tweezer(wx0=950*nm, wy0=1050*nm, θ=θtw, φ=90*degrees),
            Tweezer(wx0=975*nm, wy0=1025*nm, θ=θtw, φ=20*degrees),
            Tweezer(wx0=965*nm, wy0=1015*nm, θ=θtw, φ=30*degrees),
            Tweezer(wx0=940*nm, wy0=1020*nm, θ=θtw, φ=40*degrees),
            Tweezer(wx0=930*nm, wy0=1010*nm, θ=θtw, φ=50*degrees),
            Tweezer(wx0=915*nm, wy0=1055*nm, θ=θtw, φ=60*degrees),
            Tweezer(wx0=925*nm, wy0=1045*nm, θ=θtw, φ=70*degrees),
            Tweezer(wx0=935*nm, wy0=1035*nm, θ=θtw, φ=80*degrees),
        ][:self.num_traps]

        # NOTE: all mech-mech, and z-light couplings are 0 at the node
        # for the mech-mech couplings:
        # x-y couplings depend on A0, and j-z couplings on P0
        # A0, and P0 depend on the static light field
        # the static light field sums each trap's 'drive' energy into the cavity
        # each of these cavity drive energies goes as cosΦ, due to the linearised potential Vcs
        # z-light couplings also go with cosΦ, due to the linearised potential Vcs
        Φbead = node
        self.beads = [
            Bead(r=60.1*nm, Φ=Φbead),
            Bead(r=60.2*nm, Φ=Φbead),
            Bead(r=60.3*nm, Φ=Φbead),
            Bead(r=60.4*nm, Φ=Φbead),
            Bead(r=60.5*nm, Φ=Φbead),
            Bead(r=60.6*nm, Φ=Φbead),
            Bead(r=60.7*nm, Φ=Φbead),
            Bead(r=60.8*nm, Φ=Φbead),
            Bead(r=60.9*nm, Φ=Φbead),
        ][:self.num_traps]

        # each particle/trap/tweezer has its own physics!
        self.traps = [
            Trap(
                tweezer=self.tweezers[_],
                bead=self.beads[_],
                gas=self.gas,
                trap_index= _ + 1
            )
            for _ in range(self.num_traps)
        ]

    def setup_linearised_coherent_scattering_physics(self, **kwargs):
        """
        Setup the linearised properties of a levitated bead coherent scattering into the cavity

        Linearisation splits the optical degrees of freedom into 2 parts
        α = a dominant, static, complex valued part
        a = a small, dynamic, operator valued part

        a → α + a
        a† → α* + a†
        A → A0 + (a† + a)
        P → P0 + i(a† - a)

        α = 1/2 (A0 + iP0)
        α* = 1/2 (A0 - iP0)

        To derive α (the mean static cavity photon field), consider the full Hamiltonian
        H = Hfree + Hint
        Hfree = -ħΔa†a
        NOTE: *all* tweezers are now included in a multi-particle setup!
        Ed --> Etot = Ed1 + Ed2 + ... = ∑ Ed
        Hint = ∑ Vcs = Σ -Edsinθ(cosΦ(...)...)(A ...)
        Vcs(0) = -Ed.sinθ.cosΦ.A

        Heisenberg's EoMs + dissipation
        da/dt = i[H/ħ, a] - (κ/2)a
        da/dt = (iΔ - κ/2)a + ∑ i[Vcs/ħ, a]
        static part exists at origin
        dα/dt = 0 = (iΔ - κ/2)α +  ∑ i[Vcs(0)/ħ, a]
        i[Vcs(0)/ħ, a] = -iωd.sinθ.cosΦ[A,a] = +iωd.sinθ.cosΦ, since [A,a] = -1
        dα/dt = 0 = (iΔ - κ/2)α + ∑ iωd.sinθ.cosΦ
        α = -∑ iωd.sinθ.cosΦ / (iΔ - κ/2)
        α = +∑ ωd.sinθ.cosΦ * (-Δ + iκ/2) / (Δ² + (κ/2)²)
        """

        for trap in self.traps:
            # Cavity drive energy for a particular bead in a particular tweezer trap [J]
            trap.Ed = 0.5 * trap.bead.α * trap.tweezer.εtw * self.cavity.εcav
            # Cavity drive frequency for a particular bead in a particular tweezer trap [rad/s]
            trap.ωd = trap.Ed / ħ

        # ∑ ωd.sinθ.cosΦ
        self.ωd_total = sum(
            [trap.ωd * np.sin(trap.tweezer.θ) * np.cos(trap.bead.Φ) for trap in self.traps]
        )
        α_prefactor = self.ωd_total / (self.cavity.Δ**2 + (self.cavity.κ/2)**2)
        self.Ed_total = ħ * self.ωd_total

        # Real part of the static part of the photon annihilation operator [·]
        self.cavity.αR = α_prefactor * (-self.cavity.Δ)

        # Imaginary part of the static part of the photon annihilation operator [·]
        self.cavity.αI = α_prefactor * (self.cavity.κ/2)

        # Static part of the amplitude quadrature [·]
        # A0 = α* + α = 2αR
        self.cavity.A0 = 2 * self.cavity.αR

        # Static part of the phase quadrautre [·]
        # P0 = i(α* - α) = 2αI
        self.cavity.P0 = 2 * self.cavity.αI

        # TODO: figure out how to add optical spring corrections (per tweezer?) and
        # coherent scattering corrections (overall?) to this

    def setup_toy_parameters(self, **params):
        """ Setup parameters for the model directly """
        self.toy_param_model = params.get('model')

        # NOTE: this is attempting to copy the 'Mechanical Mode Comb' paper
        # define xyz frequencies a certain span around a midpoint
        if self.toy_param_model == 'spread':
            self.ω_mid = params.get('ω_mid')
            self.ω_span = params.get('ω_span')
            low = self.ω_mid - self.ω_span / 2
            high = self.ω_mid + self.ω_span / 2
            gap = self.ω_span / self.num_traps
            self.toy_Ωxs = np.arange(low, high, gap)
            self.Ω_gap = params.get('Ω_gap')
            self.toy_Ωys = np.array([Ωx - self.Ω_gap for Ωx in self.toy_Ωxs])
            self.toy_model_description = (
                f'\n    ω_mid = {self.ω_mid / kHz:.1f} kHz'
                f'\n    ω_span = {self.ω_span / kHz:.1f} kHz'
                f'\n    Ω_gap = {self.Ω_gap / kHz:.1f} kHz'
            )

        # NOTE: this is copying Tania's 'ParametersNpair.f90'
        elif self.toy_param_model == 'pairs':
            self.first_Ωx = params.get('first_Ωx')
            self.Ω_gap = params.get('Ω_gap')
            self.toy_Ωxs = [self.first_Ωx]
            self.toy_Ωys = []
            for ii in range(self.num_traps):
                self.toy_Ωxs.append(self.toy_Ωxs[0] - (ii + 1) * self.Ω_gap)
                self.toy_Ωys.append(self.toy_Ωxs[ii] + self.Ω_gap / 10)
            self.toy_Ωxs = np.array(self.toy_Ωxs)
            self.toy_Ωys = np.array(self.toy_Ωys)
            self.toy_model_description = (
                f'\n    first_Ωx = {self.first_Ωx / kHz:.1f} kHz'
                f'\n    Ω_gap = {self.Ω_gap / kHz:.1f} kHz'
            )

        elif self.toy_param_model == 'custom':
            self.toy_Ωxs = np.array(params.get('Ωxs'))
            self.toy_Ωys = np.array(params.get('Ωys'))
            self.toy_model_description = ''

        self.Ωz_to_Ωx_multiplier = params.get('Ωz_to_Ωx_multiplier', 10)
        self.toy_Ωzs = np.array([freq / self.Ωz_to_Ωx_multiplier for freq in self.toy_Ωxs])

        Ωs_iter = {}
        Ωs_iter['x'] = iter(self.toy_Ωxs)
        if 2 <= self.Cartesian_axes_per_trap <= 3:
            Ωs_iter['y'] = iter(self.toy_Ωys)
        if self.Cartesian_axes_per_trap == 3:
            Ωs_iter['z'] = iter(self.toy_Ωzs)

        Ωs = {label: next(Ωs_iter[label[0]]) for label in self.labels}


        self.toy_ρ = params.get('ρ', 2200)  # kg/m³
        self.toy_r = params.get('r', 75*nm)
        self.toy_m = params.get('m', self.toy_ρ * (4/3) * π * self.toy_r**3)

        self.toy_P = params.get('P', self.gas.P)
        self.toy_T = params.get('T', self.gas.T)
        self.toy_vth = params.get('vth', self.gas.vth)

        # define damping rate either with formula
        self.use_Γ_formula = params.get('use_Γ_formula', False)
        if self.use_Γ_formula:

            self.toy_Γ_formula_from_Antonio_paper = (1 + π/8) * m_gas * self.toy_vth * self.toy_P / (kB * self.toy_T) / (self.toy_ρ * self.toy_r)
            self.toy_Γ_formula_from_Tania_paper = 15 * self.toy_P * self.toy_r**2 / (self.toy_m * self.toy_vth)
            self.toy_Γ_formula_from_Tania_fortran = 16 * self.toy_P / π / self.toy_vth / self.toy_ρ / self.toy_r

            # choose Tania's fortran formula for now
            self.toy_Γ_formula = self.toy_Γ_formula_from_Tania_fortran

            Γs = {label: self.toy_Γ_formula for label in self.labels}

        # or define damping rate directly
        else:
            self.toy_Γ = params.get('toy_Γ', 1e-5*Hz)
            Γs_dict = {label: self.Γ_choice for label in self.labels}
            Γs = params.get('Γs', Γs_dict)

        # define the coupling rates
        self.toy_gx = params.get('toy_gx', 0*kHz)
        self.toy_gy = params.get('toy_gy', 0*kHz)
        self.toy_gz = params.get('toy_gz', 0*kHz)

        # assuming each bead is at the node, z and mech-mech are zero
        gs_dict = {}
        for label in self.labels:
            if 'x' in label:
                gs_dict[label] = self.toy_gx
            if 'y' in label:
                gs_dict[label] = self.toy_gy
            if 'z' in label:
                gs_dict[label] = self.toy_gz
        gs = params.get('gs', gs_dict)

        self.toy_gxy = params.get('gxy', 0*kHz)
        self.toy_gxz = params.get('gxz', 0*kHz)
        self.toy_gyz = params.get('gyz', 0*kHz)

        φ1 = params.get('φ1', 0*degrees)
        φ2 = params.get('φ2', 0*degrees)
        φs_dict = {}
        for label in self.labels:
            j, n = label[0], label[1:]
            if int(n) % 2:
                φs_dict[n] = φ1
            else:
                φs_dict[n] = φ2
        φs = params.get('φs', φs_dict)

        antinode, halfnode, node = 0, π/4, π/2
        Φs_dict = {label[1:]: node for label in self.labels}
        Φs = params.get('Φs', Φs_dict)

        self.dimensions = {}
        for _ in range(self.num_total_dims):
            label = self.labels[_]
            j, n = label[0], label[1:]
            self.dimensions[label] = Dimension(
                self,
                label,
                φ=φs[n],

                T=self.toy_T,
                m=self.toy_m,

                Ω=Ωs[label],
                Γ=Γs[label],

                g=gs[label],

                gxy=self.toy_gxy,
                gxz=self.toy_gxz,
                gyz=self.toy_gyz,
            )

    def setup_linear_amplifier_model_matrices(self):
        """
        Setup the drift matrix (A) and input noise vector (c_in)
        to help solve the quantum Langevin equations directly
        via a Linear Amplifier Model (LAM) approach

        dc(t)/dt = A.c(t) + c_in(t)

        Heisenberg (commutator with Hamiltonian)
        +
        Langevin (minus dissipation, plus input noise)
        = (
            da/dt = i[H/ħ, a] - (κ/2)a + √κ.a_in
            da†/dt = i[H/ħ, a†] - (κ/2)a† + √κ.a_in†
            ...
            dbjn/dt = i[H/ħ, bjn] - (Γjn/2)bjn + √Γjn.bjn_in
            dbjn†/dt = i[H/ħ, bjn†] - (Γjn/2)bjn† + √Γjn.bjn†_in
            ...
        )

        c = (a, a†, ..., bjn, bjn†, ...)^T = vector of annihilation and creation operators

        c_in
        = (√κ.a_in, √κ.a_in†, ..., √Γjn.bjn_in, √Γjn.bjn_in†, ...)^T
        = noise column vector times the square root of the damping rate

        A = matrix of couplings between them
        """
        N = self.num_operators
        self.identity_matrix = np.matrix(np.eye(self.num_operators))
        self.drift_matrix = np.matrix(np.zeros((N, N), dtype=np.complex128))

        Δ, κ = self.cavity.Δ, self.cavity.κ

        # the overall dampings for the nosies in the PSDs
        self.γ = np.zeros(N)
        self.γ[0] = κ
        self.γ[1] = κ

        self.drift_matrix[0, 0] = 1j*Δ - κ/2
        self.drift_matrix[1, 1] = -1j*Δ - κ/2

        for ii in range(2, N):
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

        # (√κ.a_in, √κ.a_in†, √Γx1.bx1_in, √Γx1.bx1_in†, ...)
        self.c_in = np.sqrt(self.γ)

        # Noise matrix
        # N = diag(
        #       κ.<a_in(ω),a_in†(ω)>,
        #       κ<a_in†(ω),a_in(ω)>,
        #       ...,
        #       Γjn<bjn_in(ω),bjn_in†(ω)>,
        #       Γjn<bjn_in†(ω),bjn_in(ω)>,
        #       ...,
        #   )
        #   = diag(
        #       κ.(np + 1),
        #       κ.np,
        #       ...,
        #       Γjn.(njn + 1),
        #       Γjn.njn,
        #       ...,
        #   )
        # np = photons due to optical noise = 0 for shot noise
        # njn = phonons due to jn degree of freedom ~ kBT/ħΩjn (Bose-Einstein distribution)
        self.noise_matrix = np.matrix(np.diag(self.γ * self.quantum_factor_per_noise_operator))

    def setup_hybridisation_power_series(self):
        """ Setup the determinant and adjugate terms for the hybridisation matrix """
        self.deranged_determinant_terms = generate_deranged_terms(
            self.num_total_dims,
            dimension_labels=self.labels,
        )
        self.deranged_adjugate_terms = {}
        for row_index, row_label in enumerate(self.labels):
            for column_index, column_label in enumerate(self.labels):
                ij = row_label + column_label
                self.deranged_adjugate_terms[ij] = generate_deranged_terms(
                    self.num_total_dims,
                    row_index, column_index,
                    dimension_labels=self.labels,
                )

    ################################################################################################
    # FREQUENCY FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    # Linear Amplifier Model matrices
    # ----------------------------------------------------------------------------------------------
    def T_matrix(self, ω):
        """
        The transformation matrix of input noises into operator values
        dc(t)/dt = drift @ c(t) + c_in
        dc(t)/dt = -iω.c(ω)
        so
        c(ω) = T(ω) @ c_in
        where
        T(ω) = (-iω.I - drift)⁻¹
        """
        return np.linalg.inv(-1j*ω * self.identity_matrix - self.drift_matrix)

    def c_noise_vector(self, ω):
        """
        Calculates the vector of operators, without summing all the noises together
        c = (a, a†, bx1, bx1†, ...)
        c(ω) = T(ω) @ c_in
        i.e. where each operator is a vector of all the input noises
        """
        c_noise = np.zeros((self.num_operators, self.num_operators), dtype=np.complex128)
        for ii in range(self.num_operators):
            T_row = self.T_matrix(ω)[ii, :]
            # https://stackoverflow.com/questions/3337301/numpy-matrix-to-array#3338368
            T_row = np.squeeze(np.asarray(T_row))
            c_noise[ii] = T_row * self.c_in
        return c_noise

    # ----------------------------------------------------------------------------------------------
    # Derangement and Hybridisation matrices
    # ----------------------------------------------------------------------------------------------
    def derangement_matrix(self, ω):
        """
        The matrix that transforms x to X1D
        x = R @ x + X1D
        D @ x = X1D
        D = I - R

        Invert it to solve the fully hybridised position vector using the 1D noise vectors
        x = D⁻¹ @ X1D
        """
        D_mat = np.matrix(np.eye(self.num_total_dims), dtype=np.complex128)
        for jj, label_jn in enumerate(self.labels):
            for kk, label_km in enumerate(self.labels):
                if jj == kk:
                    continue
                term = HybridisationTerm(sign=-1, pairs=[(label_jn, label_km)])
                D_mat[jj, kk] = self.hybridisation_term_value(term, ω)
        return D_mat

    # calculation step functions - these are calculated for a given frequency
    # TODO: make this a partial() function? i.e. precalculate all N^2 - N of them
    def hybridisation_term_value(self, term, ω):
        """ The value of a hybridisation term at a given frequency """
        value = complex(1)
        for pair in term.pairs:
            label_jn, label_km = pair

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

            # on-bead, intra-particle hybridisation
            if m == n:
                if jk in ['xy', 'yx']:
                    gxy = dim_jn.gxy
                    gxy_copy = dim_km.gxy
                    assert gxy == gxy_copy
                    gjk = gxy

                if jk in ['xz', 'zx']:
                    gxz = dim_jn.gxz
                    gxz_copy = dim_km.gxz
                    assert gxz == gxz_copy
                    gjk = gxz

                if jk in ['yz', 'zy']:
                    gyz = dim_jn.gyz
                    gyz_copy = dim_km.gyz
                    assert gyz == gyz_copy
                    gjk = gyz

                if self.compare_Rjk_calculations:
                    if jk in ['xy', 'yx']:
                        Rjk_A = (1j * μj / Mj) * (1j * ηC_m * gjn * gkm + gjk)

                    elif jk in ['xz', 'yz']:
                        Rjk_A = (1j * μj / Mj) * (-ηC_p * gjn * gkm + gjk)

                    elif jk in ['zx', 'zy']:
                        Rjk_A = (1j * μj / Mj) * (ηC_p * gjn * gkm + gjk)

            # off-bead, inter-particle hybridisation
            else:
                # no mech-mech coupling if not in the same tweezer trap
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
                        #  ηC_φ_T = self.cavity.ηφ_T(φnm, ω)
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
                    except AssertionError as error:
                        print(debug_str + f'\n{Rjk=}, {Rjk_B1=}, {Rjk_B2=}')
                        raise error()

            value *= Rjk

        return term.sign * value

    def deranged_determinant(self, ω):
        """ The value of the deranged determinant at a given frequency """
        value = 0
        for ii, term in enumerate(self.deranged_determinant_terms):
            value += self.hybridisation_term_value(term, ω)
        return value

    def deranged_adjugate_entry(self, ij, ω):
        """The value of given adjugate entry of the derangement matrix at a given frequency """
        value = 0
        for ii, term in enumerate(self.deranged_adjugate_terms[ij]):
            value += self.hybridisation_term_value(term, ω)
        return value

    def hybridisation_matrix(self, method, ω):
        """
        The matrix that transforms x to X1D
        X = R @ X + X1D
        D @ X = X1D
        D = I - R

        Invert it to solve the fully hybridised position vector using the 1D noise vectors
        X = D⁻¹ @ X1D
        """
        # numerical method simply inverts D directly
        if method == QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION:
            return np.linalg.inv(self.derangement_matrix(ω))

        if method == QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION:
            # 'power series' method inverts D symbolically
            # D⁻¹ = adj(D)/det(D)

            # det(D) = determinant of D
            # power series of deranged cycles = 1 - ∑ R²ν - ∑ R³ν + ...

            # adj(D) = adjugate matrix of D = tranpose of the matrix of cofactors
            # cofactor = determinant of a minor of D
            # minor = matrix left when one row and one column have been removed
            # diagonal entry adj(D)ii = det(D\i), i.e. det(D) but with row i and column i removed
            # off-diagonal entry adj(D)ij = power series of 'path-cycles' ρ = Rij + ∑ R²ρ + ∑ R³ρ + ...
            adjugate_matrix_ω = np.matrix(np.eye(self.num_total_dims), dtype=np.complex128)
            for jj, label_jn in enumerate(self.labels):
                for kk, label_km in enumerate(self.labels):
                    adjugate_matrix_ω[jj, kk] = self.deranged_adjugate_entry(label_jn + label_km, ω)
            return adjugate_matrix_ω / self.deranged_determinant(ω)

    def X1D_noise_vector(self, ω):
        """
        Column vector of the 1D solutions to the QLT equations of motion for each dimension, as a noise vector
        """
        X1D = np.zeros((self.num_total_dims, self.num_operators), dtype=np.complex128)
        for ii, dimension in enumerate(self.dimensions.values()):
            X1D[ii] = dimension.Q1D(ω)
        return X1D

    # ----------------------------------------------------------------------------------------------
    # Method dependent functions
    # ----------------------------------------------------------------------------------------------
    def hybridised_X(self, method, ω):
        """
        Column vector of the solutions to the QLT equations of motion for each dimension
        Each entry in the column vector is a noise vector
        """
        if (
            method == QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
            or
            method == QuantumLinearTheoryMethod.HYBRID_LINEAR_AMPLIFIER_MODEL
        ):
            hybrid_X = np.zeros((self.num_total_dims, self.num_operators), dtype=np.complex128)
            c = self.c_noise_vector(ω)
            for ii, label in enumerate(self.labels):
                # annihlation operator index
                ai = self.label_to_noise_index[label]
                # creation operator index
                ci = ai + 1
                # Qjn = bjn† + bjn
                hybrid_X[ii] = c[ci] + c[ai]
            return hybrid_X

        # hybridisation matrix methods
        return self.hybridisation_matrix(method, ω) @ self.X1D_noise_vector(ω)

    def light_noise_vectors(self, method, ω):
        """
        The light field inside the cavity calcualted via either
        - Numerical inversion of derangement matrix
        - Symbolic inversion of derangement matrix
        - Linear Amplifier Model (LAM)
        """

        if method == QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL:
            # goes straight to the LAM solution for the light field operators
            # already accounts for noises a_in, a_in†
            c = self.c_noise_vector(ω)
            a_hat, a_dag = c[0], c[1]

        else:
            # a = χC (i ∑GX  + √κ a_in)
            # a† = χC* (-i ∑GX + √κ a_in†)

            # this is the light coupling multiplied by the position operator (as a noise vector)
            # i.e. ∑GX = (Gx1 * x1) + (Gy1 * y1) + ...
            # each jn=x1,y1,... is a noise vector

            χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
            κ = self.cavity.κ
            hybrid_X = self.hybridised_X(method, ω)
            if method == QuantumLinearTheoryMethod.HYBRID_LINEAR_AMPLIFIER_MODEL:
                # calculates the hybridised_X using LAM, and then uses the QLT Fourier solution
                # this is already an array (c_noise_vector converts it from matrix)
                a_hat = χC * 1j * self.G_array @ hybrid_X
                a_dag = χC_star * -1j * np.conjugate(self.G_array) @ hybrid_X
            else:
                a_hat = χC * 1j * self.G_array * hybrid_X
                a_dag = χC_star * -1j * np.conjugate(self.G_array) * hybrid_X
                # convert from matrix back into a 1D array, so that indexing later works ok
                # https://stackoverflow.com/questions/3337301/numpy-matrix-to-array#3338368
                a_hat = np.squeeze(np.asarray(a_hat))
                a_dag = np.squeeze(np.asarray(a_dag))

            # adds cavity-filtered noise contribution
            # χC (√κ a_in)
            a_hat[0] += χC * np.sqrt(κ)
            # χC* (√κ a_in†)
            a_dag[1] += χC_star * np.sqrt(κ)

        return a_hat, a_dag

    def output_light_noise_vectors(self, method, ω):
        """
        Account for input-output theory to determine output light field
        a_out = -√κ a + a_in
        a_out† = -√κ a† + a_in†
        """
        a_hat, a_dag = self.light_noise_vectors(method, ω)

        κ = self.cavity.κ
        a_out_hat = -np.sqrt(κ) * a_hat
        a_out_dag = -np.sqrt(κ) * a_dag

        a_out_hat[0] = a_out_hat[0] + 1
        a_out_dag[1] = a_out_dag[1] + 1

        return a_out_hat, a_out_dag

    def homodyne_noise_vector(self, method, ω):
        """
        The homodyne quadrature of the output light, defined by the homodyne angle of the cavity
        A_out(ω) = a†_out(ω) + a_out(ω)
        P_out(ω) = i(a†_out(ω) - a_out(ω))
        θ = homodyne phase angle
        S_hom(ω) = a†_out(ω).exp(iθ) + a_out(ω).exp(-iθ) = A_out(ω).cos(θ) + P_out(ω).sin(θ)
        [1/√Hz]
        """
        a_out_hat, a_out_dag = self.output_light_noise_vectors(method, ω)

        # A_out = a_out_dag + a_out_hat
        # P_out = 1j*(a_out_dag - a_out_hat)
        # return A_out * np.cos(self.cavity.θhom) + P_out * np.sin(self.cavity.θhom)
        # return a_out_dag * np.exp(1j * self.cavity.θhom) + a_out_hat * np.exp(-1j * self.cavity.θhom)

        # NOTE: Tania defines the opposite phase quadratrue P = i(a-a†)
        A_out = a_out_hat + a_out_dag
        P_out = 1j*(a_out_hat - a_out_dag)
        return A_out * np.cos(self.cavity.θhom) + P_out * np.sin(self.cavity.θhom)

    def heterodyne_noise_vector(self, method, ω):
        """
        The heterodyne quadrature of the output light
        S_het(ω) = S_aa†(ω + Ω) + S_a†a(ω - Ω)
        [1/√Hz]
        """
        # if self.cavity.Ωhet:
            # TODO: this should be a sum of CSDs
            # a_out_hat_p, a_out_dag_p = self.output_light_noise_vectors(method, ω + self.cavity.Ωhet)
            # a_out_hat_m, a_out_dag_m = self.output_light_noise_vectors(method, ω - self.cavity.Ωhet)
            # return a_out_hat_p * a_out_dag_p + a_out_dag_m * a_out_hat_m

        a_out_hat, a_out_dag = self.output_light_noise_vectors(method, ω)
        # TODO: clarify which vector for heterodyne...
        # technically heterodyne is sum of two CSDs, S_aa†(ω + Ω) + S_a†a(ω - Ω)
        # return a_out_hat * a_out_dag + a_out_dag * a_out_hat
        return a_out_hat

    def noise_vector_to_power_spectrum(self, noise_vector):
        """
        Convert a noise vector into its power spectrum
        """
        return np.sum(self.quantum_factor_per_noise_operator * np.abs(noise_vector)**2)

    ################################################################################################
    # SPECTRUM FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    def calculate_all_hybridised_position_spectra(self, method):
        """
        Invert the derangement matrix D to get the hybridisation matrix H
        This solves the n-dimensional hybridised position vector, x
        in terms of the 1D noise vectors, X1D
        D @ X = X1D
        H = D⁻¹
        X = H @ X1D

        Calculate PSDs based on the squares of x as a noise vector
        """
        print(f'Calculating all PSDs in frequency range [{min(self.ω_range) / kHz}, {max(self.ω_range) / kHz}]  kHz using {method}')

        if method == QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL:
            return self.position_power_spectra_array

        num_ω_points = len(self.ω_range)
        N = self.num_operators
        hybrid_X_array = np.zeros((num_ω_points, self.num_total_dims, self.num_operators), dtype=np.complex128)
        psd_array = np.zeros((num_ω_points, self.num_total_dims))

        for ii, ω in enumerate(self.ω_range):
            if not ii % 100:
                print(f'{ω / kHz} kHz')
            # fill out x, as a noise vector
            hybrid_X_array[ii] = self.hybridised_X(method, ω)

            # for each dimension in x, calculate the psd
            for dd in range(self.num_total_dims):
                psd_array[ii, dd] = self.noise_vector_to_power_spectrum(hybrid_X_array[ii, dd])

        return psd_array

    def calculate_homodyne_spectrum(self, method):
        """
        Calculates homodyne spectrum of output light from the cavity, S_hh
        h = a†_out.exp(iθ) + a_out.exp(-iθ) = A_out.cos(θ) + P_out.sin(θ)
        θ = homodyne angle
        """
        print(f'Calculating homodyne spectrum in frequency range [{min(self.ω_range) / kHz}, {max(self.ω_range) / kHz}] kHz using {method}')
        homodyne_spectrum = []
        for ii, ω in enumerate(self.ω_range):
            if not ii % 100:
                print(f'{ω / kHz} kHz')
            homodyne_spectrum.append(self.noise_vector_to_power_spectrum(self.homodyne_noise_vector(method, ω)))
        return np.array(homodyne_spectrum)

    def calculate_heterodyne_spectrum(self, method):
        """
        Calculates the heterodyne spectrum of output light from the cavity, S_het
        S_het(ω) = S_aa†(ω + Ω) + S_a†a(ω - Ω)
        """
        print(f'Calculating heterodyne spectrum in frequency range [{min(self.ω_range) / kHz}, {max(self.ω_range) / kHz}] kHz using {method}')
        heterodyne_spectrum = []
        for ii, ω in enumerate(self.ω_range):
            if not ii % 100:
                print(f'{ω / kHz} kHz')
            heterodyne_spectrum.append(self.noise_vector_to_power_spectrum(self.heterodyne_noise_vector(method, ω)))
        return np.array(heterodyne_spectrum)

    # ----------------------------------------------------------------------------------------------
    # TODO: fix these remainining functions
    # ----------------------------------------------------------------------------------------------
    def hybridised_position_operator(self, label, ω):
        """A specific hybridised position operator (a specific entry in fully hybridised x) """
        # fill out the adjugate row desired
        adj_row = np.zeros(self.num_total_dims, dtype=np.complex128)
        for ii, column_label in enumerate(self.labels):
            ij = label + column_label
            adj_row[ii] = self.deranged_adjugate_entry(ij, ω)

        # X = adj(D)/det(D) * X1D
        value = (adj_row @ self.X1D_noise_vector(ω)) / self.deranged_determinant(ω)
        return value

    @cached_property
    def quantum_operator_spectral_density_array(self):
        """
        Calculate all the spectra 'at once' using 'TNT' Linear Amplifier Model (TNT LAM)
        S_cc†(ω) = T(ω) @ N @ T(ω)†
        """
        # TODO: figure out which values in this output matrix translate to which PSDs, CSDs...
        N = self.num_operators
        num_ω_points = len(self.ω_range)
        Scc_array = np.zeros((num_ω_points, N, N), dtype=np.complex128)
        for ii, ω in enumerate(self.ω_range):
            if not ii % 100:
                print(f'{ω / kHz} kHz')
            T = self.T_matrix(ω)
            Scc_array[ii] = T @ self.noise_matrix @ T.H
        return Scc_array

    @cached_property
    def position_power_spectra_array(self):
        SXX_array = np.zeros((len(self.ω_range), self.num_total_dims))
        Scc_array = self.quantum_operator_spectral_density_array
        for dd, label in enumerate(self.labels):
            # annihlation operator index
            ai = self.label_to_noise_index[label]
            # creation operator index
            ci = ai + 1
            # SXX_array[:, dd] = Scc_array[:, ci, ci] + Scc_array[:, ai, ai]
            SXX_array[:, dd] = np.real(Scc_array[:, ci, ci] + Scc_array[:, ai, ai])
        return SXX_array

    def calculate_power_spectrum(self, label, zpf_units=True):
        """ Calculates the power spectrum of a given dimension """
        print(f'Calculating PSD({label})')

        j, n = label[0], label[1:]
        zpf = self.dimensions[label_A].zpf if not zpf_units else 1
        psd = []
        for ω in self.ω_range:
            value = self.noise_vector_to_power_spectrum(self.hybridised_position_operator(label, ω))
            psd.append(value)
        return np.array(psd) * zpf**2

    def calculate_cross_spectrum(self, label_A, label_B, zpf_units=True):
        """ Calculates the cross correlation spectrum of the given dimensions """
        print(f'Calculating CSD({label_A},{label_B})')

        j, n = label_A[0], label_A[1:]
        zpf_A = self.dimensions[label_A].zpf if not zpf_units else 1

        k, m = label_B[0], label_B[1:]
        zpf_B = self.dimensions[label_B].zpf if not zpf_units else 1

        csd = []
        for ω in self.ω_range:
            position_A = self.hybridised_position_operator(label_A, ω)
            position_B = np.conjugate(self.hybridised_position_operator(label_B, ω))
            position_AB = position_A * position_B
            value = np.sum(self.quantum_factor_per_noise_operator * position_AB)
            csd.append(value)
        return np.array(csd) * zpf_A * zpf_B


# TODO: move this to levisim somehow?
class Dimension:
    """ Class to represent a mechanical degree of freedom in the system """
    def __init__(self, multitrap, label, **kwargs):

        # the multi-dimensional context in which this dimension exists!
        self.multitrap = multitrap

        # should be a string with either of 'xyz' as the first character
        # remaining characters are the bead index
        self.label = label
        self.axis, self.trap_index = label[0], label[1:]
        self.sub = multitrap.label_to_sub[label]
        # needed to know to assign Q1D
        self.noise_index = multitrap.label_to_noise_index[label]

        ############################################################################################
        # TOY MODEL PARAMETERS
        # ------------------------------------------------------------------------------------------
        # Relative phase of the tweezer associated with this dimension
        self.φ = kwargs.get('φ')

        self.T = kwargs.get('T')
        self.m = kwargs.get('m')

        # self.Ω = kwargs.get('Ω', self.trap.Ω_CS(self.axis))
        self.Ω = kwargs.get('Ω')
        self.Γ = kwargs.get('Γ')

        self.g = kwargs.get('g')
        self.gxy = kwargs.get('gxy')
        self.gxz = kwargs.get('gxz')
        self.gyz = kwargs.get('gyz')

        ############################################################################################
        # PHYSICAL MODEL PARAMETERS
        # ------------------------------------------------------------------------------------------
        self.Φ = None
        if not self.multitrap.toy_params:
            # e.g. label 'x1' will take traps[0], 'y3' takes traps[2]
            self.trap = multitrap.traps[int(self.trap_index) - 1]
            # Trapping position of the bead in the cavity's standing wave
            # [rad]
            self.Φ = self.trap.bead.Φ

        if self.φ is None:
            self.φ = self.trap.tweezer.φ

        if self.T is None:
            self.T = multitrap.gas.T

        if self.m is None:
            self.m = self.trap.bead.m

        if self.Ω is None:
            self.Ω = self.trap.Ωj[self.axis]

        if self.Γ is None:
            self.Γ = self.trap.Γth

        # Light-mech coupling goes H/ħ = -XL
        # X = b† + b
        # L = g.an† + g*.an
        # g = gAn + igPn
        # L = gAn.An + gPn.Pn
        # An = (an† + an)
        # Pn = i(an† - an)

        # a†,a = creation, annihilation operator for photons in the single cavity mode
        # an†,an = creation, annihilation operator for photons per-tweezer, 'rotated' cavity mode
        # φ = per tweezer phase
        # an = a.exp(-iφ)
        # an† = a†.exp(iφ)

        # L = g.a†.exp(iφ) + g*.a.exp(-iφ)
        # L = G.a† + G*.a = GA.A + GP.P
        # G = g.exp(iφ)
        # G* = g*.exp(-iφ)

        # L = GA.A + GP.P
        # A = (a† + a)
        # P = i(a† - a)
        # G = GA + iGP

        if self.g is None:
            self.g = self.light_mech_coupling

        if self.gxy is None:
            if multitrap.Cartesian_axes_per_trap == 1:
                self.gxy = 0
                self.gxz = 0
                self.gyz = 0

            if multitrap.Cartesian_axes_per_trap == 2:
                self.gxy = self.mech_mech_coupling('xy')
                self.gxz = 0
                self.gyz = 0

            if self.gxz is None and self.gyz is None:
                if multitrap.Cartesian_axes_per_trap == 3:
                    self.gxy = self.mech_mech_coupling('xy')
                    self.gxz = self.mech_mech_coupling('xz')
                    self.gyz = self.mech_mech_coupling('yz')

        # all coupling rates apart from gz should be negative
        for coupling in [
            self.gxy,
            self.gxz,
            self.gyz,
        ]:
            assert coupling <= 0, f'(gxy, gxz, gyz) = ({self.gxy / kHz}, {self.gxz / kHz}, {self.gyz / kHz}) kHz'

        if self.label[0] in ['x', 'y']:
            assert self.g <= 0, f'g{label} = {self.g / kHz} kHz'
        elif self.label[0] == 'z':
            assert self.g >= 0, f'g{label} = {self.g / kHz} kHz'

        ############################################################################################
        # DERIVED PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Linear coupling of (x,y,z)-motion to the cavity light quadratures with tweezer phase included
        # [rad/s]
        if not self.φ:
            arg = self.CS_coupling_phase
            # NOTE: only sympy gives exact cos(π/2) = 0
            self.G = self.g * (float(sympy.cos(arg)) + 1j * float(sympy.sin(arg)))
        else:
            self.G = self.g * np.exp(1j * (self.CS_coupling_phase + self.φ))

        # ------------------------------------------------------------------------------------------
        # Number of phonons due to the thermal bath, via the Bose-Einstein distribution
        # [·]
        ħΩ = ħ * self.Ω
        kBT = kB * self.T
        self.noise_phonons_exact = (np.exp(ħΩ / kBT) - 1)**(-1)
        self.noise_phonons_approx = kBT / ħΩ

        # TODO: derive this properly?
        Δ, κ = self.multitrap.cavity.Δ, self.multitrap.cavity.κ
        Cp = 1 / ((Δ + self.Ω)**2 + (κ/2)**2)
        Cm = 1 / ((Δ - self.Ω)**2 + (κ/2)**2)
        self.cooling_rate = κ * np.abs(self.G)**2 * (Cp - Cm)
        self.cooling_factor = self.Γ / (self.Γ + self.cooling_rate)
        self.noise_phonons_with_cooling = self.noise_phonons_approx * self.cooling_factor
        self.T_with_cooling = ħΩ * self.noise_phonons_with_cooling / kB

        self.zpf = np.sqrt(ħ/2 / (self.m * self.Ω))

    def __repr__(self):
        return f'Dimension({self.label})'

    @property
    def info(self):
        """ Prints information on the dimension """
        Φ_str = '' if not self.Φ else f'\n    (Φ{self.trap_index}/2π) = (x0/λ) = {self.Φ / (2*π):.5f}'

        print(
            f'Dimension('
            f'\n    jn = label = {self.label}'
            f'\n    subscript = {self.sub}'
            f'\n    j = axis = {self.axis}'
            f'\n    n = trap_index = {self.trap_index}'
            f'\n    noise_index = {self.noise_index}'
            f'\n'
            f'\n    φ{self.trap_index} = {self.φ / degrees:.5f} °'
            f'{Φ_str}'
            f'\n'
            f'\n    Ω{self.label} = {self.Ω / kHz:.5f} kHz'
            f'\n    Γ{self.label} = {self.Γ / Hz:.5e} Hz'
            f'\n'
            f'\n    n{self.label} = {self.noise_phonons_exact:.5e} phonons (exact)'
            f'\n    n{self.label} = {self.noise_phonons_approx:.5e} phonons (approx.)'
            f'\n'
            f'\n    cooling_rate = {self.cooling_rate / Hz:.5e} Hz'
            f'\n    cooling_factor = {self.cooling_factor:.5e}'
            f'\n    cooled n{self.label} = {self.noise_phonons_with_cooling:.5e} phonons'
            f'\n    cooled T{self.label} = {self.T_with_cooling:.5e} K'
            f'\n'
            f'\n    g{self.label} = {self.g / kHz:.5f} kHz'
            f'\n    G{self.label} = {self.G / kHz:.5f} kHz'
            f'\n'
            f'\n    gxy = {self.gxy / kHz:.5f} kHz'
            f'\n    gxz = {self.gxz / kHz:.5f} kHz'
            f'\n    gyz = {self.gyz / kHz:.5f} kHz'
            f'\n'
            f'\n    {self.label}_zpf = {self.zpf:.5e} m'
            f'\n)'
        )

    def zero_point_fluctuation(self, axis):
        """
        Zero-point fluctuation of the position of the oscillator, per axis
        [m]
        """
        # TODO: sort CS frequency corrections
        # return np.sqrt(ħ/2 / (self.bead.m * self.Ω_CS(axis)))
        return np.sqrt(ħ/2 / (self.trap.bead.m * self.trap.Ωj[axis]))

    @property
    def CS_coupling_phase(self):
        """
        Phase of linear coherent scattering coupling of (x,y,z)-motion to the cavity light
        C(Φ) = Acos(Φ) + Psin(Φ) = a†.exp(iΦ) + a.exp(-iΦ)
        - Φ = 0 couples to amplitude quadrature, C(0) = A = a† + a
        - Φ = π/2 couples to phase quadrature, C(π/2) = P = i(a† - a)
        [rad]
        """
        return {
            'x': 0,
            'y': 0,
            'z': π/2 if self.φ else sympy.pi / 2,
            'z': π/2,
        }[self.axis]

    @property
    def light_mech_coupling(self):
        """
        Linear coupling of (x,y,z)-motion to the cavity light quadratures
        [rad/s]
        """
        kc, θtw = self.trap.tweezer.k, self.trap.tweezer.θ
        ωdsinθ = self.trap.ωd * np.sin(θtw)
        Φ = self.trap.bead.Φ
        zpf = self.zero_point_fluctuation(self.axis)

        # x, y couple to amplitude, have negative signs
        if self.axis == 'x':
            return -ωdsinθ * kc * zpf * np.sin(Φ) * np.sin(θtw)

        if self.axis == 'y':
            return -ωdsinθ * kc * zpf * np.sin(Φ) * np.cos(θtw)

        # z couples to the phase quadrature, has positive sign!
        if self.axis == 'z':
            kz = (kc - 1 / self.trap.tweezer.zR)
            return ωdsinθ * kz * zpf * np.cos(Φ)

    def mech_mech_coupling(self, axes):
        """
        Linear coupling between 2 axes of motion
        [rad/s]
        """
        kc, θtw = self.trap.tweezer.k, self.trap.tweezer.θ
        Φ = self.trap.bead.Φ

        A0, P0 = self.multitrap.cavity.A0, self.multitrap.cavity.P0
        ωdsinθ = self.trap.ωd * np.sin(θtw)

        # all have negative signs
        if axes == 'xy':
            x_zpf = self.zero_point_fluctuation(axes[0])
            y_zpf = self.zero_point_fluctuation(axes[1])
            return -A0 * ωdsinθ * (kc * x_zpf) * (kc * y_zpf) * np.sin(θtw) * np.cos(θtw) * np.cos(Φ)

        if axes == 'xz':
            x_zpf = self.zero_point_fluctuation(axes[0])
            z_zpf = self.zero_point_fluctuation(axes[1])
            kz = (kc - 1 / self.trap.tweezer.zR)
            return -P0 * ωdsinθ * (kc * x_zpf) * (kz * z_zpf) * np.sin(θtw) * np.sin(Φ)

        if axes == 'yz':
            y_zpf = self.zero_point_fluctuation(axes[0])
            z_zpf = self.zero_point_fluctuation(axes[1])
            kz = (kc - 1 / self.trap.tweezer.zR)
            return -P0 * ωdsinθ * (kc * y_zpf) * (kz * z_zpf) * np.cos(θtw) * np.sin(Φ)

    ################################################################################################
    # FREQUENCY RESPONSE FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    def χM(self, ω):
        """
        The mechanical susceptibility for phonon annihilation
        [complex (rad/s)⁻¹]
        """
        return (-1j*(ω - self.Ω) + self.Γ / 2)**(-1)

    def μm(self, ω):
        """
        The difference of the conjugate mechanical susceptibilities
        for phonon annihilation and creation
        [complex (rad/s)⁻¹]
        """
        return self.χM(ω) - np.conjugate(self.χM(-ω))

    def μp(self, ω):
        """
        The sum of the conjugate mechanical susceptibilities
        for phonon annihilation and creation
        [complex (rad/s)⁻¹]
        """
        return self.χM(ω) + np.conjugate(self.χM(-ω))

    def M(self, ω):
        """
        The 1D normalisation factor for position
        [·]
        """
        return 1 + self.μm(ω) * self.multitrap.cavity.ηm(ω) * np.abs(self.G)**2

    def Q1D(self, ω):
        """
        The 1D noise vector for this dimension
        [·]
        """
        κ = self.multitrap.cavity.κ
        Γ = self.Γ
        G = self.G
        G_star = np.conjugate(G)

        χC, χC_star = self.multitrap.cavity.χC(ω), np.conjugate(self.multitrap.cavity.χC(-ω))
        χM, χM_star = self.χM(ω), np.conjugate(self.χM(-ω))

        μ = self.μm(ω)
        M = self.M(ω)

        Q = np.zeros(self.multitrap.num_operators, dtype=np.complex128)
        index = self.noise_index

        # a_in
        Q[0] = 1j * μ * np.sqrt(κ) * G_star * χC / M
        # a†_in
        Q[1] = 1j * μ * np.sqrt(κ) * G * χC_star / M
        # bj_in
        Q[index] = np.sqrt(Γ) * χM / M
        # bjn†_in
        Q[index + 1] = np.sqrt(Γ) * χM_star / M
        # Q should be empty in all other indices, since it is just the vector for the 1D position
        return Q

