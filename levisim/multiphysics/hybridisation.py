import random

from collections import namedtuple
from itertools import permutations
from sympy.combinatorics import Permutation

from levisim.core.logger import logger
from levisim.core.wrappers import timer, profiler, progress_bar


__all__ = [
    'HybridisationTerm',
    'term_string',
    'term_latex',

    'generate_subscripts_and_labels',
    'symmetric_group_with_min_fixed_points',
    'generate_deranged_terms',
    'generate_determinant_and_adjugate_terms',
    'generate_all_hybridisation_functions',
    'generate_determinant_and_adjugate_by_order',

    'truncate_power_series',
    'sieve_power_series',
    'sieve_power_series_for_sequence',
]


DEBUG = False
TIMER = False
PROFILER = False
PROGRESS_BAR = False


class HybridisationTerm(namedtuple('_HybridisationTermFields', ['sign', 'pairs'])):
    """
    One signed product of hybridisation functions, e.g. -Rx1y1(ω)·Ry1x1(ω)

    `sign` is +1 or -1 and `pairs` is a tuple of (origin, destination) dimension-label pairs,
    one per hybridisation function Rjn,km in the product; an empty `pairs` is the constant +1
    term. The number of pairs is the term's order in the power series. The frequency-dependent
    value is evaluated by MultiTrap.hybridisation_term_value.

    Still a namedtuple subclass, so it is immutable, hashable (`pairs` is stored as a tuple of
    tuples) and pickles by name. The power-series pickles under
    notebooks/multiphysics/power-series/pickle/ were written when HybridisationTerm was a plain
    namedtuple with `pairs` as a list; protocol-4 unpickling calls HybridisationTerm.__new__,
    which converts those lists to tuples, so the old files load as instances of this class.
    [·]
    """
    __slots__ = ()

    def __new__(cls, sign, pairs):
        """
        Creates a term, storing `pairs` as a tuple of (origin, destination) tuples
        """
        if sign not in (+1, -1):
            raise ValueError(f'HybridisationTerm sign must be +1 or -1, not {sign!r}')
        return super().__new__(cls, sign, tuple(tuple(pair) for pair in pairs))

    @property
    def order(self):
        """
        Order of the term, i.e. the number of hybridisation functions in the product
        """
        return len(self.pairs)

    @property
    def leading_dimensions(self):
        """
        The origin label of each hybridisation function in the product, in order
        """
        return [pair[0] for pair in self.pairs]

    def __str__(self):
        """
        Compact string form, e.g. -Rx1y1·Ry1x1 (see `term_string`)
        """
        return term_string(self)

    def latex(self, with_ω=True):
        """
        LaTeX form of the term (see `term_latex`)
        """
        return term_latex(self, with_ω=with_ω)


def term_string(term):
    """
    Generates a string that represents the hybridistaion term
    e.g. Rx1y1.Ry1x2.Rx2x1
    """
    if not term.pairs:
        return '+1'
    t_str = '+' if term.sign == 1 else '-'
    for ii, pair in enumerate(term.pairs):
        jn, km = ''.join(pair[0]), ''.join(pair[1])
        dot = '' if ii == len(term.pairs) - 1 else '·'
        t_str = f'{t_str}R{jn}{km}{dot}'
    return t_str


def term_latex(term, with_ω=True):
    """
    Generates latex that represents the hybridistaion term
    e.g. R_{x_{1} y{1}} \cdot R_{y{1}x{2}} \cdot R_{x{2}x{1}}
    """
    if not term.pairs:
        return '+1'
    t_str = '+' if term.sign == 1 else '-'
    for ii, pair in enumerate(term.pairs):
        jn, km = ''.join(pair[0]), ''.join(pair[1])
        j, n, k, m = jn[0], jn[1:], km[0], km[1:]
        j_n, k_m = f'{j}_{{{n}}}', f'{k}_{{{m}}}'

        if with_ω:
            dot = '(\omega)' if ii == len(term.pairs) - 1 else '(\omega) \cdot '
        else:
            dot = '' if ii == len(term.pairs) - 1 else ' \cdot '

        t_str = rf'{t_str}R_{{{j_n + k_m}}}{dot}'
    return t_str


def generate_subscripts_and_labels(num_traps, Cartesian_axes_per_trap):
    """
    Generates:
    - the unique subscripts for the hybridisation functions
        e.g. 'abcd' for 4 total dimensions
    - the unique labels of the form
        e.g. ['x1' 'y1', 'x2', 'y2'] for 2 particles xy

    Possible combinations (up to 10 total dimensions)
        10 = 10 particles, x (10x1)
        10 = 5 particles, xy (5x2)

        9 = 9 particles, x (9x1)
        9 = 3 particles, xyz (3x3)

        8 = 8 particles, x (8x1)
        8 = 4 particles, xy (4x2)

        7 = 7 particles, x (7x1)

        6 = 6 particles, x (6x1)
        6 = 3 particles, xy (3x2)
        6 = 2 particles, xyz (2x3)

        5 = 5 particles, x (5x1)

        4 = 4 particles, x (4x1)
        4 = 2 particles, xy (2x2)

        3 = 3 particles, x (3x1)
        3 = 1 particle, xyz (1x3)

        2 = 2 particles, x (2x1)
        2 = 1 particle, xy (1x2)

        1 = 1 particle, x (1x1)
    """
    n = num_traps * Cartesian_axes_per_trap
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    subscripts = alphabet[:n]

    labels = []
    for trap_index in range(1, num_traps + 1):
        for Cartesian_index in range(Cartesian_axes_per_trap):
            label = str('xyz'[Cartesian_index]) + str(trap_index)
            labels.append(label)

    logger.info(f'Generated {labels=} for {n}D={num_traps}×{Cartesian_axes_per_trap} system')
    return subscripts, labels


def symmetric_group_with_min_fixed_points(
    order,
    min_fixed_points,
    begin_with_first_item=None,
):
    """
    Generator for the symmetric group that produces permutations with a minimum number of fixed points
    """
    items = list(range(order))
    for perm_tuple in permutations(items):

        # these clauses are used for the off-diagonal adjugates
        if begin_with_first_item is True:
            if perm_tuple[0] != 0:
                continue
        elif begin_with_first_item is False:
            if perm_tuple[0] == 0:
                continue

        perm = Permutation(perm_tuple)
        fixed_points = sum(1 for ii in range(order) if perm.array_form[ii] == ii)

        if fixed_points >= min_fixed_points:
            yield perm


def _permutation_records(order):
    """
    All permutations of range(`order`) in lexicographic order, each as a tuple
    (array_form, number of fixed points, list of non-trivial cycles)

    A pure-Python replacement for building a sympy Permutation per tuple; the cycles are listed
    in the same order as sympy's Permutation.cyclic_form (each starting from its smallest
    element, cycles ordered by that element), which is the only thing the term signs use.
    """
    records = []
    for perm in permutations(range(order)):
        seen = [False] * order
        cycles = []
        fixed_points = 0
        for start in range(order):
            if seen[start]:
                continue
            if perm[start] == start:
                seen[start] = True
                fixed_points += 1
                continue
            cycle = []
            position = start
            while not seen[position]:
                seen[position] = True
                cycle.append(position)
                position = perm[position]
            cycles.append(cycle)
        records.append((perm, fixed_points, cycles))
    return records


def _deranged_term(perm, cycles, labels):
    """
    Determinant or diagonal-adjugate term of the permutation `perm` acting on `labels`

    Sign is (-1)^(number of non-trivial cycles): sgn(σ) = (-1)^(Σ(len - 1)) times one factor -1
    per off-diagonal entry Dij = -Rij of the derangement matrix, i.e. per moved point.
    """
    sign = (-1)**len(cycles)
    pairs = [(labels[position], labels[label]) for position, label in enumerate(perm) if position != label]
    return HybridisationTerm(sign=sign, pairs=pairs)


def _path_cycle_term(perm, cycles, path_origin, path_destination, adj_ij_labels):
    """
    Off-diagonal adjugate term of the permutation `perm` of the reduced label list `adj_ij_labels`

    Item 0 of `adj_ij_labels` is the merged origin-destination node: the cycle through it is the
    path from `path_origin` to `path_destination` and every other non-trivial cycle contributes a
    factor -1 to the sign.
    """
    sign = (-1)**sum(1 for cycle in cycles if 0 not in cycle)
    pairs = []
    for position, label in enumerate(perm):
        if position == 0 and label == 0:
            pairs.append((path_origin, path_destination))
        elif position == 0:
            pairs.append((path_origin, adj_ij_labels[label]))
        elif label == 0:
            pairs.append((adj_ij_labels[position], path_destination))
        elif position != label:
            pairs.append((adj_ij_labels[position], adj_ij_labels[label]))
    return HybridisationTerm(sign=sign, pairs=pairs)


def _adjugate_entry_terms(n, i, j, dimension_labels, max_order, records):
    """
    Terms of adjugate entry (i, j), built from the precomputed permutation `records` of order n - 1
    """
    if i == j:
        dim = dimension_labels[i]
        adj_ii_labels = [label for label in dimension_labels if label != dim]
        min_fixed_points = (n - 1) - max_order
        return [
            _deranged_term(perm, cycles, adj_ii_labels)
            for perm, fixed_points, cycles in records
            if fixed_points >= min_fixed_points
        ]

    path_origin, path_destination = dimension_labels[i], dimension_labels[j]
    adj_ij_labels = [label for label in dimension_labels if label not in (path_origin, path_destination)]
    adj_ij_labels.insert(0, path_origin + path_destination)

    # the 'determinant' part, Rij·det(D\ij), comes from permutations that fix the merged item 0;
    # that fixed point already contributes the power Rij, hence the +1 on the fixed point count
    # the 'path-cycles' part, Ri_path_j·R_cycles, comes from the permutations that move item 0
    # NOTE: the two parts are listed one after the other (not interleaved) to keep the term order
    # the original implementation produced, which keeps the summed spectra bit-identical
    min_fixed_points_1 = (n - 1) - max_order + 1
    min_fixed_points_2 = (n - 1) - max_order
    terms = [
        _path_cycle_term(perm, cycles, path_origin, path_destination, adj_ij_labels)
        for perm, fixed_points, cycles in records
        if perm[0] == 0 and fixed_points >= min_fixed_points_1
    ]
    terms += [
        _path_cycle_term(perm, cycles, path_origin, path_destination, adj_ij_labels)
        for perm, fixed_points, cycles in records
        if perm[0] != 0 and fixed_points >= min_fixed_points_2
    ]
    return terms


@timer(TIMER, debug=DEBUG)
@profiler(PROFILER)
def generate_deranged_terms(
    n,
    i=None, j=None,
    dimension_labels=None,
    max_order=None,
    debug=DEBUG,
):
    """
    Generates the power-series terms of the determinant, or of one adjugate entry, of the
    derangement matrix D = I - R of an n-dimensional system

    With `i`, `j` both None this is det(D); otherwise it is adjugate entry (i, j). The determinant
    and the diagonal adjugate entries are sums over partial derangements (permutations of n, or
    n - 1, items with their fixed points dropped); the off-diagonal entries are sums over
    'path-cycles' from dimension i to dimension j. `max_order` keeps only terms with at most that
    many hybridisation functions (None or 0 keeps all of them).

    NOTE: terms are generated in lexicographic permutation order for every `max_order`. Before
    November 2026 a falsy `max_order` used sympy's Schreier-Sims ordering instead, which gives the
    same set of terms in a different order (and failed outright for n = 1 adjugate entries);
    MultiTrap always passes a `max_order`, so its spectra are unaffected.

    To build the determinant and all n² adjugate entries together, use
    `generate_determinant_and_adjugate_terms`, which enumerates each symmetric group only once.
    [list of HybridisationTerm]
    """
    if dimension_labels is None:
        dimension_labels = list('abcdefghijklmnopqrstuvwyxz')
    dimension_labels = list(dimension_labels[:n])
    if not max_order:
        max_order = n

    if i is None and j is None:
        if debug:
            logger.debug(f'doing detD{n}')
        return [
            _deranged_term(perm, cycles, dimension_labels)
            for perm, fixed_points, cycles in _permutation_records(n)
            if fixed_points >= n - max_order
        ]

    assert 0 <= i <= (n - 1)
    assert 0 <= j <= (n - 1)
    if debug:
        logger.debug(f'doing adjD{n}_{dimension_labels[i]}{dimension_labels[j]}')
    return _adjugate_entry_terms(n, i, j, dimension_labels, max_order, _permutation_records(n - 1))


@timer(TIMER, debug=DEBUG)
@profiler(PROFILER)
def generate_determinant_and_adjugate_terms(n, labels, max_order=None):
    """
    Generates the determinant terms and all n² adjugate-entry terms of the derangement matrix
    in a single pass over each symmetric group

    S_n is enumerated once for the determinant and S_(n-1) once for all n² adjugate entries:
    every adjugate entry is a relabelling of the same permutations of n - 1 items, so their
    fixed points and cycles are computed once and reused. Returns the same lists, in the same
    order, as calling `generate_deranged_terms` once per entry.
    [(list of HybridisationTerm, {'x1y1': list of HybridisationTerm, ...})]
    """
    labels = list(labels[:n])
    if not max_order:
        max_order = n

    determinant = generate_deranged_terms(n, dimension_labels=labels, max_order=max_order)
    records = _permutation_records(n - 1)
    adjugate = {}
    for row_index, row_label in enumerate(labels):
        for column_index, column_label in enumerate(labels):
            adjugate[row_label + column_label] = _adjugate_entry_terms(
                n, row_index, column_index, labels, max_order, records,
            )
    return determinant, adjugate


def generate_all_hybridisation_functions(labels, sign=1):
    """
    Generate a list of all the hybridisation functions given a list of dimension labels
    """
    all_Rjk = []
    for jn in labels:
        for km in labels:
            if jn == km:
                continue

            Rjk = HybridisationTerm(sign=sign, pairs=[(jn, km)])
            all_Rjk.append(Rjk)

    n = len(labels)
    assert len(all_Rjk) == n**2 - n

    return all_Rjk


@timer(TIMER)
@profiler(PROFILER)
@progress_bar(loop_variable_path='labels', enabled=PROGRESS_BAR)
def generate_determinant_and_adjugate_by_order(n, labels, max_order=None):
    """
    Generate the determinant and adjugate terms for the hybridisation matrix, by order

    Builds the full power series up to `max_order` (default n) with
    `generate_determinant_and_adjugate_terms`, then truncates it down to order 1, so that
    result[order] holds every term with at most `order` hybridisation functions.
    [({order: list}, {order: {'x1y1': list, ...}})]
    """
    if max_order is None:
        max_order = n

    # first, generate the power series up to the defined max order
    deranged_determinant_by_order = {}
    deranged_adjugate_by_order = {}
    max_determinant, max_adjugate = generate_determinant_and_adjugate_terms(n, labels, max_order=max_order)
    deranged_determinant_by_order[max_order] = max_determinant
    deranged_adjugate_by_order[max_order] = max_adjugate
    logger.info(f'{len(max_determinant)} terms in determinant of order {max_order} of {n}')

    if n > 1:
        a, b = random.sample(labels, 2)
        aa, ab = f'{a}{a}', f'{a}{b}'
        logger.info(f'{len(deranged_adjugate_by_order[max_order][aa])} terms in {a} diagonal adjugate of order {max_order} of {n}')
        logger.info(f'{len(deranged_adjugate_by_order[max_order][ab])} terms in {ab} off-diagonal adjugate of order {max_order} of {n}')

    logger.info('🪚 Truncating power series...')
    # next, truncate the power series down to order 1
    for truncation_order in list(range(1, max_order))[::-1]:
        deranged_determinant_by_order[truncation_order] = truncate_power_series(
            # truncate just the next highest order series
            deranged_determinant_by_order[truncation_order + 1],
            truncation_order,
        )
        deranged_adjugate_by_order[truncation_order] = {}
        for row_index, row_label in enumerate(labels):
            for column_index, column_label in enumerate(labels):
                ij = row_label + column_label
                deranged_adjugate_by_order[truncation_order][ij] = truncate_power_series(
                    # truncate just the next highest order series
                    deranged_adjugate_by_order[truncation_order + 1][ij],
                    truncation_order,
                )
    logger.info('🪵 Truncation complete!')

    return deranged_determinant_by_order, deranged_adjugate_by_order


def truncate_power_series(power_series, truncation_power):
    """
    Truncates a `power_series` (an iterable of `HybridisationTerm`s)
    leaving only terms of order up to `truncation_power`
    """
    return list(filter(lambda term: len(term.pairs) <= truncation_power, power_series))


def sieve_power_series(power_series, sieve_power):
    """
    Sieves a `power_series` (an iterable of `HybridisationTerm`s)
    leaving only terms of order `sieve_power`
    """
    return list(filter(lambda term: len(term.pairs) == sieve_power, power_series))


def sieve_power_series_for_sequence(power_series, sieve_sequence):
    """
    Sieves a `power_series` (an iterable of `HybridisationTerm`s)
    leaving only terms whose sequence of leading dimensions matches `sieve_sequence`
    """
    return list(filter(lambda term: [pair[0] for pair in term.pairs] == sieve_sequence, power_series))
