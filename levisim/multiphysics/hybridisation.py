import random

from copy import copy
from functools import cached_property, partial
from collections import namedtuple
from itertools import permutations, chain
from sympy.combinatorics import Permutation
from sympy.combinatorics.named_groups import SymmetricGroup

from levisim.core.logger import logger

__all__ = [
    'HybridisationTerm',
    'term_string',

    'generate_subscripts_and_labels',
    'symmetric_group_with_min_fixed_points',
    'generate_deranged_terms',
    'generate_all_hybridisation_functions',
    'generate_determinant_and_adjugate_by_order',

    'truncate_power_series',
    'sieve_power_series',
    'sieve_power_series_for_sequence',
]


HybridisationTerm = namedtuple('HybridisationTerm', ['sign', 'pairs'])


def term_string(term):
    if not term.pairs:
        return '+1'
    t_str = '+' if term.sign == 1 else '-'
    for ii, pair in enumerate(term.pairs):
        pair0_str = ''.join(pair[0])
        pair1_str = ''.join(pair[1])
        dot = '' if ii == len(term.pairs) - 1 else '.'
        t_str = f'{t_str}R{pair0_str}{pair1_str}{dot}'
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


# TODO: set this up so that the loop only has to happen once to generate
# all determinant and adjugate terms required...
# TODO: figure out why the performance of adjugate calculations is so slow...
def generate_deranged_terms(
    n,
    i=None, j=None,
    dimension_labels=None,
    max_order=None,
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
        if max_order:
            min_fixed_points = n - max_order
            perm_generator = symmetric_group_with_min_fixed_points(n, min_fixed_points)
        else:
            perm_generator = SymmetricGroup(n).generate_schreier_sims()
        if debug:
            logger.debug(f'doing detD{n}')

    # adjugate entry
    if i is not None and j is not None:
        assert 0 <= i <= (n - 1)
        assert 0 <= j <= (n - 1)
        det, diag = False, i == j

        # if only terms up to a certain order are desired, generate only the necessary permutations
        # NOTE: this is useful when dealing with 7+ orders
        if max_order:
            if diag:
                # the set of permutations that generate terms up to a given order for the diagonal
                # is the same as for the determinant (namely, the partial derangement sets)
                perm_generator = symmetric_group_with_min_fixed_points(
                    n - 1,
                    (n - 1) - max_order,
                )

            else:
                # the set of permutations that generate terms up to a given order for the off-diagonal
                # is made of two parts

                # the 'determinant' part,  Rij * det(D\ij)
                # is generated by permutations from the symmetric group order (n - 1)
                # that begin with the first item
                perm_gen_1 = symmetric_group_with_min_fixed_points(
                    n - 1,
                    # the +1 accounts for the fact that 1 fixed point contributes to a desired power
                    # namely, the first item being in the first place gives Rij
                    (n - 1) - max_order + 1,
                    begin_with_first_item=True,
                )

                # the 'path-cycles' part, Ri_path_j.R_cycles
                # is generated by the remaining permutations from the symmetric group order (n - 1)
                # that *do not* begin with the first item
                perm_gen_2 = symmetric_group_with_min_fixed_points(
                    n - 1,
                    # no +1 needed here
                    (n - 1) - max_order,
                    begin_with_first_item=False,
                )
                perm_generator = chain(perm_gen_1, perm_gen_2)

        else:
            perm_generator = SymmetricGroup(n - 1).generate_schreier_sims()

        if diag:
            dim = dimension_labels[i]
            adj_ii_labels = copy(dimension_labels)
            adj_ii_labels.remove(dim)

            if debug:
                logger.debug(f'doing diagonal adjD{n}_{dim}{dim}')
                logger.debug(f'old labels {dimension_labels}')
                logger.debug(f'new labels {adj_ii_labels}')

        else:
            path_origin, path_destination = dimension_labels[i], dimension_labels[j]
            adj_ij_labels = copy(dimension_labels)
            adj_ij_labels.remove(path_origin)
            adj_ij_labels.remove(path_destination)
            adj_ij_labels.insert(0, path_origin + path_destination)

            if debug:
                logger.debug(f'doing off-diagonal adjD{n}_{path_origin}{path_destination}')
                logger.debug(f'old labels {dimension_labels}')
                logger.debug(f'new labels {adj_ij_labels}')

    all_terms = []
    while True:
        try:
            perm = next(perm_generator)
            pairs = []
            sign = 1
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
                        logger.debug(f'edge {position}→{label} == {origin}→{destination}')

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
                            logger.debug(f'edge {position}→{label} == {path_origin}→{path_destination}')

                    elif position == 0:
                        destination = adj_ij_labels[label]
                        pairs.append((path_origin, destination))
                        if debug:
                            logger.debug(f'edge {position}→{label} == {path_origin}→{destination}')

                    elif label == 0:
                        origin = adj_ij_labels[position]
                        pairs.append((origin, path_destination))
                        if debug:
                            logger.debug(f'edge {position}→{label} == {origin}→{path_destination}')

                    else:
                        # excludes fixed points
                        if position == label:
                            continue

                        origin = adj_ij_labels[position]
                        destination = adj_ij_labels[label]
                        pairs.append((origin, destination))
                        if debug:
                            logger.debug(f'edge {position}→{label} == {origin}→{destination}')

            term = HybridisationTerm(sign=sign, pairs=pairs)
            if debug:
                logger.debug(perm.full_cyclic_form, term_string(term))

            # NOTE: sorted() could help with unit testing, but not needed for now
            # doesn't seem to affect performance that much
            # could use set() instead of list()
            # all_terms.append(HybridisationTerm(sign=sign, pairs=sorted(pairs)))

            all_terms.append(term)

        except StopIteration:
            break

    return all_terms


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


def generate_determinant_and_adjugate_by_order(n, labels, max_order=None):
    """
    Generate the determinant and adjguate terms for the hybridisation matrix
    """
    if max_order is None:
        max_order = n

    # first, generate the power series up to the defined max order
    deranged_determinant_by_order = {}
    max_determinant = generate_deranged_terms(
        n,
        dimension_labels=labels,
        max_order=max_order,
    )
    deranged_determinant_by_order[max_order] = max_determinant
    logger.info(f'{len(max_determinant)} terms in determinant of order {max_order} of {n}')

    deranged_adjugate_by_order = {}
    deranged_adjugate_by_order[max_order] = {}

    for row_index, row_label in enumerate(labels):
        for column_index, column_label in enumerate(labels):
            ij = row_label + column_label
            deranged_adjugate_by_order[max_order][ij] = generate_deranged_terms(
                n,
                row_index, column_index,
                dimension_labels=labels,
                max_order=max_order,
            )

    if n > 1:
        a, b = random.sample(labels, 2)
        aa, ab = f'{a}{a}', f'{a}{b}'
        logger.info(f'{len(deranged_adjugate_by_order[max_order][aa])} terms in {a} diagonal adjugate of order {max_order} of {n}')
        logger.info(f'{len(deranged_adjugate_by_order[max_order][ab])} terms in {ab} off-diagonal adjugate of order {max_order} of {n}')

    logger.info('Truncating power series...')
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
    logger.info('Truncation complete!')

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
