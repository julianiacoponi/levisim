"""
Regenerates the hybridisation power-series pickles that MultiTrap(load_in_power_series=True) reads

Replaces the notebook notebooks/multiphysics/power-series/save_power_series_to_pickle.ipynb.
Each pickle holds the tuple

    (deranged_determinant_by_order, deranged_adjugate_by_order)

returned by `generate_determinant_and_adjugate_by_order(n, labels, max_order)` for a system of
N traps with d Cartesian axes each (n = N·d dimensions), saved as

    notebooks/multiphysics/power-series/pickle/{n}D_{N}x{d}_up_to_{max_order}.pickle

MultiTrap loads the full-order file, max_order = n. The three files tracked in git (the ones the
test suite needs) are the default targets:

    python scripts/generate_power_series_pickles.py            # 1×2, 2×2, 3×2 at full order
    python scripts/generate_power_series_pickles.py --check    # compare with the files on disk
    python scripts/generate_power_series_pickles.py --systems 4×2 2×3 --all-orders

`--check` regenerates in memory and compares the term lists with the stored files, term by term
and in order, without writing anything. The bytes differ from the original files, which were
written when HybridisationTerm stored `pairs` as a list; the unpickled terms compare equal.

Approximate generation times quoted in the original notebook: 6 dimensions and below are quick,
7D about a minute, 8D 5-10 minutes, 9D about an hour.
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import levisim  # noqa: F401
except ImportError:
    # running from a clone without `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT))

from levisim.core.utils import load_variable_from_pickle, save_variable_to_pickle  # noqa: E402
from levisim.multiphysics.hybridisation import (  # noqa: E402
    generate_determinant_and_adjugate_by_order,
    generate_subscripts_and_labels,
)

PICKLE_DIR = REPO_ROOT / 'notebooks' / 'multiphysics' / 'power-series' / 'pickle'

# the files tracked in git, which the test suite loads
TRACKED_SYSTEMS = ['1×2', '2×2', '3×2']

# the full set the original notebook generated (9×1 and 3×3 were commented out there: ~1 hour each)
ALL_SYSTEMS = ['1×1', '2×1', '3×1', '4×1', '5×1', '6×1', '7×1', '8×1', '1×2', '2×2', '3×2', '4×2', '1×3', '2×3']


def parse_system(system):
    """'3×2' or '3x2' -> (3, 2)"""
    traps, axes = system.replace('x', '×').split('×')
    return int(traps), int(axes)


def pickle_path(N, d, max_order, directory=PICKLE_DIR):
    return Path(directory) / f'{N * d}D_{N}x{d}_up_to_{max_order}.pickle'


def generate(N, d, max_order):
    """(deranged_determinant_by_order, deranged_adjugate_by_order) for N traps with d axes each"""
    _, labels = generate_subscripts_and_labels(N, d)
    return generate_determinant_and_adjugate_by_order(N * d, labels, max_order=max_order)


def compare(stored, regenerated):
    """
    Lists the differences between two power series, comparing term lists in order

    Returns an empty list if every order of the determinant and of every adjugate entry holds
    the same HybridisationTerms in the same order.
    """
    differences = []
    (det_a, adj_a), (det_b, adj_b) = stored, regenerated
    if sorted(det_a) != sorted(det_b):
        differences.append(f'determinant orders {sorted(det_a)} != {sorted(det_b)}')
    for order in sorted(set(det_a) & set(det_b)):
        if list(det_a[order]) != list(det_b[order]):
            differences.append(f'determinant order {order}: {len(det_a[order])} vs {len(det_b[order])} terms')
        if sorted(adj_a[order]) != sorted(adj_b[order]):
            differences.append(f'adjugate order {order}: different entries')
            continue
        for entry in adj_a[order]:
            if list(adj_a[order][entry]) != list(adj_b[order][entry]):
                differences.append(
                    f'adjugate order {order} entry {entry}: '
                    f'{len(adj_a[order][entry])} vs {len(adj_b[order][entry])} terms'
                )
    return differences


def count_terms(power_series):
    determinant, adjugate = power_series
    top = max(determinant)
    return len(determinant[top]), sum(len(terms) for terms in adjugate[top].values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('--systems', nargs='+', default=None,
                        help=f'systems as N×d (or NxD); default: the tracked files {TRACKED_SYSTEMS}')
    parser.add_argument('--all', action='store_true',
                        help=f'every system the original notebook generated: {ALL_SYSTEMS}')
    parser.add_argument('--all-orders', action='store_true',
                        help='write one file per max_order = 1 ... n (as the notebook did), not just n')
    parser.add_argument('--check', action='store_true',
                        help='compare with the stored files instead of writing them; exit 1 on a mismatch')
    parser.add_argument('--out-dir', type=Path, default=PICKLE_DIR,
                        help='output directory (default: notebooks/multiphysics/power-series/pickle)')
    args = parser.parse_args(argv)

    systems = ALL_SYSTEMS if args.all else (args.systems or TRACKED_SYSTEMS)
    if not args.check:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    mismatches = 0
    for system in systems:
        N, d = parse_system(system)
        n = N * d
        orders = range(1, n + 1) if args.all_orders else [n]
        for max_order in orders:
            path = pickle_path(N, d, max_order, args.out_dir)
            start = time.perf_counter()
            power_series = generate(N, d, max_order)
            elapsed = time.perf_counter() - start
            num_det, num_adj = count_terms(power_series)
            summary = (
                f'{system} up to order {max_order}: {num_det} determinant and {num_adj} adjugate terms'
                f' ({elapsed:.2f} s)'
            )
            if args.check:
                if not path.exists():
                    print(f'MISSING  {path.name}  ({summary})')
                    mismatches += 1
                    continue
                differences = compare(load_variable_from_pickle(path), power_series)
                print(f'{"DIFFERENT" if differences else "SAME"}  {path.name}  ({summary})')
                for difference in differences:
                    print(f'    {difference}')
                mismatches += bool(differences)
            else:
                save_variable_to_pickle(power_series, path)
                print(f'wrote  {path.name}  ({summary}, {path.stat().st_size} bytes)')
    return 1 if (args.check and mismatches) else 0


if __name__ == '__main__':
    sys.exit(main())
