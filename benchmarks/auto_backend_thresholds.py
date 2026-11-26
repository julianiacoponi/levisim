"""
Re-measures the thresholds of backend='auto' (owner decision, round 5): run on a quiet machine with
numba installed, from the repository root: python benchmarks/auto_backend_thresholds.py

- free space: wall time of the first FreeSpaceSimulation in a fresh process (kernel loading
  included) and of a second one, NumPy vs numba, for 2^10..2^18 steps -> AUTO_NUMBA_MIN_STEPS
- linear CavitySimulation: time per step against the state dimension n (shelf cavity, k beads
  with 3 axes), for 1 and 8 trajectories -> AUTO_NUMBA_MAX_STATE
"""
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = """
import sys; sys.path.insert(0, {root!r})
import time
from levisim import Trap
from levisim.simulation.free_space import FreeSpaceSimulation
N={N}; b={b!r}
t0=time.perf_counter(); s=FreeSpaceSimulation(trap=Trap(), num_samples=N, seed=1, backend=b); t1=time.perf_counter()
assert s.backend_used==b
FreeSpaceSimulation(trap=Trap(), num_samples=N, seed=2, backend=b); t2=time.perf_counter()
print(t1-t0, t2-t1)
"""
CAV = """
import sys; sys.path.insert(0, {root!r})
import time
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils import shelf
from levisim.physics.tweezer import Tweezer
from levisim.physics.bead import Bead
from levisim.simulation.cavity import CavitySimulation
k={k}; K={K}; b={b!r}; N={N}
m=MultiTrap(num_traps=k, Cartesian_axes_per_trap=3, cavity_kwargs=shelf.DEFAULT_cavity_kwargs,
            gas_kwargs=shelf.DEFAULT_gas_kwargs, tweezers=[Tweezer() for _ in range(k)], beads=[Bead() for _ in range(k)])
kw=dict(backend=b, initial_state='zero', num_trajectories=K)
CavitySimulation(m, num_samples=128, seed=0, **kw)
t0=time.perf_counter(); s=CavitySimulation(m, num_samples=N, seed=1, **kw); t1=time.perf_counter()
print(s.backend_used, s.Φ.shape[0], (t1-t0)/N*1e6)
"""


def run(code):
    p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False)
    if p.returncode:
        raise RuntimeError(p.stderr)
    return p.stdout.strip().splitlines()[-1].split()


def main():
    out = {'free_space': {}, 'cavity_linear': {}}
    for e in range(10, 19):
        for b in ('numpy', 'numba'):
            rs = [run(FS.format(root=ROOT, N=2**e, b=b)) for _ in range(3)]
            first, warm = (statistics.median(float(r[i]) for r in rs) for i in (0, 1))
            out['free_space'][f'2^{e} {b}'] = {'first_run_s': first, 'second_run_s': warm}
            print(f'free space 2^{e:<2d} {b:5s} first {first:.3f} s  second {warm:.4f} s', flush=True)
    for K, ks, N in ((1, (1, 2, 4, 8, 12, 16, 20, 24, 32, 40), 2**12), (8, (4, 16, 32, 48), 2**10)):
        for k in ks:
            for b in ('numpy', 'numba'):
                rs = [run(CAV.format(root=ROOT, k=k, K=K, b=b, N=N)) for _ in range(3)]
                us = statistics.median(float(r[2]) for r in rs)
                out['cavity_linear'][f'K{K} k{k} {b}'] = {'n': int(rs[0][1]), 'us_per_step': us}
                print(f'cavity K={K} n={rs[0][1]:>4s} {b:5s} {us:8.2f} μs/step', flush=True)
    with open('auto_backend_thresholds.json', 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
