"""
Maximum relative difference of each QLT method from LINEAR_AMPLIFIER_MODEL

Prints the table quoted on the 'Calculation methods' docs page. POWER_SERIES_HYBRIDISATION is
left out: it needs a power-series pickle, and the shipped ones stop at 6 dimensions.
"""
import numpy as np

from levisim.core.units import kHz
from levisim.multiphysics.enums import QuantumLinearTheoryMethod as Q
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils import shelf

CASES = [
    # (label, num_traps, axes per trap, ω_range, extra kwargs)
    ('physical defaults', 2, 1, np.linspace(50, 350, 301) * kHz, {}),
    ('physical defaults', 2, 2, np.linspace(50, 350, 301) * kHz, {}),
    ('physical defaults', 1, 3, np.linspace(50, 350, 301) * kHz, {}),
    ('PHASE_DEPENDENT toy', 2, 2, np.arange(250, 320, 0.5) * kHz, dict(toy_params=shelf.PHASE_DEPENDENT_toy_params)),
    ('OFF_NODE pairs toy', 2, 1, np.arange(250, 320, 0.5) * kHz, dict(toy_params=shelf.OFF_NODE_toy_params_pairs)),
    ('OFF_NODE pairs toy', 2, 2, np.arange(250, 320, 0.5) * kHz, dict(toy_params=shelf.OFF_NODE_toy_params_pairs)),
]
METHODS = [Q.HYBRID_LINEAR_AMPLIFIER_MODEL, Q.TNT_LINEAR_AMPLIFIER_MODEL, Q.NUMERICAL_HYBRIDISATION,
           Q.ANALYTIC_INSIDE_CAVITY, Q.ANALYTIC_CAVITY_OUTPUT]


def max_rel(a, b):
    return np.max(np.abs(np.asarray(b) / np.asarray(a) - 1))


rows = []
for name, traps, axes, ω_range, kwargs in CASES:
    mtrap = MultiTrap(num_traps=traps, Cartesian_axes_per_trap=axes, ω_range=ω_range,
                      gas_kwargs=shelf.DEFAULT_gas_kwargs, cavity_kwargs=shelf.DEFAULT_cavity_kwargs, **kwargs)
    S_hh = mtrap.homodyne_spectrum(Q.LINEAR_AMPLIFIER_MODEL)
    psd, _ = mtrap.all_position_power_spectra(Q.LINEAR_AMPLIFIER_MODEL)
    for method in METHODS:
        try:
            hom = f'{max_rel(S_hh, mtrap.homodyne_spectrum(method)):.1e}'
        except ValueError:
            hom = 'n/a (raises ValueError)'
        psd_m, _ = mtrap.all_position_power_spectra(method)
        rows.append((f'{name}, {traps}×{axes}', method.name, hom, f'{max_rel(psd, psd_m):.1e}'))

print('RESULT | system | method | homodyne | position PSDs |')
for row in rows:
    print('RESULT | ' + ' | '.join(row) + ' |')
