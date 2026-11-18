"""
Checks levisim's PSD normalisation and sideband convention against the Lyapunov steady state

For one x motion in a red-detuned cavity, compares ∫ S_qq(ω) dω/2π with ⟨q²⟩ from
levisim.multiphysics.covariance, and S_qq(+Ω)/S_qq(-Ω) with (n + 1)/n. Quoted on the
'Conventions' docs page.
"""
import numpy as np

from levisim.core.units import kHz
from levisim.multiphysics.covariance import steady_state_covariance
from levisim.multiphysics.enums import QuantumLinearTheoryMethod as Q
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs

ω_range = np.arange(-3000, 3000, 0.02) * kHz
mtrap = MultiTrap(num_traps=1, Cartesian_axes_per_trap=1, ω_range=ω_range,
                  gas_kwargs=DEFAULT_gas_kwargs, cavity_kwargs=DEFAULT_cavity_kwargs)
psd, _ = mtrap.all_position_power_spectra(Q.LINEAR_AMPLIFIER_MODEL)
S_qq = psd[:, 0]

V = steady_state_covariance(mtrap)  # quadrature order (A, P, X_x1, P_x1)
q2 = V[2, 2].real
integral = np.trapezoid(S_qq, ω_range) / (2 * np.pi)
n = (q2 - 1) / 2

Ω = mtrap.dimensions['x1'].Ω
S_plus = S_qq[np.argmin(np.abs(ω_range - Ω))]
S_minus = S_qq[np.argmin(np.abs(ω_range + Ω))]

print(f'RESULT Ω/2π = {Ω / kHz:.3f} kHz')
print(f'RESULT ∫S_qq dω/2π = {integral:.4f}, ⟨q²⟩ (Lyapunov) = {q2:.4f}, rel. diff = {integral / q2 - 1:.1e}')
print(f'RESULT n = (⟨q²⟩ - 1)/2 = {n:.2f}')
print(f'RESULT S_qq(+Ω)/S_qq(-Ω) = {S_plus / S_minus:.4f}, (n + 1)/n = {(n + 1) / n:.4f}')
