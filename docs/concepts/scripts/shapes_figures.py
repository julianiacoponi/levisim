"""
Figures of docs/concepts/particles_and_materials.md: the polarisability and gas damping of
equal-volume spheroids against their aspect ratio, and Mie theory against the dipole model

    python docs/concepts/scripts/shapes_figures.py

Writes docs/concepts/figures/spheroid_tensors.png and docs/concepts/figures/mie_vs_dipole.png
(the second needs the optional backend, pip install "levisim[mie]"), and prints the numbers quoted
on the page.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import nm
from levisim.physics.bead import Bead
from levisim.physics.gas import Gas
from levisim.physics.mie import compare_with_dipole
from levisim.physics.particle import Sphere
from levisim.physics.shapes import Spheroid

FIGURES = Path(__file__).resolve().parents[1] / 'figures'
set_plt_rcParams_to_xmgrace_style()

# ---- equal-volume spheroids ------------------------------------------------------------------------
r0 = Bead().r
gas = Gas()
sphere = Sphere(r=r0)
α0 = sphere.α
Γ0 = sphere.translational_damping_tensor(gas)[0, 0]
Γrot0 = sphere.rotational_damping_tensor(gas)[0, 0]
ratios = np.geomspace(0.1, 10, 81)
α_par, α_perp, Γ_par, Γ_perp, Γrot_par, Γrot_perp = ([] for _ in range(6))
for p in ratios:
    b = r0 / p**(1/3)
    sp = Spheroid(p * b, b)
    α = np.diag(sp.static_polarisability_tensor())
    Γ = np.diag(sp.translational_damping_tensor(gas))
    Γrot = np.diag(sp.rotational_damping_tensor(gas))
    α_par.append(α[2] / α0)
    α_perp.append(α[0] / α0)
    Γ_par.append(Γ[2] / Γ0)
    Γ_perp.append(Γ[0] / Γ0)
    Γrot_par.append(Γrot[2] / Γrot0)
    Γrot_perp.append(Γrot[0] / Γrot0)
for p in (0.2, 3.0):
    i = int(np.argmin(np.abs(ratios - p)))
    print(f'RESULT aspect ratio {ratios[i]:.2f}: α∥/α_sphere = {α_par[i]:.4f}, α⊥/α_sphere = {α_perp[i]:.4f}, '
          f'Γ∥/Γ_sphere = {Γ_par[i]:.4f}, Γ⊥/Γ_sphere = {Γ_perp[i]:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.semilogx(ratios, α_par, label=r'$\alpha_\parallel$ (along the symmetry axis)')
ax.semilogx(ratios, α_perp, '--', label=r'$\alpha_\perp$')
ax.axhline(1, color='0.6', lw=1)
ax.set_xlabel('aspect ratio $a/b$')
ax.set_ylabel(r'$\alpha / \alpha_\mathrm{sphere}$')
ax.legend(frameon=False)
ax = axes[1]
ax.semilogx(ratios, Γ_par, label=r'$\Gamma_{\mathrm{cm},\parallel}$')
ax.semilogx(ratios, Γ_perp, '--', label=r'$\Gamma_{\mathrm{cm},\perp}$')
ax.semilogx(ratios, Γrot_par, ':', label=r'$\Gamma_{\mathrm{rot},\parallel}$')
ax.semilogx(ratios, Γrot_perp, '-.', label=r'$\Gamma_{\mathrm{rot},\perp}$')
ax.axhline(1, color='0.6', lw=1)
ax.set_xlabel('aspect ratio $a/b$')
ax.set_ylabel(r'$\Gamma / \Gamma_\mathrm{sphere}$')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / 'spheroid_tensors.png', dpi=150)

# ---- Mie against the dipole model ------------------------------------------------------------------
radii = np.geomspace(10, 1500, 120) * nm
comparisons = [compare_with_dipole(Sphere(r=r)) for r in radii]
for r in (Bead().r, 300*nm, 1000*nm):
    c = compare_with_dipole(Sphere(r=r))
    print(f'RESULT r = {r / nm:.1f} nm, kr = {c.x:.3f}: Re α_dipole/Re α_Mie - 1 = {c.polarisability_ratio - 1:+.3e}, '
          f'σ_sca dipole/Mie - 1 = {c.scattering_ratio - 1:+.3e}, scattering outside a₁ = {c.multipole_fraction:.3e}')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.loglog(radii / nm, [abs(c.polarisability_ratio - 1) for c in comparisons], label=r'$|\mathrm{Re}\,\alpha_\mathrm{dip}/\mathrm{Re}\,\alpha_\mathrm{Mie} - 1|$')
ax.loglog(radii / nm, [abs(c.scattering_ratio - 1) for c in comparisons], '--', label=r'$|\sigma_\mathrm{sca,dip}/\sigma_\mathrm{sca,Mie} - 1|$')
ax.set_xlabel('sphere radius $r$ [nm]')
ax.set_ylabel('relative difference')
ax.legend(frameon=False, loc='lower right')
ax = axes[1]
ax.loglog(radii / nm, [c.multipole_fraction for c in comparisons])
ax.set_xlabel('sphere radius $r$ [nm]')
ax.set_ylabel(r'fraction of $\sigma_\mathrm{sca}$ outside $a_1$')
for ax in axes:
    for r in (Bead().r, 300*nm, 1000*nm):
        ax.axvline(r / nm, color='0.6', lw=1, ls=':')
fig.tight_layout()
fig.savefig(FIGURES / 'mie_vs_dipole.png', dpi=150)
