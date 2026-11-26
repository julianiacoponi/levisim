"""
A small library of materials for levitated particles, each number with its source

Every value below was retrieved from the cited source when this module was written (November
2026); the DOI or URL sits next to the number. Where a value depends on the sample (porosity,
grade, crystal quality, film versus bulk) the docstring of the material says so. Optical values
from tabulated data were read from the refractiveindex.info database
(https://github.com/polyanskiy/refractiveindex.info-database, CC0), which reproduces each source's
table. SILICON, DIAMOND and GOLD carry the full n, κ tables of their sources (shipped as CSV files in
levisim/physics/optical_data, each with its citation in the header) as a `RefractiveIndexTable`,
interpolated linearly in n and κ at any wavelength inside the table.

Access a material by its module constant (e.g. `FUSED_SILICA`) or by name with
`get_material('fused_silica')`; `MATERIALS` maps every name to its `Material`.

Materials are immutable and shared: derive a variant with `material.replace(...)`.
"""

import numpy as np

from levisim.core.constants import μ0
from levisim.physics.material import (
    Material,
    RefractiveIndexTable,
    load_refractive_index_table,
)

__all__ = [
    'BISMUTH',
    'DIAMOND',
    'DIAMOND_NK',
    'FUSED_SILICA',
    'GOLD',
    'GOLD_NK',
    'LEVISIM_DEFAULT_SILICA',
    'MATERIALS',
    'NDFEB',
    'PYROLYTIC_GRAPHITE',
    'SILICON',
    'SILICON_NK',
    'VATERITE',
    'get_material',
]


NIST_SRD_126 = (
    'NIST Standard Reference Database 126 (Hubbell and Seltzer, NISTIR 5632), Table 1, '
    'https://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html'
)

####################################################################################################
# Silica
# --------------------------------------------------------------------------------------------------
LEVISIM_DEFAULT_SILICA = Material(
    name='levisim default silica',
    # supplier specification of the silica particles used in the experiments (microParticles GmbH,
    # SiO2-R spheres), 1.85 g/cm³:
    # https://www.microparticles-shop.de/Monodisperse-Particles-for-Research-Purposes/Silica-particles-SiO2-R-in-the-size-range-150-nm-25-m:::1_6.html
    ρ=1850.0,
    # εr = 1.98 (n = 1.4071) is the value used in the group's earlier simulations; its origin
    # (a measurement, a supplier value or a fit) is not recorded. It is inconsistent with ρ = 1850
    # under Lorentz-Lorenz (see notes) and kept for continuity with published numbers
    permittivity=1.98,
    references={
        'ρ': (
            'microParticles GmbH SiO2-R specification (1.85 g/cm³), https://www.microparticles-shop.de/'
            'Monodisperse-Particles-for-Research-Purposes/Silica-particles-SiO2-R-in-the-size-range-150-nm-25-m:::1_6.html'
        ),
        'εr': 'value used in earlier group simulations; source not recorded',
    },
    notes=(
        'The default material of Bead(). KNOWN INCONSISTENCY, kept deliberately: ρ and εr are not '
        'mutually consistent under the Lorentz-Lorenz void mixture (εr-1)/(εr+2) ∝ density with bulk '
        'fused silica (ρ = 2201 kg/m³, εr(1064 nm) = 2.1014, FUSED_SILICA) as the reference: '
        'εr = 1.98 corresponds to 2018 kg/m³, and ρ = 1850 kg/m³ to εr = 1.875. Both values are kept '
        'for continuity with published and thesis numbers; for a self-consistent porous silica use '
        'e.g. FUSED_SILICA.replace(ρ=..., permittivity=...) with the Lorentz-Lorenz pair.'
    ),
)

FUSED_SILICA = Material(
    name='fused silica',
    # Corning HPFS 7980 fused silica product information sheet: density 2.201 g/cm³ at 25 °C
    # https://www.corning.com/media/worldwide/csm/documents/1e624b3034474193b5b00ea6f558dd3d2.pdf
    ρ=2201.0,
    # Sellmeier formula of Malitson, J. Opt. Soc. Am. 55, 1205 (1965), doi:10.1364/JOSA.55.001205,
    # valid 0.21-3.71 μm at 20 °C
    permittivity='Malitson',
    # SI volume susceptibility of SiO2, -11.28 × 10⁻⁶: Houlton et al., Rev. Sci. Instrum. 89,
    # 125107 (2018), doi:10.1063/1.5051667, Table 1, from Wapler et al., J. Magn. Reson. 242, 233
    # (2014), doi:10.1016/j.jmr.2014.02.005
    χ_m=-11.28e-6,
    references={
        'ρ': (
            'Corning HPFS 7980 product information sheet (2.201 g/cm³), '
            'https://www.corning.com/media/worldwide/csm/documents/1e624b3034474193b5b00ea6f558dd3d2.pdf'
        ),
        'εr': 'Malitson, J. Opt. Soc. Am. 55, 1205 (1965), doi:10.1364/JOSA.55.001205',
        'χ_m': (
            'Houlton et al., Rev. Sci. Instrum. 89, 125107 (2018), doi:10.1063/1.5051667, Table 1 '
            '(SiO2: -11.28 × 10⁻⁶), from Wapler et al., J. Magn. Reson. 242, 233 (2014), '
            'doi:10.1016/j.jmr.2014.02.005'
        ),
    },
    notes=(
        'Bulk synthetic fused silica. Stöber-type nanospheres are porous and can be 10-20% less dense '
        'with a lower index; see LEVISIM_DEFAULT_SILICA. The diamagnetic susceptibility of different '
        'silica glasses differs at the percent level (e.g. -11.03 × 10⁻⁶ is also quoted for fused silica).'
    ),
)

####################################################################################################
# Silicon
# --------------------------------------------------------------------------------------------------
# n, κ from 0.25 to 1.45 μm: Green, Sol. Energ. Mat. Sol. Cells 92, 1305 (2008),
# doi:10.1016/j.solmat.2008.06.009, intrinsic Si at 300 K, every 10 nm (κ is band-to-band absorption);
# n beyond 1.45 μm up to 14 μm: Li, J. Phys. Chem. Ref. Data 9, 561 (1980), doi:10.1063/1.555624, 293 K
# table, with κ = 0 (free-carrier absorption neglected). The two join at 1.45 μm, where Green gives
# n = 3.485 and Li 3.4845. At 1064 nm the table gives the Green interpolation between 1.06 μm
# (3.553, 9.3631e-5) and 1.07 μm (3.550, 6.8118e-5), and at 1550 nm Li's tabulated n = 3.4757.
_SI_GREEN = load_refractive_index_table('silicon_green2008.csv')
_SI_LI = load_refractive_index_table('silicon_li1980_293K.csv')
_si_beyond = _SI_LI._λ_μm > _SI_GREEN._λ_μm[-1]
SILICON_NK = RefractiveIndexTable(
    np.concatenate([_SI_GREEN.λ, _SI_LI.λ[_si_beyond]]),
    np.concatenate([_SI_GREEN.n_values, _SI_LI.n_values[_si_beyond]]),
    np.concatenate([_SI_GREEN.κ_values, _SI_LI.κ_values[_si_beyond]]),
    source=(
        'Green, Sol. Energ. Mat. Sol. Cells 92, 1305 (2008), doi:10.1016/j.solmat.2008.06.009 (0.25-1.45 μm, '
        '300 K); Li, J. Phys. Chem. Ref. Data 9, 561 (1980), doi:10.1063/1.555624 (1.5-14 μm, 293 K, κ = 0); '
        'via the refractiveindex.info database (CC0)'
    ),
    name='silicon (Green 2008 + Li 1980)',
    λ_μm=np.concatenate([_SI_GREEN._λ_μm, _SI_LI._λ_μm[_si_beyond]]),
)

SILICON = Material(
    name='silicon',
    # NIST SRD 126 Table 1: Si, 2.330 g/cm³
    ρ=2330.0,
    permittivity=SILICON_NK,
    references={
        'ρ': NIST_SRD_126 + ' (Si: 2.330 g/cm³)',
        'εr(0.25-1.45 μm)': 'Green, Sol. Energ. Mat. Sol. Cells 92, 1305 (2008), doi:10.1016/j.solmat.2008.06.009',
        'εr(1.5-14 μm)': 'Li, J. Phys. Chem. Ref. Data 9, 561 (1980), doi:10.1063/1.555624',
        'tables': 'refractiveindex.info database (CC0), https://github.com/polyanskiy/refractiveindex.info-database, '
                  'shipped as levisim/physics/optical_data/silicon_*.csv',
    },
    notes=(
        'Crystalline, intrinsic. Two sources at slightly different temperatures (300 K below 1.45 μm, '
        '293 K above), joined at 1.45 μm (n = 3.485 against 3.4845). At 1064 nm silicon absorbs weakly '
        '(κ ≈ 8 × 10⁻⁵, band-to-band), which heats a levitated particle; doping adds free-carrier '
        'absorption not included here, and κ = 0 beyond 1.45 μm.'
    ),
)

####################################################################################################
# Diamond
# --------------------------------------------------------------------------------------------------
# Phillip and Taft, Phys. Rev. 136, A1445 (1964), doi:10.1103/PhysRev.136.A1445 (Kramers-Kronig
# analysis, natural diamond), n, κ from 0.0354 to 10 μm. Near the usual lasers the rows are
# (1.033201562 μm: 2.390522955), (1.239841874 μm: 2.388124787), (1.549802343 μm: 2.387783491),
# (2.066403124 μm: 2.38324149), with κ = 0
DIAMOND_NK = load_refractive_index_table('diamond_phillip1964.csv', name='diamond (Phillip and Taft 1964)')

DIAMOND = Material(
    name='diamond',
    # Ioffe NSM archive, diamond basic parameters at 300 K: 3.515 g/cm³
    # https://www.ioffe.ru/SVA/NSM/Semicond/Diamond/basic.html
    ρ=3515.0,
    permittivity=DIAMOND_NK,
    references={
        'ρ': 'Ioffe NSM archive, Diamond basic parameters (3.515 g/cm³), https://www.ioffe.ru/SVA/NSM/Semicond/Diamond/basic.html',
        'εr': 'Phillip and Taft, Phys. Rev. 136, A1445 (1964), doi:10.1103/PhysRev.136.A1445',
        'tables': 'refractiveindex.info database (CC0), https://github.com/polyanskiy/refractiveindex.info-database, '
                  'shipped as levisim/physics/optical_data/diamond_phillip1964.csv',
    },
    notes=(
        'Source-dependent at the 1% level: ellipsometry of single-crystal CVD diamond (Taylor et al., '
        'Diam. Relat. Mater. 135, 109837 (2023), doi:10.1016/j.diamond.2023.109837) gives n = 2.3744 at '
        '1064 nm and 2.3682 at 1550 nm, against 2.3902 and 2.3878 here. The table is sparse in the near '
        'infrared (rows at 1.03, 1.24, 1.55 and 2.07 μm), so interpolated values between rows carry the '
        'linear-interpolation error (below 1e-4 in n). Nanodiamonds contain sp² carbon and are less '
        'dense (about 3.1-3.2 g/cm³).'
    ),
)

####################################################################################################
# Vaterite (birefringent CaCO3)
# --------------------------------------------------------------------------------------------------
# Bishop et al., Phys. Rev. Lett. 92, 198104 (2004), doi:10.1103/PhysRevLett.92.198104: vaterite is
# positive uniaxial with no = 1.55 and ne = 1.65
_vaterite_no, _vaterite_ne = 1.55, 1.65

VATERITE = Material(
    name='vaterite',
    # Arita et al., Sci. Adv. 6, eaaz9858 (2020), doi:10.1126/sciadv.aaz9858: ρ = 2.54 g/cm³ (the
    # density used for their levitated vaterite microspheres)
    ρ=2540.0,
    # body frame: optic axis along z
    permittivity=np.diag([_vaterite_no**2, _vaterite_no**2, _vaterite_ne**2]),
    references={
        'ρ': 'Arita et al., Sci. Adv. 6, eaaz9858 (2020), doi:10.1126/sciadv.aaz9858',
        'εr': 'Bishop et al., Phys. Rev. Lett. 92, 198104 (2004), doi:10.1103/PhysRevLett.92.198104',
    },
    notes=(
        'Bulk-crystal indices without a stated wavelength (treated as non-dispersive). Synthetic '
        'vaterite spheres are polycrystalline with a spherulitic optic-axis structure, and their '
        'effective birefringence is smaller, Δn ≈ 0.06-0.09 (e.g. Arita 2020 measured 0.09 ± 0.02).'
    ),
)

####################################################################################################
# Gold
# --------------------------------------------------------------------------------------------------
# Olmon et al., Phys. Rev. B 86, 235147 (2012), doi:10.1103/PhysRevB.86.235147, evaporated gold,
# n, κ from 0.3 to 24.93 μm, e.g. (1.06 μm: 0.1705, 7.075), (1.07 μm: 0.1731, 7.152), (1.55 μm: 0.3226, 10.62)
GOLD_NK = load_refractive_index_table('gold_olmon2012_evaporated.csv', name='evaporated gold (Olmon et al. 2012)')

GOLD = Material(
    name='gold',
    # NIST SRD 126 Table 1: Au, 19.32 g/cm³
    ρ=19320.0,
    permittivity=GOLD_NK,
    references={
        'ρ': NIST_SRD_126 + ' (Au: 19.32 g/cm³)',
        'εr': 'Olmon et al., Phys. Rev. B 86, 235147 (2012), doi:10.1103/PhysRevB.86.235147 (evaporated gold)',
        'tables': 'refractiveindex.info database (CC0), https://github.com/polyanskiy/refractiveindex.info-database, '
                  'shipped as levisim/physics/optical_data/gold_olmon2012_evaporated.csv',
    },
    notes=(
        'Strongly sample-dependent in the infrared: Olmon et al. report different ε for evaporated, '
        'template-stripped and single-crystal gold, and Johnson and Christy (1972, '
        'doi:10.1103/PhysRevB.6.4370) differ again. Nanoparticles smaller than the electron mean free '
        'path (about 40 nm) have extra surface damping (larger Im ε), not included.'
    ),
)

####################################################################################################
# Diamagnets
# --------------------------------------------------------------------------------------------------
# Simon, Heflinger and Geim, Am. J. Phys. 69, 702 (2001), doi:10.1119/1.1375157, Table I: SI volume
# susceptibilities -χ × 10⁶ of 170 (bismuth metal), 450 (pyrolytic graphite, field perpendicular to
# the sheets) and 85 (pyrolytic graphite, field parallel to the sheets)
_χ_Bi = -170e-6
_χ_PG_perpendicular = -450e-6
_χ_PG_parallel = -85e-6

PYROLYTIC_GRAPHITE = Material(
    name='pyrolytic graphite',
    # Liu et al., Micromachines 14, 1242 (2023), doi:10.3390/mi14061242: ρ = 2.2 g/cm³
    ρ=2200.0,
    permittivity=None,
    # body frame: graphite c axis (normal to the sheets) along z
    χ_m=np.diag([_χ_PG_parallel, _χ_PG_parallel, _χ_PG_perpendicular]),
    references={
        'ρ': 'Liu et al., Micromachines 14, 1242 (2023), doi:10.3390/mi14061242',
        'χ_m': 'Simon, Heflinger and Geim, Am. J. Phys. 69, 702 (2001), doi:10.1119/1.1375157, Table I',
    },
    notes=(
        'Anisotropic diamagnet. χ_m depends on the grade and falls in magnitude with temperature '
        '(about 1 ppm/K out of plane); ρ depends on the grade (about 2.1-2.26 g/cm³). '
        'No optical data here (graphite is absorbing and optically anisotropic).'
    ),
)

BISMUTH = Material(
    name='bismuth',
    # NIST SRD 126 Table 1: Bi, 9.747 g/cm³
    ρ=9747.0,
    permittivity=None,
    χ_m=_χ_Bi,
    references={
        'ρ': NIST_SRD_126 + ' (Bi: 9.747 g/cm³)',
        'χ_m': 'Simon, Heflinger and Geim, Am. J. Phys. 69, 702 (2001), doi:10.1119/1.1375157, Table I',
    },
    notes=(
        'Polycrystalline bismuth metal; the single crystal is anisotropic. Other compilations give '
        'χ_m = -1.66 × 10⁻⁴. No optical data here.'
    ),
)

####################################################################################################
# Hard magnet
# --------------------------------------------------------------------------------------------------
# Sagawa et al., J. Appl. Phys. 55, 2083 (1984), doi:10.1063/1.333572, typical sintered Nd15B8Fe77
# specimen: Br = 1.23 T, density 7.4 Mg/m³, specific resistivity 1.4 μΩm
_NdFeB_Br = 1.23
_NdFeB_ρ = 7400.0
_NdFeB_resistivity = 1.4e-6

NDFEB = Material(
    name='NdFeB (sintered)',
    ρ=_NdFeB_ρ,
    permittivity=None,
    # remanent magnetisation M_r = B_r/μ0, body frame: easy axis along z
    magnetisation=np.array([0.0, 0.0, _NdFeB_Br / μ0]),
    conductivity=1 / _NdFeB_resistivity,
    references={
        'ρ': 'Sagawa et al., J. Appl. Phys. 55, 2083 (1984), doi:10.1063/1.333572',
        'magnetisation': 'Sagawa et al. (1984), doi:10.1063/1.333572: B_r = 1.23 T, M_r = B_r/μ0',
        'conductivity': 'Sagawa et al. (1984), doi:10.1063/1.333572: specific resistivity 1.4 μΩm',
    },
    notes=(
        'The original 1984 sintered magnet; commercial grades span B_r ≈ 1.0-1.5 T. The recoil '
        'susceptibility (μ_rec - 1 ≈ 0.05) is not included (χ_m = 0). Micro- and nanoscale NdFeB '
        'particles generally have a lower remanence than bulk.'
    ),
)

####################################################################################################
MATERIALS = {
    'levisim_default_silica': LEVISIM_DEFAULT_SILICA,
    'fused_silica': FUSED_SILICA,
    'silicon': SILICON,
    'diamond': DIAMOND,
    'vaterite': VATERITE,
    'gold': GOLD,
    'pyrolytic_graphite': PYROLYTIC_GRAPHITE,
    'bismuth': BISMUTH,
    'ndfeb': NDFEB,
}


def get_material(name):
    """
    The library `Material` called `name` (case-insensitive; spaces and hyphens read as underscores)
    """
    key = str(name).strip().lower().replace(' ', '_').replace('-', '_')
    if key not in MATERIALS:
        raise KeyError(f'Unknown material {name!r}; the library has {sorted(MATERIALS)}')
    return MATERIALS[key]

