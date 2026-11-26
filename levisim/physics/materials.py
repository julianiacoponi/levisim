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
    'NDFEB_GRADES',
    'NDFEB_RECOIL_PERMEABILITY',
    'PYROLYTIC_GRAPHITE',
    'SILICON',
    'SILICON_NK',
    'VATERITE',
    'get_material',
    'ndfeb_grade',
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

# Electrical conductivity of pyrolytic graphite: Chen, Ammu, Masania, Steeneken and Alijani,
# "Diamagnetic composites for high-Q levitating resonators", arXiv:2208.10393, Supplementary Table I
# (material properties of their levitated pyrolytic graphite FEM): σ ∥ sheets (in-plane, σx,y) =
# 200000 S/m and σ ⊥ sheets (along the c axis, σz) = 200 S/m, which they take from Pappis and Blum,
# J. Am. Ceram. Soc. 44, 592 (1961) (not read here)
_σ_PG_in_plane = 2.0e5
_σ_PG_c_axis = 200.0

PYROLYTIC_GRAPHITE = Material(
    name='pyrolytic graphite',
    # Liu et al., Micromachines 14, 1242 (2023), doi:10.3390/mi14061242: ρ = 2.2 g/cm³
    ρ=2200.0,
    permittivity=None,
    # body frame: graphite c axis (normal to the sheets) along z
    χ_m=np.diag([_χ_PG_parallel, _χ_PG_parallel, _χ_PG_perpendicular]),
    conductivity=np.diag([_σ_PG_in_plane, _σ_PG_in_plane, _σ_PG_c_axis]),
    references={
        'ρ': 'Liu et al., Micromachines 14, 1242 (2023), doi:10.3390/mi14061242',
        'χ_m': 'Simon, Heflinger and Geim, Am. J. Phys. 69, 702 (2001), doi:10.1119/1.1375157, Table I',
        'conductivity': (
            'Chen et al., Diamagnetic composites for high-Q levitating resonators, arXiv:2208.10393, '
            'https://arxiv.org/abs/2208.10393, Supplementary Table I: in-plane 2 × 10⁵ S/m, c axis 200 S/m '
            '(their source: Pappis and Blum, J. Am. Ceram. Soc. 44, 592 (1961))'
        ),
    },
    notes=(
        'Anisotropic diamagnet and conductor (body z = c axis, normal to the sheets). χ_m depends on the '
        'grade and falls in magnitude with temperature (about 1 ppm/K out of plane); ρ depends on the '
        'grade (about 2.1-2.26 g/cm³). The conductivity is a tensor: in-plane 2 × 10⁵ S/m, along the c '
        'axis 200 S/m, and is grade-dependent. EddyCurrentDamping needs an isotropic σ: pass '
        'conductivity=... explicitly (the in-plane value bounds the sphere damping from above, the c-axis '
        'value from below). No optical data here (graphite is absorbing and optically anisotropic).'
    ),
)

# Electrical conductivity of bismuth: Xiong, Guo, Chen, Du, Li and Wu, Commun. Phys. 8, 80 (2025),
# doi:10.1038/s42005-025-02010-7, the value used in their FEM of and fit to the eddy-current damping
# of a levitated bismuth sphere: σ = 8.7 × 10⁵ S/m (their source of the number is not stated). Single
# crystals are anisotropic: about 1.35 and 1.10 μΩ·m along and across the trigonal axis at 300 K
# ("Finite- and Quantum-Size Effects of Bismuth Nanowires", IntechOpen, https://www.intechopen.com/chapters/10520).
_σ_Bi = 8.7e5

BISMUTH = Material(
    name='bismuth',
    # NIST SRD 126 Table 1: Bi, 9.747 g/cm³
    ρ=9747.0,
    permittivity=None,
    χ_m=_χ_Bi,
    conductivity=_σ_Bi,
    references={
        'ρ': NIST_SRD_126 + ' (Bi: 9.747 g/cm³)',
        'χ_m': 'Simon, Heflinger and Geim, Am. J. Phys. 69, 702 (2001), doi:10.1119/1.1375157, Table I',
        'conductivity': (
            'Xiong et al., Commun. Phys. 8, 80 (2025), doi:10.1038/s42005-025-02010-7 '
            '(σ = 8.7 × 10⁵ S/m, used for their levitated bismuth sphere)'
        ),
    },
    notes=(
        'Polycrystalline bismuth metal; the single crystal is anisotropic. Other compilations give '
        'χ_m = -1.66 × 10⁻⁴. The conductivity is that of bulk bismuth at room temperature (resistivity '
        'about 1.1-1.35 μΩ·m depending on the crystal direction); sintered or powder-built particles '
        'conduct far less (Xiong et al. reduce the eddy damping 1000-fold that way).'
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

# Recoil permeability of sintered NdFeB, μ_rec ≈ 1.05: Constantinides and Gulick, "NdFeB for High
# Temperature Motor Applications", Arnold Magnetic Technologies, SMMA Fall Technical Conference (2004),
# https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/NdFeB-for-high-temperature-motor-applications.pdf
# ("Typical values of Recoil Permeability are about 1.05 for sintered Ferrite, SmCo and NdFeB"); also
# Kwon et al., Rare Metals (2020), doi:10.1007/s12598-019-01336-w (≈ 1.05, fully magnetised).
# The recoil slope is measured along the easy axis: M = M_r + (μ_rec - 1) H there.
NDFEB_RECOIL_PERMEABILITY = 1.05

# Typical (nominal) remanence B_r [T] of common sintered grades at 20 °C: Arnold Magnetic
# Technologies, Neodymium-Iron-Boron Magnet Catalog, Rev. 210607, summary table of basic grades,
# https://www.arnoldmagnetics.com/wp-content/uploads/2019/06/Arnold-Neo-Catalog.pdf. The same catalogue's
# grade sheets give density 7.6 g/cm³ and electrical resistivity 150 μΩ·cm along and 130 μΩ·cm across
# the easy axis for every basic grade (e.g. N35, N42 and N52).
NDFEB_GRADES = {
    'N35': 1.210,
    'N38': 1.260,
    'N40': 1.270,
    'N42': 1.315,
    'N45': 1.350,
    'N48': 1.400,
    'N50': 1.425,
    'N52': 1.450,
    'N55': 1.490,
}
_ARNOLD_NEO = 'Arnold Magnetic Technologies, Neodymium-Iron-Boron Magnet Catalog, Rev. 210607, ' \
    'https://www.arnoldmagnetics.com/wp-content/uploads/2019/06/Arnold-Neo-Catalog.pdf'
_RECOIL_SOURCE = (
    'Constantinides and Gulick, NdFeB for High Temperature Motor Applications (Arnold Magnetic Technologies, '
    'SMMA 2004), https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/NdFeB-for-high-temperature-motor-applications.pdf; '
    'Kwon et al., Rare Metals (2020), doi:10.1007/s12598-019-01336-w'
)

NDFEB = Material(
    name='NdFeB (sintered)',
    ρ=_NdFeB_ρ,
    permittivity=None,
    # recoil susceptibility μ_rec - 1 along the easy axis (body z); no cited transverse value, so 0
    χ_m=np.diag([0.0, 0.0, NDFEB_RECOIL_PERMEABILITY - 1]),
    # remanent magnetisation M_r = B_r/μ0, body frame: easy axis along z
    magnetisation=np.array([0.0, 0.0, _NdFeB_Br / μ0]),
    conductivity=1 / _NdFeB_resistivity,
    references={
        'ρ': 'Sagawa et al., J. Appl. Phys. 55, 2083 (1984), doi:10.1063/1.333572',
        'magnetisation': 'Sagawa et al. (1984), doi:10.1063/1.333572: B_r = 1.23 T, M_r = B_r/μ0',
        'conductivity': 'Sagawa et al. (1984), doi:10.1063/1.333572: specific resistivity 1.4 μΩm',
        'χ_m': 'recoil permeability μ_rec ≈ 1.05 along the easy axis: ' + _RECOIL_SOURCE,
    },
    notes=(
        'The original 1984 sintered magnet; commercial grades span B_r ≈ 1.2-1.5 T (NDFEB_GRADES, '
        'ndfeb_grade()). χ_m holds the recoil susceptibility μ_rec - 1 = 0.05 along the easy axis (body z) '
        'only; the transverse susceptibility (set by the anisotropy field) is not included. levisim\'s '
        'magnet dynamics (MeissnerImagePlane) use the permanent moment only; χ_m is recorded for '
        'reference. Micro- and nanoscale NdFeB particles generally have a lower remanence than bulk.'
    ),
)


def ndfeb_grade(grade, recoil_susceptibility=True):
    """
    A sintered NdFeB `Material` of a commercial grade ('N35' ... 'N55', see `NDFEB_GRADES`) with its
    typical remanence B_r (M_r = B_r/μ0 along the body z axis), density 7.6 g/cm³ and the conductivity
    tensor of 1/(130 μΩ·cm) across and 1/(150 μΩ·cm) along the easy axis (Arnold Magnetic Technologies
    catalogue, Rev. 210607). With `recoil_susceptibility` (default), χ_m = μ_rec - 1 = 0.05 along the
    easy axis (`NDFEB_RECOIL_PERMEABILITY`); False gives χ_m = 0.
    """
    key = str(grade).strip().upper()
    if key not in NDFEB_GRADES:
        raise KeyError(f'Unknown NdFeB grade {grade!r}; known grades: {sorted(NDFEB_GRADES)}')
    B_r = NDFEB_GRADES[key]
    χ_easy = NDFEB_RECOIL_PERMEABILITY - 1 if recoil_susceptibility else 0.0
    return Material(
        name=f'NdFeB {key} (sintered, typical)',
        ρ=7600.0,
        permittivity=None,
        χ_m=np.diag([0.0, 0.0, χ_easy]),
        magnetisation=np.array([0.0, 0.0, B_r / μ0]),
        conductivity=np.diag([1 / 1.30e-6, 1 / 1.30e-6, 1 / 1.50e-6]),
        references={
            'magnetisation': f'{_ARNOLD_NEO}: typical B_r of {key} = {B_r} T',
            'ρ': f'{_ARNOLD_NEO}: grade sheets, density 7.6 g/cm³',
            'conductivity': f'{_ARNOLD_NEO}: grade sheets, resistivity 150 μΩ·cm ∥ and 130 μΩ·cm ⊥ easy axis',
            'χ_m': 'recoil permeability μ_rec ≈ 1.05 along the easy axis: ' + _RECOIL_SOURCE,
        },
        notes=(
            'Nominal catalogue values; each grade has a ± spread of about 30-40 mT in B_r (e.g. N42: '
            '1.28-1.35 T), and B_r falls by about 0.12 %/°C. Easy axis along body z.'
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

