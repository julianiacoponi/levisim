# Particles and materials

A levitated particle has two independent descriptions: what it is made of, and what shape it has.
levisim keeps them apart. A {class}`~levisim.physics.material.Material` holds the bulk properties of
a substance, and a {class}`~levisim.physics.particle.Particle` combines one material with a shape to
give the quantities the dynamics need: the mass $m_n$, the inertia tensor, the charge and the
polarisability. {class}`~levisim.physics.bead.Bead`, the particle of every calculation so far, is now
a {class}`~levisim.physics.particle.Sphere` made of a named default material, and gives exactly the
numbers it gave before.

## Materials

`Material(name, ρ, permittivity=..., χ_m=..., magnetisation=..., conductivity=...)` stores, in SI
units,

- the mass density $\rho$ [kg/m³];
- the relative permittivity $\varepsilon_r$, which may be a constant, a complex number
  $\varepsilon' + i\varepsilon''$ with $\varepsilon'' > 0$ for an absorbing material, a $3\times3$
  tensor in the body frame for a birefringent crystal, a function $\varepsilon_r(\lambda)$ of the
  vacuum wavelength, the name of a dispersion model (`'Malitson'`, the Sellmeier formula for fused
  silica), or a table of values at given wavelengths;
- the magnetic volume susceptibility $\chi_m$ (dimensionless, negative for a diamagnet), a scalar
  or a body-frame tensor;
- a remanent magnetisation $M_r$ [A/m] for a permanent magnet, and an electrical conductivity
  $\sigma$ [S/m].

`material.εr(λ)` returns the permittivity at a vacuum wavelength, `material.n(λ)` the refractive
index $n = \sqrt{\varepsilon_r}$ (complex if absorbing, three principal indices if anisotropic), and
flags such as `is_absorbing(λ)`, `is_anisotropic(λ)`, `is_diamagnetic` and `is_dispersive` say what
kind of material it is. Materials are immutable, because the library objects are shared by every
particle that uses them; `material.replace(ρ=...)` derives a modified copy.

```python
from levisim.physics.materials import FUSED_SILICA, GOLD, VATERITE
from levisim.core.units import nm

FUSED_SILICA.n(1064*nm)     # 1.44963 (Malitson 1965)
GOLD.εr(1550*nm)            # complex, Re ε < 0
VATERITE.εr_tensor()        # diag(no², no², ne²), optic axis along body z
```

### The material library

{mod}`levisim.physics.materials` holds a small library. Each number was retrieved from the source
cited next to it in the code, and `material.references` lists the sources; where a value depends on
the sample, `material.notes` says so.

| Name | ρ [kg/m³] | Optical response | Magnetic response |
|---|---|---|---|
| `LEVISIM_DEFAULT_SILICA` | 1850 | $\varepsilon_r = 1.98$ | - |
| `FUSED_SILICA` | 2201 | Malitson (1965) Sellmeier, 0.21-3.71 μm | - |
| `SILICON` | 2330 | complex at 1064 nm, real at 1550 nm | - |
| `DIAMOND` | 3515 | 1064 and 1550 nm | - |
| `VATERITE` | 2540 | uniaxial tensor, $n_o = 1.55$, $n_e = 1.65$ | - |
| `GOLD` | 19320 | complex at 1064 and 1550 nm | - |
| `PYROLYTIC_GRAPHITE` | 2200 | - | anisotropic diamagnet |
| `BISMUTH` | 9747 | - | diamagnet, $\chi_m = -1.7 \times 10^{-4}$ |
| `NDFEB` | 7400 | - | $B_r = 1.23$ T, conductive |

`LEVISIM_DEFAULT_SILICA` is the material of `Bead()`. Its density is the supplier's value for the
porous silica spheres used in the experiments, and $\varepsilon_r = 1.98$ is the value used in
earlier simulations, whose origin is not recorded. The two are not mutually consistent under the
Lorentz-Lorenz relation for a porous material, $(\varepsilon_r - 1)/(\varepsilon_r + 2) \propto \rho$,
with bulk fused silica as the reference: $\varepsilon_r = 1.98$ corresponds to
2018 kg/m³, and $\rho = 1850$ kg/m³ to $\varepsilon_r = 1.875$. They are kept unchanged so that
no existing number moves.

## Particles

A `Particle` has a material, a charge in units of the elementary charge $e$ (default 0), the
vacuum wavelength $\lambda$ at which a dispersive permittivity is evaluated, the trapping phase
$\Phi$ along the cavity axis, and an orientation (a unit quaternion, stored but not yet used). Its
subclasses define the shape. The only shape so far is the sphere of radius $r$:

$$
V = \tfrac{4}{3}\pi r^3, \qquad m_n = \rho V, \qquad \mathbf I = \tfrac{2}{5} m_n r^2\, \mathbb 1,
\qquad \alpha = 3 V \varepsilon_0 \frac{\varepsilon_r - 1}{\varepsilon_r + 2},
$$

the last being the Clausius-Mossotti polarisability (in matrix form,
$3V\varepsilon_0(\boldsymbol\varepsilon - \mathbb 1)(\boldsymbol\varepsilon + 2\,\mathbb 1)^{-1}$, for
an anisotropic material). `particle.polarisability_tensor(λ, radiation_reaction=True)` adds the
radiation-reaction correction,

$$
\boldsymbol\alpha_\text{eff} = \boldsymbol\alpha \left(\mathbb 1 - i\frac{k^3}{6\pi\varepsilon_0}\boldsymbol\alpha\right)^{-1},
\qquad k = 2\pi/\lambda,
$$

which for a scalar $\alpha$ is `Trap.α_eff`. The sphere's gas-damping tensors in the free-molecular
regime are those of Table I of Martinetz et al., Physical Review E 97, 052112 (2018),
doi:10.1103/PhysRevE.97.052112. With full accommodation the translational rate is exactly
`Trap.Γth`, and the rotational rate is

$$
\Gamma_\text{rot} = \frac{10\, n_g r^2 \sqrt{2\pi m_g k_B T}}{3 m_n} = \frac{5/4}{1 + \pi/8}\,\Gamma_\text{th}.
$$

Non-spherical shapes, their polarisability and drag tensors, and rotational dynamics are planned;
for now the base class raises `NotImplementedError` for anything a shape must supply.

## Bead, Sphere and Trap

`Bead` keeps all of its keywords (`r`, `ρ`, `εr`, `permittivity_model`, `λ`, `Φ`) and attributes,
and gains `material=` and `charge=`. With no material keywords it uses `LEVISIM_DEFAULT_SILICA`;
`ρ`, `εr` or `permittivity_model='Malitson'` build an ad-hoc material with those values. These give
identical numbers:

```python
from levisim.physics import Bead, Sphere
from levisim.physics.materials import FUSED_SILICA
from levisim.core.units import nm

Bead()                                     # == Sphere()
Bead(ρ=2201.0, permittivity_model='Malitson', λ=1550*nm)
Sphere(material=FUSED_SILICA, λ=1550*nm)   # same ρ, εr, m, α as the line above
```

`Trap` still holds its particle as `trap.bead`; `trap.particle` is the same object, and
`Trap(particle=...)` is accepted in place of `Trap(bead=...)`.
