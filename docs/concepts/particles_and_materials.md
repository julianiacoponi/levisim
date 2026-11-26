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
2018 kg/m³, and $\rho = 1850$ kg/m³ to $\varepsilon_r = 1.875$.

```{note}
The inconsistency is known and deliberate: both values are kept for continuity with the published
and thesis numbers computed with them, so that no existing result moves. For a self-consistent porous
silica, derive a material with a Lorentz-Lorenz pair, e.g.
`FUSED_SILICA.replace(name='porous silica', ρ=1850.0, permittivity=1.875)` (εr at 1064 nm).
```

## Particles

A `Particle` has a material, a charge in units of the elementary charge $e$ (default 0), the
vacuum wavelength $\lambda$ at which a dispersive permittivity is evaluated, the trapping phase
$\Phi$ along the cavity axis, and an orientation (a unit quaternion, see
{ref}`below <orientation>`). Its subclasses define the shape. The simplest is the sphere of radius
$r$:

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

## Non-spherical shapes

{mod}`levisim.physics.shapes` adds five shapes. Each is made of one homogeneous material, has its
symmetry axis along the body $z$ axis, and supplies the volume $V$, the surface area $A$, the mass
$m_n = \rho V$, the body-frame inertia tensor, the body-frame polarisability tensor and the gas
friction and diffusion tensors.

| Class | Defining lengths | Inertia tensor (body frame) |
|---|---|---|
| `Ellipsoid(a, b, c)` | semi-axes along $x, y, z$ | $\tfrac{m_n}{5}\,\mathrm{diag}(b^2 + c^2,\ a^2 + c^2,\ a^2 + b^2)$ |
| `Spheroid(a, b)` | $a$ along the symmetry axis, $b$ equatorial | `Ellipsoid(b, b, a)` |
| `Cylinder(length, radius)`, `Rod` | $\ell$, $R$ | $m_n\,\mathrm{diag}(\ell^2/12 + R^2/4,\ \ell^2/12 + R^2/4,\ R^2/2)$ |
| `Disk(radius, thickness)` | a `Cylinder` with $\ell$ = thickness | as `Cylinder` |
| `Dumbbell(r1, r2, separation)` | two spheres, centres $d \ge r_1 + r_2$ apart | $I_{zz} = \tfrac{2}{5}(m_1 r_1^2 + m_2 r_2^2)$, $I_{xx} = I_{yy} = I_{zz} + \mu d^2$ |

where $\mu = m_1 m_2/(m_1 + m_2)$ is the reduced mass of the two spheres (parallel-axis theorem).
`Spheroid(a, a)` and `Ellipsoid(a, a, a)` reproduce `Sphere(r=a)`, and so `Bead()`, bit for bit: in
that limit they use the sphere's own formulas, alone and inside a `Trap`.

### Polarisability

A homogeneous ellipsoid in a uniform field is uniformly polarised, and its quasi-static
polarisability is (Bohren and Huffman, *Absorption and Scattering of Light by Small Particles*,
Wiley (1983), §5.3)

$$
\boldsymbol\alpha = \varepsilon_0 V (\boldsymbol\varepsilon - \mathbb 1)\left[\mathbb 1 + \mathbf L(\boldsymbol\varepsilon - \mathbb 1)\right]^{-1},
\qquad
\alpha_i = \varepsilon_0 V \frac{\varepsilon_r - 1}{1 + L_i(\varepsilon_r - 1)} \ \text{for a scalar } \varepsilon_r,
$$

with the depolarisation factors $\mathbf L = \mathrm{diag}(L_x, L_y, L_z)$,

$$
L_i = \frac{a_x a_y a_z}{2}\int_0^\infty \frac{ds}{(s + a_i^2)\sqrt{(s + a_x^2)(s + a_y^2)(s + a_z^2)}},
\qquad L_x + L_y + L_z = 1,
$$

which levisim evaluates as Carlson's integral $R_D$. For a spheroid of aspect ratio $p = a/b$ they
have closed forms: for $p > 1$ (prolate), $e^2 = 1 - 1/p^2$ and
$L_\parallel = \frac{1 - e^2}{e^2}\left(\frac{\operatorname{artanh} e}{e} - 1\right)$; for $p < 1$ (oblate),
$e^2 = 1 - p^2$ and $L_\parallel = \frac{1}{e^2}\left(1 - \frac{\sqrt{1 - e^2}\arcsin e}{e}\right)$;
in both cases $L_\perp = (1 - L_\parallel)/2$. At $L_i = 1/3$ the formula is Clausius-Mossotti. A
birefringent material, such as `VATERITE`, enters through its permittivity tensor in the
particle's body frame, in the matrix form above. The radiation reaction is applied to the whole
tensor, $\boldsymbol\alpha_\text{eff} = \boldsymbol\alpha(\mathbb 1 - i k^3\boldsymbol\alpha/6\pi\varepsilon_0)^{-1}$.

Two shapes need an approximation. A finite cylinder is not uniformly polarised, so `Cylinder`,
`Rod` and `Disk` use the depolarisation factors of the spheroid with the same aspect ratio
$\ell/2R$, with the cylinder's own volume; this becomes exact for the infinitely long rod,
$\mathbf L = (\tfrac12, \tfrac12, 0)$, and the infinitely thin disk, $\mathbf L = (0, 0, 1)$. A
`Dumbbell` is two point dipoles at the sphere centres, each with its Clausius-Mossotti
$\boldsymbol\alpha_i$, coupled by the quasi-static near field
$\mathbf G = (3\hat z\hat z - \mathbb 1)/4\pi\varepsilon_0 d^3$:
$\mathbf p_1 = \boldsymbol\alpha_1(\mathbf E + \mathbf G\mathbf p_2)$ and the same with 1 and 2
exchanged. For two equal spheres this gives
$\alpha_\parallel = 2\alpha_s/(1 - 2g)$ and $\alpha_\perp = 2\alpha_s/(1 + g)$, with
$g = \alpha_s/4\pi\varepsilon_0 d^3$. The point-dipole picture leaves out the higher multipoles
that touching spheres induce near their contact, so it underestimates the anisotropy. For exact
values, use a T-matrix or discrete-dipole code and `TabulatedParticle`.

```{figure} figures/spheroid_tensors.png
:width: 100%

Spheroids with the volume of the default bead ($r = 60.1$ nm), against their aspect ratio $a/b$.
Left: polarisability along and across the symmetry axis, relative to the sphere's. Right:
free-molecular gas damping rates of the centre of mass (cm) and of rotation (rot), relative to the
sphere's, for $\alpha_c = \gamma_s = 1$.
```

At aspect ratio 3 the polarisability along the axis is 1.198 times the sphere's and 0.924 times
across it. A prolate spheroid aligned with the tweezer polarisation is therefore trapped more
stiffly than a sphere of the same mass, by $\sqrt{1.198} = 1.095$ in every trap frequency.

(orientation)=
### Orientation and translational trapping

`particle.orientation` is a unit quaternion $(w, x, y, z)$ of the body frame relative to the lab
frame. `particle.rotation_matrix` is the corresponding $\mathbf R$, so that
`lab_polarisability_tensor()` returns $\mathbf R\boldsymbol\alpha\mathbf R^\mathsf{T}$ and
`lab_inertia_tensor` returns $\mathbf R\mathbf I\mathbf R^\mathsf{T}$. The helpers
`axis_angle_quaternion` and `body_axis_to_quaternion` (in {mod}`levisim.physics.particle`) build
orientations; for example, `body_axis_to_quaternion((0, 1, 0))` points a spheroid's symmetry axis
along the tweezer polarisation. The orientation is the reference of the rotational dynamics and
librations described in [Rotation and libration](rotation.md).

Every levisim tweezer propagates along $z$ and is polarised along the lab $y$ axis. For a fixed
orientation, the time-averaged dipole energy in the field $E\hat{\mathbf e}$ is
$-\tfrac14\,\mathrm{Re}(\hat{\mathbf e}\cdot\boldsymbol\alpha_\text{eff}\cdot\hat{\mathbf e})|E|^2$,
because the components of the induced dipole perpendicular to $\hat{\mathbf e}$ do no work. A
`Trap` therefore uses $\alpha_\text{eff} = \hat{\mathbf y}\cdot\mathbf R\boldsymbol\alpha_\text{eff}\mathbf R^\mathsf{T}\cdot\hat{\mathbf y}$,
and its stiffness, trap frequencies, potential, forces, scattering cross section and equilibrium
offset follow from that projection. A sphere of an isotropic material keeps the original scalar
arithmetic, and its numbers are unchanged. `particle.α` is the projected static polarisability,
which `MultiTrap` uses for the coherent-scattering drive.

The photon-recoil noise is generalised in the same way. The induced dipole
$\mathbf p \propto \boldsymbol\alpha_\text{eff}\hat{\mathbf y}$ need not lie along the polarisation.
Its emission pattern $1 - (\hat{\mathbf p}\cdot\mathbf n)^2$ gives the direction weights
$c_j = 2 - \hat p_j^2 + 5\delta_{jz}$, which are $(2, 1, 7)$ for $\hat{\mathbf p} = \hat{\mathbf y}$
(`Tweezer.recoil_coefficients`).

The gas damping of a tilted particle is the lab-frame tensor
$\mathbf R\boldsymbol\Gamma_\text{cm}\mathbf R^\mathsf{T}$ (`Trap.Γth_tensor`). `MultiTrap` and
`FreeSpaceSimulation` use its diagonal for each axis (`Trap.Γth_j`), with the matching thermal force
noise (`Trap.Sth_j`). The off-diagonal elements, which couple the axes through gas friction, are
left out.

### Gas damping

Martinetz, Hornberger and Stickler, Physical Review E 97, 052112 (2018),
doi:10.1103/PhysRevE.97.052112, derive the friction and diffusion tensors of any rigid body in the
free-molecular regime. Gas molecules are reflected either specularly or diffusely. The diffusely
reflected fraction is the accommodation coefficient $\alpha_c$, and those molecules are re-emitted at the
surface temperature $T_s = \gamma_s^2 T$. levisim's defaults, $\alpha_c = \gamma_s = 1$, are those of
`Trap.Γth`. The $6\times6$ friction matrix acting on the momenta $(\mathbf P, \mathbf J)$ is their
Eq. (40),

$$
\boldsymbol\Gamma = n_g\sqrt{\frac{m_g k_B T}{2\pi}}\oint dA
\left\{c_1\begin{pmatrix}\mathbf n\\ \mathbf r\times\mathbf n\end{pmatrix}\otimes\begin{pmatrix}\mathbf n\\ \mathbf r\times\mathbf n\end{pmatrix}
+ \alpha_c\begin{pmatrix}\mathbb 1 & -\hat{\mathbf r}\\ \hat{\mathbf r} & -\hat{\mathbf r}^2\end{pmatrix}\right\}
\begin{pmatrix}m_n\mathbb 1 & 0\\ 0 & \mathbf I\end{pmatrix}^{-1},
\qquad c_1 = 4 - 3\alpha_c + \frac{\pi\alpha_c\gamma_s}{2},
$$

where $\mathbf r$ is the surface point relative to the centre of mass, $\mathbf n$ the outward
normal, $\hat{\mathbf r}\mathbf a = \mathbf r\times\mathbf a$, and $n_g = P/k_B T$ the gas number
density. The diffusion matrix, their Eq. (41), satisfies
$\mathbf D = k_B T\,\boldsymbol\Gamma\,\mathrm{diag}(m_n\mathbb 1, \mathbf I)$ for $\gamma_s = 1$.
Spheres and cylinders use the closed forms of Table I of the paper. Ellipsoids, spheroids and
dumbbells use these surface integrals, evaluated once on a Gauss quadrature and cached
({mod}`levisim.physics.gas_damping`). The tests check the numerical integrals against Table I for
the sphere and the cylinder, to $10^{-12}$, for several values of $\alpha_c$ and $\gamma_s$.
`rotranslational_friction_matrix` and `rotranslational_diffusion_matrix` return the full $6\times6$
matrices. For an unequal dumbbell, these matrices couple rotation and translation.

The model assumes that every surface element receives the full, unshadowed thermal flux. This is
exact for convex bodies. For a dumbbell, the shadowing of each sphere by the other is left out.

## Tabulated particles

levisim does not contain an electromagnetic solver. For a particle that the analytic shapes do not
describe, compute its tensors with a T-matrix, discrete-dipole or finite-element code and pass them
in:

```python
from levisim.physics import TabulatedParticle, Spheroid, Gas

tab = TabulatedParticle(
    mass=m, inertia=[I1, I2, I3],                       # or a 3×3 tensor
    polarisability={'λ': wavelengths, 'α': tensors},    # (n,), (n, 3, 3); or a 3×3 tensor, or α(λ)
    translational_damping=Γ_cm, rotational_damping=Γ_rot, damping_gas=Gas(P=..., T=...),
    volume=V,
)
tab = TabulatedParticle.from_particle(Spheroid(150e-9, 50e-9), gas=Gas())   # tabulate a shape
TabulatedParticle(**tab.to_dict())                                           # round trip
```

The inputs are validated on construction:

- the inertia tensor must be symmetric and positive definite, and obey $I_1 + I_2 \ge I_3$;
- each polarisability tensor must be symmetric (reciprocity) and passive ($\mathrm{Im}\,\boldsymbol\alpha$
  positive semi-definite, with levisim's $e^{-i\omega t}$ convention);
- the damping tensors must have real, non-negative eigenvalues.

Damping tensors given as arrays hold for the accommodation they were computed with. They are scaled
to another gas by the free-molecular law $\Gamma \propto P\sqrt{m_g/T}$. A tabulated polarisability
is interpolated linearly between wavelengths, and requesting a wavelength outside the table raises
an error.

## Beyond the dipole limit: Mie theory

The dipole model holds while $kr \ll 1$. {mod}`levisim.physics.mie`, an optional backend
(`pip install "levisim[mie]"`, which uses `miepython`, MIT licence), evaluates the exact Mie
solution for a sphere. It returns the extinction and scattering cross sections and the
dipole-equivalent polarisability $\alpha_\text{Mie} = 6\pi i\varepsilon_0 a_1/k^3$ from the
electric-dipole coefficient $a_1$. That polarisability includes the radiation reaction exactly.
`compare_with_dipole(sphere)` compares it with levisim's $\alpha_\text{eff}$.

```{figure} figures/mie_vs_dipole.png
:width: 100%

Silica spheres ($\varepsilon_r = 1.98$) at 1064 nm, against the sphere radius. Left: relative
difference between the dipole model and Mie theory in Re α and in the scattering cross section.
Right: fraction of the Mie scattering cross section not in the electric dipole $a_1$. Dotted lines:
$r = 60.1$ nm, 300 nm, 1 μm.
```

| $r$ | $kr$ | $\mathrm{Re}\,\alpha_\text{dip}/\mathrm{Re}\,\alpha_\text{Mie} - 1$ | $\sigma_\text{sca,dip}/\sigma_\text{sca,Mie} - 1$ | $\sigma_\text{sca}$ outside $a_1$ |
|---|---|---|---|---|
| 60.1 nm (default bead) | 0.355 | +1.3 × 10<sup>−3</sup> | +2.3 × 10<sup>−3</sup> | 3.6 × 10<sup>−4</sup> |
| 300 nm | 1.77 | +0.15 | +0.097 | 0.40 |
| 1 μm | 5.9 | −1.06 | −0.95 | 0.98 |

For the default bead the dipole model is good to about $10^{-3}$, much better than the naive
$(kr)^2 \approx 0.13$. The relative $O(x^2)$ correction to $a_1$ is
$\tfrac{3x^2}{5}(\varepsilon_r - 2)/(\varepsilon_r + 2)$, which nearly vanishes for
$\varepsilon_r = 1.98$. For $\varepsilon_r = 4$ at the same size the difference is $-2.4\%$. At
300 nm the higher multipoles carry 40 % of the scattering. At 1 μm the dipole picture fails: the
real part of the dipole coefficient changes sign.

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
