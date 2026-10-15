# `levisim`

There are 2 layers of `levisim` directory because:
- The git repo is called `levisim`
- The python package inside the git repo is also called `levisim`

### Note on importing from `levisim` into Python environments

Importing `levisim` into a Python session (this could be a Python file, a Jupyter notebook, or the Python [REPL](https://en.wikipedia.org/wiki/Read–eval–print_loop) itself) whose 'current working directory' (`os.cwd()`) has inside it also a directory called `levisim` results in an empty import due to "[name shadowing](https://en.wikipedia.org/wiki/Variable_shadowing)". This provides a good reason to keep notebooks in a local `notebooks/` directory!


# Code pathways for calculations of spectra with each QLT method

The aim of this document is to clarify the various code pathways and mathematical data flows that contribute to calculating the optical and mechanical spectra of interest in the multiparticle trap system. As such I will focus on how the key mathematical expressions flow through the functions of `levisim`, without expounding every detail of the calculations (one can just read the actual code for that, of course!).

#### Definitions
- $`N`$ is the total number of mechanical degrees of freedom ('dimensions'), each with 2 quantum operators for phonon annihilation and creation $`\hat{b}_{j_n}, \hat{b}_{j_n}^\dagger`$ for each Cartesian axis of motion $`j \in {x, y, z}`$ of the $`n^\text{th}`$ trapped particle
- $`2N + 2`$ is the total number of quantum operators (including 2 for cavity photon annihilation and creation $`\hat{a}, \hat{a}^\dagger`$)
- $`M_n`$ is the mass of the particle in the $`n^\text{th}`$ trap
- $`\Omega_{j_n}`$ is the resonant frequency for the $`j_n`$ mechanical degree of freedom, as calculated by the linearised physics model of the multiparticle cavity trap
- $`\hat{q}_{j_{n}}`$ is the dimensionless position operator, i.e. in units of the zero-point motion $`q_{j_n, \text{zpf}} \equiv \sqrt{\frac{\hbar}{2} \cdot \frac{1}{M_n\Omega_{j_n}}}`$
- $`\Gamma_{j_n}`$ is the damping / loss rate for the $`j_n`$ mechanical degree of freedom, assumed to be solely due to gas damping (i.e. does not account for optical sources)
- $`\kappa`$ is the optical damping / loss rate for light inside the cavity

#### Pre-calculated things include

> #### for Linear Amplifier Model (LAM) methods

- `drift_matrix`: $`\mathbf{A} \equiv`$ matrix of coupling and damping frequencies for all creation and annihilation operators (i.e. the coefficients in the $`2N +2`$ Heisenberg-Langevin equations). Crucially, this is static in time and thus is not a spectral quantity i.e. it does not functionally vary with $`\omega`$ even in Fourier transformed coordinates.

- `T_matrix`: $`\mathbf{T}(\omega) \equiv (-i\omega + \mathbf{A})^{-1}`$ is the 'transform' matrix that inverts the drift matrix after appropriate shifting in frequency space due to Fourier transformation

- `c_in`: $`\hat{\mathbf{c}}_{\text{in}}(\omega) \equiv (\sqrt{\kappa}\tilde{a}_{\text{in}}(\omega), \ \sqrt{\kappa}\tilde{a}_{\text{in}}^\dagger(\omega), \ \ldots, \ \sqrt{\Gamma_{j_n}}\tilde{b}_{j_n, \text{in}}(\omega), \sqrt{\Gamma_{j_n}}\tilde{b}_{j_n, \text{in}}^\dagger(\omega), \ \ldots)^\top`$ is the column vector of input noises multiplied by the appropriate damping / loss rate

- `noise_matrix`: $`\mathbf{N} \equiv \mathrm{diag} (\kappa (n_{\text{c}} + 1), \ \kappa n_{\text{c}}, \ \ldots, \ \Gamma_{j_n}(n_{j_n} + 1), \ \Gamma_{j_n} n_{j_n}, \ \ldots)`$, where $`n_\text{c}`$ is the shot noise (thermal photons inside the cavity, typically set to zero) and $`n_{j_n}`$ is the number of thermal phonons in the given degree of freedom

> #### for hybridisation methods

- `x1D_noise_vector`: $`\mathbf{\hat{x}}_{\text{1D}}(\omega) \equiv (\ldots, \hat{q}_{j_n, \text{1D}}(\omega), \ \ldots) =`$ solutions to the Heisenberg-Langevin equations for each dimension treated separately

- `derangement_matrix`:
```math
\mathbf{D}(\omega)
\equiv
\mathbf{I} - \mathbf{R}(\omega)
=
\begin{pmatrix}
  1 & -R_{ab}(\omega) & -R_{ac}(\omega) & \ldots
  \\
  -R_{ba}(\omega) & 1 & -R_{bc}(\omega) & \ldots
  \\
  -R_{ca}(\omega) & -R_{cb}(\omega) & 1 & \ldots
  \\
  \ldots & \ldots & \ldots & \ldots
\end{pmatrix}
```
has 1s on diagonal and (the negative of) the hybridisation functions $`R_{j_n k_m}(\omega)`$ (shown in the matrix with subscript relabelling to $`abc`$... for simplicity) that relates the 1D solutions of the $`k_m`$ degree of freedom to the $`j_n`$ degree of freedom via $`\hat{q}_{j_n}(\omega) = \sum_{k_m \neq j_n} R_{j_n k_m}(\omega) \hat{q}_{k_m}(\omega) + \hat{q}_{j_n, \text{1D}}(\omega)`$

---

#### Notes on notation

- `next function / bit of code`

$`\Uparrow`$

- `function / bit of code` $`\leftarrow`$ $`mathematical \ expressions \ for \ the \ code`$

The above indicates that the mathematical expression $`maths`$ is calculated in the  `function / bit of code` and passed up to the `next function / bit of code` above.

- $`A[B]`$ indicates the $`B`$ component of the vector $`A`$ (usually a noise vector, where each component is an input noise e.g. $`\hat{x}_1[\hat{a}_\text{in}^\dagger]`$ refers to the input cavity photon creation noise's contribution to the spectral value of the $`x_1`$ position)

- $`\hat{h}`$ : hats indicate operator-valued quantities

- $`\tilde{t}`$ : tildes indicates an input noise operator, i.e. a stochastic entity that needs multiplication by something with dimensions of $`\sqrt{\omega}`$ to be equal to the degree of freedom of noise it represents e.g. the combined term $`\sqrt{\kappa} \tilde{a}_\text{in}(\omega)`$ contributes noise to the spectral value of $`\hat{a}(\omega)`$

- I will assume from now that all operator, noise, and matrix quantites are spectral (with functional dependence on $`\omega`$) unless otherwise stated

- $`\mathbb{M}_{\text{TOT}}`$ is `M_norm` in the code, and the overlined transduction functions $`\overline{\mathbb{F}}_{a}, \overline{\mathbb{F}}_{\mathrm{Q}}`$... are the `FFa`, `FFQ`... functions (the doubled `F` standing in for the overline). $`\mathbb{F}`$ without an overline transduces an input noise into $`\mathrm{\hat{Q}}_\phi`$, and $`\overline{\mathbb{F}}`$ into $`\mathrm{\hat{Q}}^\dagger_\phi`$

- $`\chi_{j_n}(\omega)`$ is the mechanical susceptibility of the $`j_n`$ dimension (`dim.χ`), and $`\mu_{j_n}(\omega) \equiv \chi_{j_n}(\omega) - \chi^*_{j_n}(-\omega)`$ their difference (`dim.μm`), which is what multiplies the light quadrature in a hybridised position

---

## (1) `QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL`
All calculation routes under this method start with application of the $`\mathbf{T}`$ matrix onto the $`\hat{\mathbf{c}}_\text{in}`$ input noise vector, giving the operator vector $`\hat{\mathbf{c}}`$, where each entry is itself a noise vector. Thus, this method _linearly amplifies_ the noises into the physical solutions.

> ### `homodyne_spectrum` & `heterodyne_spectrum`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

- `homodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}^\dagger e^{i\theta_\text{hom}} + \hat{a}_\text{out}e^{-i\theta_\text{hom}}`$; $`\theta_\text{hom} \equiv`$ homodyning angle or phase
- `heterodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}`$ *should be a sum of CSDs incorporating the beating frequency $`\Omega_\text{het}`$

$`\Uparrow`$

- `output_light_noise_vectors` $`\leftarrow`$ $`\begin{cases} \hat{a}_{\text{out}} \equiv -\sqrt{\kappa}\hat{a} + \tilde{a}_{\text{in}} & \\ \hat{a}_{\text{out}}^\dagger \equiv -\sqrt{\kappa}\hat{a}^\dagger + \tilde{a}_{\text{in}}^\dagger &\end{cases}`$

$`\Uparrow`$

- `light_noise_vectors` $`\leftarrow`$ $`\begin{cases} \hat{a} = \hat{\mathbf{c}}[\hat{a}] & \\ \hat{a}^\dagger = \hat{\mathbf{c}}[\hat{a}^\dagger]  &\end{cases}`$

$`\Uparrow`$

- `c_noise_vector` $`\leftarrow`$ $`\hat{\mathbf{c}}`$

$`\Uparrow`$

- `T_matrix @ c_in` $`\leftarrow`$ $`\mathbf{T} \cdot \hat{\mathbf{c}}_\text{in}`$

> ### `all_position_power_spectra`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

-  `hybridised_positions` $`\leftarrow`$ all $`\hat{q}_{j_n} \equiv (\hat{b}_{j_n}^\dagger + \hat{b}_{j_n})`$ $`\leftarrow`$ all $`\begin{cases} \hat{b}_{j_n} = \hat{\mathbf{c}}[\hat{b}_{j_n}] & \\ \hat{b}_{j_n}^\dagger = \hat{\mathbf{c}}[\hat{b}_{j_n}^\dagger] &\end{cases}`$

$`\Uparrow`$

- `c_noise_vector` $`\leftarrow`$ $`\hat{\mathbf{c}}`$

$`\Uparrow`$

- `T_matrix @ c_in` $`\leftarrow`$ $`\mathbf{T} \cdot \hat{\mathbf{c}}_\text{in}`$

> ### `position_power_spectrum` / `position_cross_spectrum`
- `noise_vector_to_power_spectrum` / `noise_vectors_to_cross_spectrum`

$`\Uparrow`$

- `hybridised_position_noise_vector` $`\leftarrow`$ only one at a time at a time (inefficient!) $`\hat{q}_{j_n} \equiv (\hat{b}_{j_n}^\dagger + \hat{b}_{j_n})`$

$`\Uparrow`$

- `hybridised_positions[label]` $`\leftarrow`$ all $`\begin{cases} \hat{b}_{j_n} = \hat{\mathbf{c}}[\hat{b}_{j_n}] & \\ \hat{b}_{j_n}^\dagger = \hat{\mathbf{c}}[\hat{b}_{j_n}^\dagger] &\end{cases}`$

$`\Uparrow`$

- `c_noise_vector` $`\leftarrow`$ $`\hat{\mathbf{c}}`$

$`\Uparrow`$

- `T_matrix @ c_in` $`\leftarrow`$ $`\mathbf{T} \cdot \hat{\mathbf{c}}_\text{in}`$

---

## (2) `QuantumLinearTheoryMethod.HYBRID LINEAR_AMPLIFIER_MODEL`

This method is characterised again by starting with the linear amplification of the noises $`\hat{\mathbf{c}} = \mathbf{T}\cdot\hat{\mathbf{c}}_\text{in}`$, but then it deviates by using the phonon entries $`\hat{b}_{j_n, \text{in}}, \hat{b}_{j_n, \text{in}}^\dagger`$ to calculate the light operators $`\hat{a}, \hat{a}^\dagger`$. Thus only the calculation of optical spectra here differs as compared to method (1) above.

> ### `homodyne_spectrum` & `heterodyne_spectrum`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

- `homodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}^\dagger e^{i\theta_\text{hom}} + \hat{a}_\text{out}e^{-i\theta_\text{hom}}`$
- `heterodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}`$

$`\Uparrow`$

- `output_light_noise_vectors` $`\leftarrow`$ $`\begin{cases} \hat{a}_{\text{out}} \equiv -\sqrt{\kappa}\hat{a} + \tilde{a}_{\text{in}} & \\ \hat{a}_{\text{out}}^\dagger \equiv -\sqrt{\kappa}\hat{a}^\dagger + \tilde{a}_{\text{in}}^\dagger &\end{cases}`$

$`\Uparrow`$

- `light_noise_vectors` $`\leftarrow`$ $`\begin{cases} \hat{a} = \chi_{\text{c}}(\omega)[i \sum_{j_n}G_{j_n} \hat{q}_{j_n} + \sqrt{\kappa}\tilde{a}_{\text{in}}] & \\ \hat{a}^\dagger = \chi^*_{\text{c}}(-\omega)[-i \sum_{j_n}G^*_{j_n} \hat{q}_{j_n} + \sqrt{\kappa}\tilde{a}_{\text{in}}^\dagger] &\end{cases}`$

$`\Uparrow`$

-  `hybridised_positions` $`\leftarrow`$ all $`\hat{q}_{j_n} \equiv (\hat{b}_{j_n}^\dagger + \hat{b}_{j_n})`$ $`\leftarrow`$ all $`\begin{cases} \hat{b}_{j_n} = \hat{\mathbf{c}}[\hat{b}_{j_n}] & \\ \hat{b}_{j_n}^\dagger =\hat{\mathbf{c}}[\hat{b}_{j_n}^\dagger] &\end{cases}`$

$`\Uparrow`$

- `c_noise_vector` $`\leftarrow`$ $`\hat{\mathbf{c}}`$

$`\Uparrow`$

- `T_matrix @ c_in` $`\leftarrow`$ $`\mathbf{T} \cdot \hat{\mathbf{c}}_\text{in}`$

> ### `all_position_power_spectra`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

-  `hybridised_positions` $`\leftarrow`$ all $`\hat{q}_{j_n} \equiv (\hat{b}_{j_n}^\dagger + \hat{b}_{j_n})`$ $`\leftarrow`$ all $`\begin{cases} \hat{b}_{j_n} = \hat{\mathbf{c}}[\hat{b}_{j_n}] & \\ \hat{b}_{j_n}^\dagger = \hat{\mathbf{c}}[\hat{b}_{j_n}^\dagger] &\end{cases}`$

$`\Uparrow`$

- `c_noise_vector` $`\leftarrow`$ $`\hat{\mathbf{c}}`$

$`\Uparrow`$

- `T_matrix @ c_in` $`\leftarrow`$ $`\mathbf{T} \cdot \hat{\mathbf{c}}_\text{in}`$

> ### `position_power_spectrum` / `position_cross_spectrum`
- `noise_vector_to_power_spectrum` / `noise_vectors_to_cross_spectrum`

$`\Uparrow`$

- `hybridised_position_noise_vector` $`\leftarrow`$ only one at a time (inefficient!) $`\hat{q}_{j_n} \equiv (\hat{b}_{j_n}^\dagger + \hat{b}_{j_n})`$

$`\Uparrow`$

- `hybridised_positions[label]` $`\leftarrow`$ all $`\begin{cases} \hat{b}_{j_n} = \hat{\mathbf{c}}[\hat{b}_{j_n}] & \\ \hat{b}_{j_n}^\dagger = \hat{\mathbf{c}}[\hat{b}_{j_n}^\dagger] &\end{cases}`$

$`\Uparrow`$

- `c_noise_vector` $`\leftarrow`$ $`\hat{\mathbf{c}}`$

$`\Uparrow`$

- `T_matrix @ c_in` $`\leftarrow`$ $`\mathbf{T} \cdot \hat{\mathbf{c}}_\text{in}`$

---

## (3) `QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL`
This method is unique in that it is only able to calculate position spectra all at once (and not optical spectra at all, since it does not account for input-output theory) as it goes straight to the array of all spectral densities (power and cross) $`S_{\mathbf{c}\hat{\mathbf{c}}}`$ via the 'TNT' expression that effectively diagonalises the systems noises according to its damping and coupling rates. Since the array contains the PSDs and CSDs of the quantum operators (and not the quadrature operators), both the diagonal terms and off-diagonal terms need to be added to get the positions.

> ### `all_position_power_spectra`
- `position_power_spectra_array` $`\leftarrow`$ all $`S_{j_n j_n} = \overbrace{S_{\mathbf{c} \mathbf{c}^\dagger}[\hat{b}_{j_n}, \hat{b}_{j_n}] + S_{\mathbf{c} \mathbf{c}^\dagger}[\hat{b}_{j_n}^\dagger, \hat{b}_{j_n}^\dagger]}^\text{diagonal terms} + \overbrace{S_{\mathbf{c} \mathbf{c}^\dagger}[\hat{b}_{j_n}^\dagger, \hat{b}_{j_n}] + S_{\mathbf{c} \mathbf{c}^\dagger}[\hat{b}_{j_n}, \hat{b}_{j_n}^\dagger]}^\text{off-diagonal terms}`$

$`\Uparrow`$

- `quantum_operator_spectral_density_array` $`\leftarrow`$ $`S_{\mathbf{c} \mathbf{c}^\dagger} \equiv \mathbf{T} \mathbf{N} \mathbf{T}^\dagger`$

$`\Uparrow`$

- `T_matrix @ N_matrix @ T_matrix.H` $`\leftarrow`$ $`\mathbf{T} \mathbf{N} \mathbf{T}^\dagger`$


---

## (4) `QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION`
This method is characterised by the numerical matrix inversion of the derangement matrix $`\mathbf{D}`$, that gives the hybridisation matrix $`\mathbf{H}`$. Since the matrix that is inverted here is $`N \times N`$, it is typically faster than the $`(2N + 2) \times (2N + 2)`$ matrix inversion of the linear amplification methods.

> ### `homodyne_spectrum` & `heterodyne_spectrum`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

- `homodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}^\dagger e^{i\theta_\text{hom}} + \hat{a}_\text{out}e^{-i\theta_\text{hom}}`$
- `heterodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}`$

$`\Uparrow`$

- `output_light_noise_vectors` $`\leftarrow`$ $`\begin{cases} \hat{a}_{\text{out}} \equiv -\sqrt{\kappa}\hat{a} + \tilde{a}_{\text{in}} & \\ \hat{a}_{\text{out}}^\dagger \equiv -\sqrt{\kappa}\hat{a}^\dagger + \tilde{a}_{\text{in}}^\dagger &\end{cases}`$

$`\Uparrow`$

- `light_noise_vectors` $`\leftarrow`$ $`\begin{cases} \hat{a} = \chi_{\text{c}}[i \sum_{j_n}G_{j_n} \hat{q}_{j_n} + \sqrt{\kappa}\tilde{a}_{\text{in}}] & \\ \hat{a}^\dagger = \chi^*_{\text{c}}(-\omega)[-i \sum_{j_n}G^*_{j_n} \hat{q}_{j_n} + \sqrt{\kappa}\tilde{a}_{\text{in}}^\dagger] &\end{cases}`$

$`\Uparrow`$

- `hybridised_positions` $`\leftarrow`$ all $`\hat{q}_{j_n}`$ via $`\mathbf{\hat{x}} \equiv (\ldots, \hat{q}_{j_n}, \ \ldots)^\top = \mathbf{H} \cdot \mathbf{\hat{x}}_{\text{1D}}`$

$`\Uparrow`$

- `hybridisation_matrix` $`\leftarrow`$ $`\mathbf{H} \equiv \mathbf{D}^{-1}`$

$`\Uparrow`$

- `np.linalg.inv(derangement_matrix)` $`\leftarrow`$ $`\mathbf{D}^{-1} \equiv (\mathbf{I} - \mathbf{R})^{-1}`$

> ### `all_position_power_spectra`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

- `hybridised_positions` $`\leftarrow`$ all $`\hat{q}_{j_n}`$ via $`\mathbf{\hat{x}} \equiv (\ldots, \hat{q}_{j_n}, \ \ldots)^\top = \mathbf{H} \cdot \mathbf{\hat{x}}_{\text{1D}}`$

$`\Uparrow`$

- `hybridisation_matrix` $`\leftarrow`$ $`\mathbf{H} \equiv \mathbf{D}^{-1}`$

$`\Uparrow`$

- `np.linalg.inv(derangement_matrix)` $`\leftarrow`$ $`\mathbf{D}^{-1} \equiv (\mathbf{I} - \mathbf{R})^{-1}`$

> ### `position_power_spectrum` / `position_cross_spectrum`
- `noise_vector_to_power_spectrum` / `noise_vectors_to_cross_spectrum`

$`\Uparrow`$

- `hybridised_position_noise_vector`

$`\Uparrow`$

- `hybridised_positions[label]` $`\leftarrow`$ only one at a time (inefficient!) $`\hat{q}_{j_n}`$ via $`\mathbf{\hat{x}} \equiv (\ldots, \hat{q}_{j_n}, \ \ldots)^\top = \mathbf{H} \cdot \mathbf{\hat{x}}_{\text{1D}}`$

$`\Uparrow`$

- `hybridisation_matrix` $`\leftarrow`$ $`\mathbf{H} \equiv \mathbf{D}^{-1}`$

$`\Uparrow`$

- `np.linalg.inv(derangement_matrix)` $`\leftarrow`$ $`\mathbf{D}^{-1} \equiv (\mathbf{I} - \mathbf{R})^{-1}`$

---

## (5) `QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION`
This methods is characterised by *symbolic* matrix inversion that uses combinatorial solutions for the determinant and adjugate matrix of the derangement matrix involving power series of hybridisation functions. This method is unique in the ability calculate a single position without calculating any others, by picking out a particular row of the adjugate matrix. However, this method becomes incredibly slow for $`N > 6`$, since the symbolic inversion requires factorial scaling $`O(N!)`$.


> ### `homodyne_spectrum` & `heterodyne_spectrum`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

- `homodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}^\dagger e^{i\theta_\text{hom}} + \hat{a}_\text{out}e^{-i\theta_\text{hom}}`$
- `heterodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}`$

$`\Uparrow`$

- `output_light_noise_vectors` $`\leftarrow`$ $`\begin{cases} \hat{a}_{\text{out}} \equiv -\sqrt{\kappa}\hat{a} + \tilde{a}_{\text{in}} & \\ \hat{a}_{\text{out}}^\dagger \equiv -\sqrt{\kappa}\hat{a}^\dagger + \tilde{a}_{\text{in}}^\dagger &\end{cases}`$

$`\Uparrow`$

- `light_noise_vectors` $`\leftarrow`$ $`\begin{cases} \hat{a} = \chi_{\text{c}}[i \sum_{j_n}G_{j_n} \hat{q}_{j_n} + \sqrt{\kappa}\tilde{a}_{\text{in}}] & \\ \hat{a}^\dagger = \chi^*_{\text{c}}(-\omega)[-i \sum_{j_n}G^*_{j_n} \hat{q}_{j_n} + \sqrt{\kappa}\tilde{a}_{\text{in}}^\dagger] &\end{cases}`$

$`\Uparrow`$

- `hybridised_positions` $`\leftarrow`$ all $`\hat{q}_{j_n}`$ via $`\mathbf{\hat{x}} \equiv (\ldots, \hat{q}_{j_n}, \ \ldots)^\top = \mathbf{H} \cdot \mathbf{\hat{x}}_{\text{1D}}`$

$`\Uparrow`$

- `hybridisation_matrix` $`\leftarrow`$ $`\mathbf{H} \equiv \mathbf{D}^{-1}`$

$`\Uparrow`$

- `deranged_adjugate_matrix / deranged_determinant` $`\leftarrow`$ $`\mathbf{D}^{-1} \equiv \frac{\mathrm{adj} \ \mathbf{D}}{\det \mathbf{D}}`$

> ### `all_position_power_spectra`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

- `hybridised_positions` $`\leftarrow`$ all $`\hat{q}_{j_n}`$ via $`\mathbf{\hat{x}} \equiv (\ldots, \hat{q}_{j_n}, \ \ldots)^\top = \mathbf{H} \cdot \mathbf{\hat{x}}_{\text{1D}}`$

$`\Uparrow`$

- `hybridisation_matrix` $`\leftarrow`$ $`\mathbf{H} \equiv \mathbf{D}^{-1}`$

$`\Uparrow`$

- `deranged_adjugate_matrix / deranged_determinant` $`\leftarrow`$ $`\mathbf{D}^{-1} \equiv \frac{\mathrm{adj} \ \mathbf{D}}{\det \mathbf{D}}`$

> ### `position_power_spectrum` / `position_cross_spectrum`
- `noise_vector_to_power_spectrum` / `noise_vectors_to_cross_spectrum`

$`\Uparrow`$

- `hybridised_position_noise_vector` $`\leftarrow`$ one $`\hat{q}_{j_n}`$ for PSD, two for CSD

$`\Uparrow`$

- `deranged_adjugate_matrix_row @ x1D_noise_vector / deranged_determinant` $`\leftarrow`$ $`\hat{q}_{j_n} = \frac{[\mathrm{adj} \ \mathbf{D}(\omega)]_{j_n \ \text{row}}}{\det \mathbf{D}} \cdot \mathbf{\hat{x}}_{\text{1D}}`$

---

## (6) `QuantumLinearTheoryMethod.ANALYTIC_INSIDE_CAVITY`
Both analytic methods replace matrix inversion with the closed-form solution for the two phase-dependent collective modes $`\mathrm{\hat{Q}}_\phi, \mathrm{\hat{Q}}^\dagger_\phi`$, so the cost is independent of $`N`$ and, like Power Series H, a single position can be had without solving for any other. This one solves for the intracavity field first and reaches the output field through the input-output relations.

> ### `homodyne_spectrum` & `heterodyne_spectrum`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

- `homodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}^\dagger e^{i\theta_\text{hom}} + \hat{a}_\text{out}e^{-i\theta_\text{hom}}`$
- `heterodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}`$

$`\Uparrow`$

- `output_light_noise_vectors` $`\leftarrow`$ $`\begin{cases} \hat{a}_{\text{out}} \equiv -\sqrt{\kappa}\hat{a} + \tilde{a}_{\text{in}} & \\ \hat{a}_{\text{out}}^\dagger \equiv -\sqrt{\kappa}\hat{a}^\dagger + \tilde{a}_{\text{in}}^\dagger &\end{cases}`$

$`\Uparrow`$

- `light_noise_vectors` $`\leftarrow`$ $`\hat{a}, \hat{a}^\dagger`$ in terms of the collective modes, where $`\bar{g}`$ is the total collective coupling rate (`gbar`)
```math
\hat{a} = \chi_{\text{c}}(\omega)
\begin{bmatrix}
    i\overline{g}\mathbb{M}_{\text{TOT}}^{-1}\mathbb{F}_{a} + \sqrt{\kappa }
    \\ i\overline{g}\mathbb{M}_{\text{TOT}}^{-1}\mathbb{F}_{a^{\dagger }}
    \\ i\overline{g}\mathbb{M}_{\text{TOT}}^{-1}\mathbb{F}_{\mathrm{Q}}
    \\ i\overline{g}\mathbb{M}_{\text{TOT}}^{-1}\mathbb{F}_{\mathrm{Q}^{\dagger }}
\end{bmatrix}
\cdot
\begin{bmatrix}
    \tilde{a}_{\text{in}}
    \\ \tilde{a}_{\text{in}}^{\dagger }
    \\ \tilde{\mathrm{Q}}_{\text{TOT}}
    \\ \tilde{\mathrm{Q}}_{\text{TOT}}^{\dagger}
\end{bmatrix}
```

```math
\hat{a}^{\dagger } = \chi _{\text{c}}^{*}( -\omega )
\begin{bmatrix}
    -i\overline{g}\mathbb{M}_{\text{TOT}}^{-1}\overline{\mathbb{F}}_{a}
    \\ -i\overline{g}\mathbb{M}_{\text{TOT}}^{-1}\overline{\mathbb{F}}_{a^{\dagger }} +\sqrt{\kappa}
    \\ -i\overline{g}\mathbb{M}_{\text{TOT}}^{-1}\overline{\mathbb{F}}_{\mathrm{Q}}
    \\ -i\overline{g}\mathbb{M}_{\text{TOT}}^{-1}\overline{\mathbb{F}}_{\mathrm{Q}^{\dagger }}
\end{bmatrix}
\cdot
\begin{bmatrix}
    \tilde{a}_{\text{in}}
    \\ \tilde{a}_{\text{in}}^{\dagger }
    \\ \tilde{\mathrm{Q}}_{\text{TOT}}
    \\ \tilde{\mathrm{Q}}_{\text{TOT}}^{\dagger }
\end{bmatrix}
```

> ### `all_position_power_spectra`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

- `hybridised_positions` $`\leftarrow`$ each row filled in turn by `hybridised_position_noise_vector`
    - unlike every hybridisation method, there is no matrix to invert, so the rows are independent of one another

$`\Uparrow`$

- `hybridised_position_noise_vector` $`\leftarrow`$ $`\hat{q}_{j_n} = ig_{j_n} \mu_{j_n} \hat{C}(\phi_n) + \sqrt{\Gamma_{j_n}} \tilde{\mathrm{Q}}_{\text{THERM}, j_n}`$
    - $`\hat{C}(\phi_n) \equiv \hat{a}^\dagger e^{i\phi_n} + \hat{a}e^{-i\phi_n}`$ is the quadrature of cavity light this dimension couples to

$`\Uparrow`$

- `light_noise_vectors` $`\leftarrow`$ $`\hat{a}, \hat{a}^\dagger`$ as above

> ### `position_power_spectrum` / `position_cross_spectrum`
- `noise_vector_to_power_spectrum` / `noise_vectors_to_cross_spectrum`

$`\Uparrow`$

- `hybridised_position_noise_vector` $`\leftarrow`$ one $`\hat{q}_{j_n}`$ for PSD, two for CSD
    - as for Power Series H, no other dimension's solution is needed - but here because the collective-mode solution is closed-form rather than because a single adjugate row can be picked out

$`\Uparrow`$

- `light_noise_vectors` $`\leftarrow`$ $`\hat{a}, \hat{a}^\dagger`$ as above

---

## (7) `QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT`
The same closed-form collective-mode solution as (6), but arranged the other way round: the *output* field is written directly in terms of the input noises, and `light_noise_vectors` recovers the intracavity field by reversing the input-output relations. The two should agree everywhere, and are compared against each other in `tests/test_noise_vector_equivalence.py`.

> ### `homodyne_spectrum` & `heterodyne_spectrum`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

- `homodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}^\dagger e^{i\theta_\text{hom}} + \hat{a}_\text{out}e^{-i\theta_\text{hom}}`$
- `heterodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}`$

$`\Uparrow`$

- `output_light_noise_vectors` $`\leftarrow`$ each component of $`\hat{a}_{\text{out}}`$ written straight onto the $`2N + 2`$ input noises

```math
\begin{aligned}
\hat{a}_{\text{out}}[\tilde{a}_{\text{in}}]
    &= 1 - \kappa\chi_{\text{c}}
       - i\bar{g}\sqrt{\kappa}\,\chi_{\text{c}}\,\mathbb{M}_{\text{TOT}}^{-1}\mathbb{F}_{a}
\\
\hat{a}_{\text{out}}[\tilde{a}^{\dagger}_{\text{in}}]
    &= - i\bar{g}\sqrt{\kappa}\,\chi_{\text{c}}\,\mathbb{M}_{\text{TOT}}^{-1}\mathbb{F}_{a^{\dagger}}
\\
\hat{a}_{\text{out}}[\tilde{b}_{j_n,\text{in}}]
    &= -i\sqrt{\kappa}\,\chi_{\text{c}}\,\mathbb{M}_{\text{TOT}}^{-1}
       \left(\mathbb{F}_{\mathrm{Q}}e^{i\phi_n} + \mathbb{F}_{\mathrm{Q}^{\dagger}}e^{-i\phi_n}\right)
       \sqrt{\Gamma_{j_n}}\,g_{j_n}\,\chi_{j_n}(\omega)
\\
\hat{a}_{\text{out}}[\tilde{b}^{\dagger}_{j_n,\text{in}}]
    &= -i\sqrt{\kappa}\,\chi_{\text{c}}\,\mathbb{M}_{\text{TOT}}^{-1}
       \left(\mathbb{F}_{\mathrm{Q}}e^{i\phi_n} + \mathbb{F}_{\mathrm{Q}^{\dagger}}e^{-i\phi_n}\right)
       \sqrt{\Gamma_{j_n}}\,g_{j_n}\,\chi^{*}_{j_n}(-\omega)
\end{aligned}
```

```math
\begin{aligned}
\hat{a}^{\dagger}_{\text{out}}[\tilde{a}_{\text{in}}]
    &= + i\bar{g}\sqrt{\kappa}\,\chi^{*}_{\text{c}}(-\omega)\,\mathbb{M}_{\text{TOT}}^{-1}\overline{\mathbb{F}}_{a}
\\
\hat{a}^{\dagger}_{\text{out}}[\tilde{a}^{\dagger}_{\text{in}}]
    &= 1 - \kappa\chi^{*}_{\text{c}}(-\omega)
       + i\bar{g}\sqrt{\kappa}\,\chi^{*}_{\text{c}}(-\omega)\,\mathbb{M}_{\text{TOT}}^{-1}\overline{\mathbb{F}}_{a^{\dagger}}
\\
\hat{a}^{\dagger}_{\text{out}}[\tilde{b}_{j_n,\text{in}}]
    &= +i\sqrt{\kappa}\,\chi^{*}_{\text{c}}(-\omega)\,\mathbb{M}_{\text{TOT}}^{-1}
       \left(\overline{\mathbb{F}}_{\mathrm{Q}}e^{i\phi_n} + \overline{\mathbb{F}}_{\mathrm{Q}^{\dagger}}e^{-i\phi_n}\right)
       \sqrt{\Gamma_{j_n}}\,g_{j_n}\,\chi_{j_n}(\omega)
\\
\hat{a}^{\dagger}_{\text{out}}[\tilde{b}^{\dagger}_{j_n,\text{in}}]
    &= +i\sqrt{\kappa}\,\chi^{*}_{\text{c}}(-\omega)\,\mathbb{M}_{\text{TOT}}^{-1}
       \left(\overline{\mathbb{F}}_{\mathrm{Q}}e^{i\phi_n} + \overline{\mathbb{F}}_{\mathrm{Q}^{\dagger}}e^{-i\phi_n}\right)
       \sqrt{\Gamma_{j_n}}\,g_{j_n}\,\chi^{*}_{j_n}(-\omega)
\end{aligned}
```

The $`1 - \kappa\chi_{\text{c}}`$ terms are what is left of $`\hat{a}_{\text{out}} = -\sqrt{\kappa}\hat{a} + \tilde{a}_{\text{in}}`$ once $`\hat{a}[\tilde{a}_{\text{in}}] = \sqrt{\kappa}\chi_{\text{c}}`$ is substituted in - the input noise that reflects straight off the cavity rather than entering it. The $`e^{\pm i\phi_n}`$ signs follow `tweezer_phase_sign_convention`; the expressions above are written for the $`+`$ convention.

> ### `all_position_power_spectra`
- identical to (6) above - the positions do not depend on which side of the input-output relations was solved first, so this method reaches them through the same `hybridised_positions` $`\rightarrow`$ `hybridised_position_noise_vector` $`\rightarrow`$ `light_noise_vectors` chain

$`\Uparrow`$

- `light_noise_vectors` $`\leftarrow`$ $`\hat{a}, \hat{a}^\dagger`$ recovered by *reversing* the relations above, i.e. $`\hat{a} = (\tilde{a}_{\text{in}} - \hat{a}_{\text{out}})/\sqrt{\kappa}`$
    - pointlessly convoluted, as noted in the noise-vector section below, but it does mean the two analytic methods exercise different code and so can be checked against each other

> ### `position_power_spectrum` / `position_cross_spectrum`
- identical to (6) above

---
---

# Code pathways for calculations of noise vectors with each QLT method


$`N \equiv`$ number of dimensions

$`2N + 2 =`$ number of input noises (2 per dimension, + 2 from the shared cavity light)

$`F \equiv`$ number of $`\omega`$ values in `ω_range` property

---

## (1) `hybridisation_matrix`
#### $`\Rightarrow (N, N)`$-shaped `numpy.matrix`: matrix of coefficents of 1D position contributions to fully hybridised positions i.e. each entry is a hybridisation solution $`\mathbf{H} \equiv (h_{j_n k_m})`$
- **Numerical Hybridisation**
    - Directly inverts the derangement matrix
    - i.e. $`\mathbf{H} \equiv \mathbf{D}^{-1} \equiv (\mathbf{I} - \mathbf{R})^{-1}`$
    - $`\mathbf{R}`$ is a matrix with
        - $`0`$'s on the diagonal
        - hybridisation functions $`R_{j_n k_m}`$ on all off-diagonals
    - $`\mathbf{I}`$ the identity matrix
- **Power Series Hybridisation**
    - Constructs **two** power series of hybridisation terms
        - One power series for the **determinant** $`\det (\mathbf{D})`$
        - One power series for **adjugate matrix** $`\mathrm{adj}(\mathbf{D})`$
            - Defined as the **transpose of the matrix of cofactors**
            - A **cofactor** is the determinant of a matrix's **minor**
                - A **minor** is the matrix left over when deleting a given row and given column of a given matrix 
    - Divides the adjugate matrix by the determinant
    - i.e. $`\mathbf{D}^{-1} \equiv \frac{\mathrm{adj}(\mathbf{D})}{\det (\mathbf{D})}`$
- **ALL OTHERS METHODS**
    - _not relevant_

---

## (2) `hybridised_positions`
#### $`\Rightarrow (N, 2N + 2)`$-shaped `numpy.ndarray`: column vector of each dimension's position operator as a noise vector
- **LAM** _(but not TNT)_
    - Takes the relevant $`\hat{b}_{j_n}, \hat{b}_{j_n}^\dagger`$ entries from `c_noise_vector`
    - i.e. $`\hat{\mathbf{c}} = \mathbf{T} \cdot \hat{\mathbf{c}}_{\text{in}}`$
        - $`\mathbf{T} \equiv (-i\omega + \mathbf{A})^{-1}`$
        - $`\mathbf{A} \equiv`$ "drift" matrix of couplings
- **Hybridisation**
    - Multiplies `hybridisation_matrix @ x1D_noise_vector`
    - i.e. $`\mathbf{\hat{x}} = \mathbf{H} \cdot \mathbf{\hat{x}}_\text{1D}`$
- **Analytic**
    - Fills out each entry via `hybridised_position_noise_vector`

---

## (3) `hybridised_position_noise_vector`
#### $`\Rightarrow (2N + 2,)`$-shaped `numpy.ndarray`: noise vector of a given dimension's position operator
- **Analytic**
    - Takes $`\hat{a},\hat{a}^\dagger`$ from `light_noise_vectors` to directly calculate $`\hat{q}_{j_n} = ig_{j_n} \mu_{j_n} \hat{C}(\phi_n) + \sqrt{\Gamma} \tilde{Q}_{\text{THERM}}`$
        - $`\hat{C}(\phi_n) \equiv \hat{a}^\dagger e^{i\phi_n} + \hat{a}e^{-i\phi_n}`$
            - using the $`+`$ sign convention for the per-tweezer phase
- **Power Series Hybridisation**
    - Constructs a particular row of the adjugate matrix
    - Then does `adj_row @ x1D_noise_vector / deranged_determinant`
    - i.e. $`\hat{q}_{j_n} = \sum_{k_m} h_{j_n k_m} \hat{q}_{k_m, \text{1D}}`$
- **ALL OTHER METHODS**: _(LAM, Hybrid LAM, and Numerical H)_
    - Calculates the full `hybridised_positions` and simply takes the relevant entry from it

---

## (4) `light_noise_vectors`
#### $`\Rightarrow`$ `2-tuple` of $`(2N + 2,)`$-shaped `numpy.ndarray`s: noise vectors of the annihilation and creation operators $`\hat{a}, \hat{a}^\dagger`$ of the shared cavity light mode
- **Analytic Inside Cavity**
    - Takes the phase-dependent collective position operator $`\mathrm{\hat{Q}}_\phi \equiv \sum_{j_n} \frac{g_{j_n}}{\bar{g}} e^{i\phi_n} \hat{q}_{j_n} = \mathbb{M}_{\text{TOT}}^{-1}[\mathbb{F}_a \tilde{a}_{\text{in}} + \ldots]`$ etc.
    - Plugs this into $`\hat{a} = \chi_{\text{c}}(\omega)[i\bar{g}\mathrm{\hat{Q}}_\phi + \sqrt{\kappa}\tilde{a}_{\text{in}}]`$ etc.
- **Analytic Cavity Output**
    - Takes $`\hat{a}_{\text{out}}, \hat{a}_{\text{out}}^\dagger`$ from `output_light_noise_vectors` and _reverses_ the input-output relations (this is obviously pointlessly convoluted but it should still work!)
- **LAM** _(but not Hybrid, nor TNT)_
    - Takes the relevant $`\hat{a}, \hat{a}^\dagger`$ entries from `c_noise_vector`
    - i.e. $`\hat{\mathbf{c}} = \mathbf{T} \cdot \hat{\mathbf{c}}_{\text{in}}`$
- **ALL OTHER METHODS**: _(Hybrid LAM, Numerical H and Power Series H)_
    - Calculate $`\hat{a} = \chi_{\text{c}}(\omega)[i \sum_{j_n} G_{j_n} \hat{q}_{j_n} + \sqrt{\kappa} \tilde{a}_{\text{in}}]`$
    - $`G_{j_n} \equiv g_{j_n} e^{i\phi_n}`$ etc.

---

## (5) `output_light_noise_vectors`
#### $`\Rightarrow`$ `2-tuple` of $` (2N + 2,)`$-shaped `numpy.ndarray`s: noise vectors of the annihilation and creation operators $`\hat{a}_{\text{out}}, \hat{a}^\dagger_{\text{out}}`$ of the output cavity light
- **Analytic Cavity Output**
    - Takes the input-output relations $`\hat{a}_{\text{out}} = -\sqrt{\kappa}\hat{a} + \tilde{a}_{\text{in}}`$ etc.
    - Inserts first $`\hat{a} = \chi_{\text{c}}(\omega)[i\bar{g}\mathrm{\hat{Q}}_\phi + \ldots]`$ etc.
    - Then $`\mathrm{\hat{Q}}_\phi = \mathbb{M}_{\text{TOT}}^{-1}[\mathbb{F}_a \tilde{a}_{\text{in}} + \ldots]`$ etc.
    - Gives $`\hat{a}_{\text{out}}`$ in terms of $`\mathbb{M}_{\text{TOT}}, \mathbb{F}_a,\ldots`$ etc.
- **ALL OTHER METHODS**: _(all LAM, both hybridisation, & analytic inside cavity)_
    - take $`\hat{a}, \hat{a}^\dagger`$ from `light_noise_vectors` and applies the input-output relations

---

## (6) `all_position_power_spectra` (not a function of $`\omega`$)
#### $`\Rightarrow`$ `(psd_array, psd_dict)`
##### `psd_array` is a $`(F, N)`$-shaped `numpy.ndarray` with all the PSD values for each dimension's position at each frequency value
##### `psd_dict` is a $`N`$-lengthed `dict` with each entry's key being the dimension label `str`, and each value being a $`(F,)`$-shaped `numpy.ndarray` of its position operator values at each frequency
- **TNT LAM**
    - Calculates `psd_array` _directly_ via "matrix of all spectra" $`S_{\mathbf{c}\mathbf{c}^\dagger} \equiv \mathbf{T}\mathbf{N}\mathbf{T}^\dagger`$
    - $`\mathbf{N} = \mathrm{diag}(\kappa(n_{\text{c}} + 1), \ \kappa n_{\text{c}}, \ \ldots, \Gamma_{j_n}(n_{j_n} + 1), \Gamma_{j_n} n_{j_n}, \ldots)`$ is the "noise" matrix
    - $`n_{\text{c}} \equiv`$ number of noise photons ($`n_{\text{c}} = 0`$ for shot noise)
    - $`n_{j_n} \equiv`$ number of phonons for $`j_n`$ dimension via Bose-Einstein statistics
- **ALL OTHER METHODS**: _(LAM, Hybrid LAM, both hybridisation, and both analytic)_
    - Loops over `ω_range` and, at each $`\omega`$, builds the whole $`(N, 2N + 2)`$ `hybridised_positions` array, then reduces each dimension's row to one PSD value with `noise_vector_to_power_spectrum`
    - i.e. $`S_{q_{j_n}q_{j_n}}(\omega) = \sum_{\text{noises}} \eta_{\text{noise}} \ \big|\hat{q}_{j_n}[\text{noise}]\big|^2`$
        - $`\eta_{\text{noise}} \equiv`$ `quantum_factor_per_noise_operator`, the occupation factors $`(n_{\text{c}} + 1, \ n_{\text{c}}, \ \ldots, n_{j_n} + 1, \ n_{j_n}, \ldots)`$
        - the rates $`\kappa`$ and $`\Gamma_{j_n}`$ do **not** appear here: they are already inside the noise vectors, via the $`\sqrt{\kappa}\tilde{a}_\text{in}`$ and $`\sqrt{\Gamma_{j_n}}\tilde{\mathrm{Q}}_{\text{THERM}}`$ prefactors. This is the one place the TNT route differs in form, since it carries them explicitly in $`\mathbf{N}`$
    - This is why TNT is the only method that can fill `psd_array` without ever constructing a noise vector, and equally why it is the only method that _cannot_ give a single position on its own
- **Note on `zpf_scaled`**
    - $`\hat{q}_{j_n}`$ is dimensionless throughout (in units of the zero-point fluctuation), so `zpf_scaled=True` returns the PSD as calculated
    - `zpf_scaled=False` multiplies each dimension's PSD by $`x_{\text{zpf}, j_n}^2`$ to put it back into $`\text{m}^2/(2\pi\cdot\text{Hz})`$

---

# Planned refactor: mixins for `multitrap`

`MultiTrap` currently branches on `method` inside each calculation function, which is why `multitrap.py` is over 2,000 lines. Mapping the branches out while documenting the above, the split falls fairly cleanly into three mixins plus the parts that are genuinely shared:

- **`LinearAmplifierMixin`** - `(1)`, `(2)`, `(3)`. Owns `drift_matrix`, `T_matrix`, `c_noise_vector`, and the TNT-only `quantum_operator_spectral_density_array` / `position_power_spectra_array`. Takes positions and light operators by slicing $`\hat{\mathbf{c}}`$.
- **`HybridisationMixin`** - `(4)`, `(5)`. Owns `derangement_matrix`, `hybridisation_matrix`, `deranged_determinant`, `deranged_adjugate_matrix` and `x1D_noise_vector`. Numerical and Power Series differ only in _how_ $`\mathbf{D}^{-1}`$ is obtained, so they are two methods on one mixin rather than two mixins.
- **`AnalyticMixin`** - `(6)`, `(7)`. Owns `gbar`, `M_norm`, the $`\mathbb{F}`$ / $`\overline{\mathbb{F}}`$ transduction functions and `Q_phi`. Inside-cavity and cavity-output differ only in which end of the input-output relations is solved first.
- **Shared on the base class** - `dimensions`, `cavity`, `ω_range`, `quantum_factor_per_noise_operator`, `noise_vector_to_power_spectrum`, `noise_vectors_to_cross_spectrum`, and the spectrum entry points that dispatch to a mixin.

The one real obstacle is that `hybridised_position_noise_vector` and `light_noise_vectors` are mutually recursive across mixins for the analytic methods - a position needs the light operators, which for Analytic Cavity Output come from the output operators, which need the collective modes. So the dispatch has to live on the base class rather than inside any one mixin.

# Disclaimer regarding use of LLMs

While in early 2025 I dabbled with using ChatGPT and DeepSeek to give me starting points for certain more generic parts of this code (help with wrappers etc.) the only thing that remains of legitimate value that I gained from using those tools is the first attempt at the `docker-compose.yaml`, which I otherwise would have likely laboured a bit harder to do. Otherwise this code was proudly written without any LLM generated code :-)

### October 2026 update

That remains true of every commit on `main`. In one session I pointed Claude (Anthropic) at this repository, the untracked `notebooks/` folder and the `fortran/` folder, and asked it to propose fixes as pull requests. It opened a stack of them, and **this paragraph and the two sections above it are themselves part of that work** - so the claim above no longer covers the whole repository if any of those branches are merged. Each pull request description carries the same note.

For the record, the things it found that I had not:

- `pyproject.toml` shipped no subpackages, so `pip install levisim` produced a package that raised `ModuleNotFoundError: No module named 'levisim.core'` on import. Working from the repo root hid it completely.
- `import levisim` could not succeed on a fresh clone at all: `shelf.py` built 17 `MultiTrap` instances at import time, several of which read power-series pickles from the untracked `notebooks/`.
- Three latent `NameError`s - `Megasamples` for `megasamples`, a missing `colours_50` import, and a bare `k` where `self.k` was meant - each making its enclosing function raise on any call. None was reachable from anywhere in the repo, which is why they had survived.
- `core/logger.py` added only an Elasticsearch sink and then logged at import, so with no ELK stack running every `import levisim` printed a traceback and **no `logger` call has ever reached the console**. That is why there are 83 `print()` calls in here.
- Nine `cmplx()` calls in the Fortran had no `kind` argument, so every complex susceptibility $`\chi`$ was computed in **single precision** and widened back to double - a median relative error of 2e-7 in the reference data.
- Three of the four build lines in the old `fortran/README.txt` did not compile, because the `num_traps` rename only reached three of the files.
- `physics/kick.py` defined `info` twice, so `Kick.info` returned the string instead of printing it.
- The comparison helpers default to `difficulty_level='very hard'`, a relative-difference threshold of 1e-12 that double-precision linear algebra cannot reach. This is why the test notebooks reported hundreds of failing method pairs that were not real disagreements - though the analytic-vs-rest disagreement on the **spectra** (mean 28-147%) is real, and is the thing the "BROKEN!" heading in `test_all_systems.ipynb` was correctly pointing at.

It also could not fix, and I should: the `kappa_nano (Hz) 1.216e-86` that `NParticle.f90` prints, and the two `NOT SURE: CHECK!!!` susceptibilities in `3DEQRom.f90`.

License chosen through [choosealicense.com](https://choosealicense.com)
