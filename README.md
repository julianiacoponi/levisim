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

# TODO: finish the below

## (6) `QuantumLinearTheoryMethod.ANALYTIC_INSIDE_CAVITY`

> ### `homodyne_spectrum` & `heterodyne_spectrum`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

- `homodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}^\dagger e^{i\theta_\text{hom}} + \hat{a}_\text{out}e^{-i\theta_\text{hom}}`$
- `heterodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}`$

$`\Uparrow`$

- `output_light_noise_vectors` $`\leftarrow`$ $`\begin{cases} \hat{a}_{\text{out}} \equiv -\sqrt{\kappa}\hat{a} + \tilde{a}_{\text{in}} & \\ \hat{a}_{\text{out}}^\dagger \equiv -\sqrt{\kappa}\hat{a}^\dagger + \tilde{a}_{\text{in}}^\dagger &\end{cases}`$

$`\Uparrow`$

- `light_noise_vectors` $`\leftarrow`$ $`\bar{g}`$ $`\overline{g}`$
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

> ### `position_power_spectrum` / `position_cross_spectrum`

---

## (6) `QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT`

> ### `homodyne_spectrum` & `heterodyne_spectrum`
- `noise_vector_to_power_spectrum`

$`\Uparrow`$

- `homodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}^\dagger e^{i\theta_\text{hom}} + \hat{a}_\text{out}e^{-i\theta_\text{hom}}`$
- `heterodyne_noise_vector` $`\leftarrow`$ $`\hat{a}_\text{out}`$

$`\Uparrow`$

# TODO: Correct this
- `output_light_noise_vectors` $`\leftarrow`$

> ### `all_position_power_spectra`

> ### `position_power_spectrum` / `position_cross_spectrum`

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

# TODO: finish (6) for all the other methods

# TODO: refactor `multitrap` so there are mixins for each of LAM, Hybridisation and Analytic methods

# Disclaimer regarding use of LLMs

While in early 2025 I dabbled with using ChatGPT and DeepSeek to give me starting points for certain more generic parts of this code (help with wrappers etc.) the only thing that remains of legitimate value that I gained from using those tools is the first attempt at the `docker-compose.yaml`, which I otherwise would have likely laboured a bit harder to do. Otherwise this code is proudly written without any LLM generated code :-)

License chosen through [choosealicense.com](https://choosealicense.com)
