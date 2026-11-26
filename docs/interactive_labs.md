# Interactive 3D web labs

`levisim` includes photorealistic Three.js optical-bench laboratories that compute live motional
and optical spectra directly on an in-game RF spectrum analyzer monitor.

## Launching locally (with live Python `levisim` backend)

```sh
levisim-web                      # opens http://127.0.0.1:8765/ (lab portal)
levisim-web --lab tweezer        # open directly to the free-space tweezer lab
levisim-web --lab cavity         # open directly to the coherent-scattering cavity lab
python -m levisim.web --no-open  # serve without launching a browser tab
```

When served through `levisim-web`, the browser computes interactive 60 FPS previews using exact
JavaScript ports of `levisim`'s equations (cross-checked against Python in `tests/test_web_free_space.py`
and `tests/test_web_cavity.py`) and synchronises with the live Python `levisim` server
(`/api/tweezer/spectra`, `/api/tweezer/simulate`, and `/api/cavity/spectra`) for exact `Trap`,
`FreeSpaceSimulation`, and `MultiTrap` calculations.

## GitHub Pages (zero install)

Every build of this documentation copies `levisim/web/static/` to `labs/`:

```{raw} html
<ul>
  <li><a href="labs/index.html">Interactive Labs Portal</a></li>
  <li><a href="labs/free_space_tweezer/index.html">Free-Space Optical Tweezer Lab</a> (<code>Trap</code>, <code>SpectralDensityTheory</code>, PRR 6)</li>
  <li><a href="labs/coherent_scattering_cavity/index.html">Coherent-Scattering Cavity Trap Lab</a> (<code>MultiTrap</code>, QLT, PRR 5 &amp; PRR 8)</li>
</ul>
```

## Free-space optical tweezer lab

Models a 1064 nm elliptical Gaussian optical tweezer inside a darkroom quantum optics lab with a
sapphire-viewport UHV vacuum chamber, opposing NA 0.77 aspheric objectives, four radial feedback
and directional-force electrodes (PRR 6), N-BK7 glass PBS and 50/50 beam-splitter cubes, balanced
forward-scattering $x$/$y$ photodetectors, and a benchtop RF spectrum analyzer monitor:

- **Direct 3D diegetic controls**: drag the trapping laser head ($P$), the $\lambda/2$ waveplate
  ($w_{x0}$ ellipticity), the four-electrode ring (directed-force angle $\psi$), the vacuum gauge
  ($P_\mathrm{gas}$), or the feedback controller ($\Gamma_\mathrm{fb}$); click the in-game
  monitor to focus the camera onto it.
- **In-game monitor**: switch between **RF Analyzer**, **`levisim` `xmgrace`**, and **Quantum OLED**
  themes; toggle the physical monitor power **ON/OFF**; or open the synced **2D Pop-Out** analyzer
  with frequency crosshair readouts.
- **Spectra**: one-sided motional PSDs $S_{xx}, S_{yy}, S_{zz}$, directional cross-spectra
  $C_{xy}, Q_{xy}$ (showing the quadrant sign-flip across $\psi = \pm 45^\circ$), optional overlaid
  simulated periodogram points, and optical balanced-detector spectra in shot-noise units.

## Coherent-scattering cavity trap lab

Models a high-finesse Fabry-Pérot optical cavity with thick fused-silica curved HR mirrors, copper
piezo flexure stages, a volumetric intracavity standing wave whose brightness tracks $|\alpha|^2$,
a toggleable UHV sapphire vacuum chamber, $N = 1\text{–}8$ vertical high-NA tweezer traps, a
downstream N-BK7 50/50 beam-splitter balanced homodyne stage, and a benchtop RF spectrum analyzer:

- **Direct 3D diegetic controls**: drag the right cavity mirror piezo (detuning $\Delta/2\pi$),
  the left mirror standing-wave stage ($\Phi$), the $\lambda/2$ polarisation waveplate ($\theta$),
  the electro-optic phase modulator ($\phi_2$), the local-oscillator piezo mirror ($\theta_\mathrm{hom}$),
  or the UHV vacuum gauge ($P_\mathrm{gas}$); click the monitor to zoom straight in.
- **Live QLT & stability**: computes full $(2N+2)\times(2N+2)$ Linear Amplifier Model homodyne
  $S_{hh}(\omega)$, heterodyne $S_\mathrm{het}(\omega)$, hybridised position PSDs $S_{x_n x_n}, S_{y_n y_n}$,
  and cross-spectra $\mathrm{Re}\,S_{x_1 y_1}(\omega)$ alongside exact complex drift-matrix
  eigenvalues $\lambda$ on a live stability inset — flagging non-reciprocal instability
  ($\phi_2 \in [54.4^\circ, 125.6^\circ]$ or blue detuning $\Delta > 0$) both on screen and in 3D.
- **Presets**: Quickstart $2\times 2$ ($\phi_2 = 30^\circ$ stable and $90^\circ$ unstable),
  PRR 5 node / cancellation point ($\Phi = 0.248\pi$) / off-node with 3D normal-mode orientation
  vectors, and PRR 8 4- and 8-particle mechanical mode combs.
