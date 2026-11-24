# Interactive 3D web labs

`levisim` includes photorealistic Three.js optical-bench laboratories that compute live motional
and optical spectra directly on an in-game RF spectrum analyzer monitor.

## Launching locally (with live Python `levisim` backend)

```sh
levisim-web                      # opens http://127.0.0.1:8765/free_space_tweezer/
python -m levisim.web --no-open  # serve without launching a browser tab
```

When served through `levisim-web`, the browser computes interactive 60 FPS previews using an exact
JavaScript port of `levisim`'s equations (cross-checked against Python in `tests/test_web_free_space.py`)
and synchronises with the live Python `levisim` server (`/api/tweezer/spectra` and `/api/tweezer/simulate`)
for exact `Trap` calculations and stochastic `Simulation` runs.

## GitHub Pages (zero install)

Every build of this documentation copies `levisim/web/static/` to `labs/`:

```{raw} html
<ul>
  <li><a href="labs/index.html">Interactive Labs Portal</a></li>
  <li><a href="labs/free_space_tweezer/index.html">Free-Space Optical Tweezer Lab</a> (<code>Trap</code>, <code>SpectralDensityTheory</code>, PRR 6)</li>
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
