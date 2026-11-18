# Testing

## Running the tests

From the repo root:

```sh
pip install pytest pytest-xdist
python -m pytest -n auto          # in parallel
python -m pytest -m "not slow"    # skip the tests that take more than ~10 s
```

`pytest.ini` sets `pythonpath = .` and `testpaths = tests`, so the in-tree package is found without installing it and without
setting `PYTHONPATH`. One test (`tests/test_import.py::test_no_multitrap_fixtures_built_at_import`)
imports `levisim` in a fresh subprocess, which relies on Python putting the working directory on
`sys.path`; it fails if `PYTHONSAFEPATH` is set in the environment, unless `levisim` is installed.

The suite runs on every push and pull request through GitHub Actions
(`.github/workflows/tests.yml`).

## What the suites cover

| File | Covers |
|---|---|
| `test_import.py` | `import levisim` on a clean clone: no fixtures built at import, lazy fixtures resolve, optional dependencies (OpenCV, ...) are guarded |
| `test_public_api.py` | the package's public export surface, pinned |
| `test_constants.py`, `test_units.py` | the physical constants re-exported from SciPy, and unit-conversion sanity |
| `test_fixed_call_sites.py` | regressions for three call sites that used to raise `NameError` |
| `test_drift_matrix.py` | assembly of the drift matrix $\mathbf A$ for the Linear Amplifier Model |
| `test_method_equivalence.py` | spectra from the seven calculation methods agree with each other |
| `test_noise_vector_equivalence.py` | noise vectors agree across methods (marked `slow`) |
| `test_fortran_reference.py` | spectra against the Fortran reference data |
| `test_gaussian_states.py` | two-mode Gaussian-state measures (PPT, log-negativity, discord, witnesses) |
| `test_two_tone.py`, `test_two_tone_traps.py`, `test_focal_expansion.py` | covariance dynamics, the multi-tone Floquet drive, the two-tone scheme, and traps built from the focal field |

Two markers are registered in `tests/conftest.py`: `slow` (more than about 10 s) and
`needs_power_series` (needs a power-series pickle; the three small ones the suite uses are
tracked in the repo).

## The Fortran reference

`fortran/` holds an independent Fortran implementation of the multiparticle spectra, and the
`.dat` files it wrote. `tests/test_fortran_reference.py` rebuilds the same 8 × 2 system in
`levisim` (the `FORTRAN_8x2` fixture: eight particles with near-degenerate $x$ and $y$
frequencies in pairs) and compares the heterodyne spectrum, the homodyne spectrum and all position
PSDs against `FTanXYZ.dat`, `FTCBM.dat` and `FTanOPT.dat`. Agreement is limited by the files
themselves: they are written with seven significant figures, and the measured maximum relative
difference is $7.4 \times 10^{-7}$. The test is skipped when the data files are absent.

To regenerate the data, see `fortran/README.md`: `make pairs` (or `equal`, `random`) builds into
`fortran/build/<variant>/`, which needs `gfortran` and LAPACK.
