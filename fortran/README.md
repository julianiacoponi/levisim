# `levisim` Fortran reference implementation

Calculates quantum noise spectra for the multiparticle cavity trap by full linear-amplifier
numerics, and compares them against the collective-mode formula. The `.dat` files this writes
are the reference data that `levisim`'s Python implementation is checked against
(see `tests/test_fortran_reference.py`).

## Building

```sh
make pairs      # frequencies in pairs: 312 kHz, 15 kHz inter-pair gap
make equal      # uniform ladder from 320 kHz in 3 kHz steps, two modes parked at 5 MHz
make random     # ladder from 350 kHz in 7.5 kHz steps with seeded pseudo-random jitter
make lint       # compile-only pass with warnings enabled
```

LAPACK is taken from the system. Under conda, point the linker at the environment:

```sh
make pairs LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
```

Each variant builds into `build/<variant>/` and writes its output `.dat` files there, so the
variants neither overwrite each other nor clobber the reference data at the top level. **Run
the executable from inside its build directory**, since the output unit numbers are opened as
bare relative filenames.

The variants differ *only* in which parameter file fills the mechanical frequency layout; the
main program, susceptibilities and spectrum code are shared.

| Variant | Parameter file | Layout |
|---|---|---|
| `pairs` | `ParametersNpair.f90` | pairs at 312 kHz, 15 kHz gap between pairs, +1.5 kHz on the second axis |
| `equal` | `ParametersEqual.f90` | uniform 3 kHz ladder from 320 kHz, modes (3,x) and (6,y) parked at 5 MHz |
| `random` | `ParametersRandom.f90` | 7.5 kHz ladder from 350 kHz plus seeded jitter |

System size is set in `Nparticle.h` (`num_traps`, `Cartesian_axes_per_trap`); physical constants
and the cavity/tweezer configuration are in `CSCAVITY.h`.

## Output files

| File | Contents |
|---|---|
| `FTanXYZ.dat` | ω, then the position PSD for X1, Y1, X2, Y2, X3, Y3 |
| `FTCBM.dat` | ω, heterodyne and PSD via the collective-mode formula |
| `FTanOPT.dat` | ω, heterodyne and symmetrised homodyne by linear amplifier |
| `OMEGAS.dat`, `Omegas1.dat`, `Omegas2.dat` | the mechanical frequency layout actually used |

In the source: `PSD` is S_QQ by linear amplifier, `PSDCBM` is S_QQ by formula, `HetCBM` is
heterodyne by formula, and `SHet1` is heterodyne by linear amplifier.

## Verification status

The `pairs` build reproduces the committed `.dat` reference files **bit-for-bit on every
shared row** (maximum absolute difference exactly 0), and `OMEGAS.dat` byte-identically.

Two things are worth knowing about that reference data:

1. **The committed `.dat` files are truncated.** A complete run writes 800 rows;
   `FTanXYZ.dat` holds 799 and `FTCBM.dat`/`FTanOPT.dat` hold 769. The values present are
   exact — the tail is simply missing, presumably from a run that was interrupted or copied
   while still writing.
2. **They were produced before the `cmplx` precision fix** described below, so regenerating
   them now gives values that differ by a median relative ~2e-7. The reference files are kept
   as-is to preserve provenance.

`Omegas2.dat` in the reference set is non-empty, but the `pairs` variant never writes to that
unit — only `equal` and `random` do. That file therefore came from a different variant's run.

## Fixes applied during review

- **`cmplx()` was defaulting to single precision.** Nine calls in `3DEQRom.f90` and
  `Suscept.f90` built complex susceptibilities as `cmplx(double, double)` with no `kind`
  argument, which returns `COMPLEX(4)`. The results were assigned to `double complex`
  variables, so the values were truncated to ~7 significant digits and widened again. All nine
  now pass `kind=8`. Measured effect on the output: median relative change 2e-7, maximum 1e-6 —
  the expected signature of single-precision truncation.
- **Three of the four documented build lines did not compile.** `ParametersNP.f90`,
  `ParametersN8.f90`, `ParametersNpairGap.f90`, `ParametersBlock.f90` and `Parameters3P.f90`
  still used the pre-rename interface (`SUBROUTINE Parameters` with `Npcle`/`Nmode`/`NT`),
  which no longer matches `Nparticle.h`. `ParametersNP` and `ParametersN8` have been ported to
  the current `FILL_PARAMETER_ARRAYS` interface as `ParametersEqual.f90` and
  `ParametersRandom.f90`; the originals are kept under `legacy/`.
- **The random variant was not reproducible.** `ParametersN8.f90` called `RANDOM_NUMBER`
  without ever seeding the generator, so each run produced a different frequency layout.
  `ParametersRandom.f90` seeds from a fixed constant, overridable with `LEVISIM_RANDOM_SEED`.
- **CRLF line endings** in `Nparticle.h` and `CSCAVITY.h` converted to LF.
- **Compiler warnings** reduced from 112 to 18 under `-Wall -Wextra -Wconversion`, with zero
  precision-loss warnings remaining. The 18 are benign: unused z-axis arrays (there is no z
  motion while `Cartesian_axes_per_trap = 2`) and three unused format labels. They are left in
  place deliberately, as scaffolding for the planned z-axis extension.

## Known issues, not fixed

- `NParticle.f90` prints `kappa_nano (Hz) 1.216e-86`. A value that small is almost certainly
  not meaningful and the formula behind it is worth checking.
- `3DEQRom.f90` carries two comments reading `NOT SURE: CHECK!!!` beside the mechanical
  susceptibilities `chi_M_X` and `chi_M_Y`, which are evaluated at the *other* axis's
  resonance. That looks deliberate but is flagged in the source as unverified.
- No `intent(in)`/`intent(out)` anywhere — noted as a to-do in the original header comment of
  `NParticle.f90`.

## `legacy/`

Earlier sources kept for reference, not built by the Makefile:

- `2Particle.f90`, `3Particle.f90` — older main programs using the `ANALYT`/`NORM` interface.
- `PSDhet.f90` — heterodyne spectrum code for that older interface (it defines `ANALYT`,
  `Homodyne`, `Heterodyne`, not the `CALCULATE_*` subroutines the current main calls, so it
  cannot substitute for `PSD.f90`).
- `ParametersNP.f90`, `ParametersN8.f90`, `ParametersNpairGap.f90`, `ParametersBlock.f90`,
  `Parameters3P.f90` — pre-rename parameter files.
- `symmetric_group.f90`, `symmetric_group_iterative.f90` — standalone permutation utilities.
- `HetAdagg/` — a distinct earlier snapshot of the whole tree. Its sources differ from the
  top-level ones throughout and it contains `ParametersNpairOld.f90` and a `CSCAVITY copy.h`,
  so it is treated as an archive rather than a maintained variant.
