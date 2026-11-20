# Power-series pickles shipped with levisim

`MultiTrap(load_in_power_series=True)` reads `{n}D_{N}x{d}_up_to_{order}.pickle` from here
(or from `power_series_dir=` / `$LEVISIM_POWER_SERIES_DIR` when given). Only the 2D, 4D and 6D
series are shipped (about 270 kB); regenerate or extend them with
`python scripts/generate_power_series_pickles.py`.
