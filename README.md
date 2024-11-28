### `levisim`

There are 2 layers of `levisim` directory because:
    - The git repo is called `levisim`
    - The python package inside the git repo is called `levisim`

The terminal command to compile and run the fortran is
```bash
gfortran -O2 -llapack NParticle.f90  3DEQRom.f90 ParametersNpair.f90 Suscept.f90 PSD.f90 NoiseVec2D.f90; ./a.out
```

Note on importing from `levisim` into python environments:

Importing `levisim` from python session (a file, a jupyter notebook, the python REPL itself) whose 'current working directory' (`os.cwd()`) has inside it also a directory called `levisim` results in an empty import due to name 'shadowing'. One reason why its handy to keep notebooks in the `notebooks/` directory!

License chosen through [choosealicense.com](https://choosealicense.com)