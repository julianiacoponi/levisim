### `levisim`

There are 2 layers of `levisim` directory because:
- The git repo is called `levisim`
- The python package inside the git repo is also called `levisim`

Note on importing from `levisim` into python environments:

Importing `levisim` from python session (a file, a jupyter notebook, the python REPL itself) whose 'current working directory' (`os.cwd()`) has inside it also a directory called `levisim` results in an empty import due to "name shadowing" -- a good reason to keep notebooks in the `notebooks/` directory!

License chosen through [choosealicense.com](https://choosealicense.com)