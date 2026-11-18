"""
Fails if the Sphinx build warned about anything other than docstrings in the package

Usage (from the repo root):
    sphinx-build -b html docs docs/_build/html -w docs/_build/warnings.log
    python docs/check_sphinx_warnings.py docs/_build/warnings.log

The hand-written pages under docs/ are held to "no warnings" (equivalent to sphinx-build -W).
Warnings raised while rendering the docstrings of levisim itself - mostly free-form physics notes
that are not valid reStructuredText - are counted and listed by module, but do not fail the build:
they are fixed in the code, not here.
"""
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / 'levisim'
RECORD_START = re.compile(r'^(?P<location>.*?):?\s*(?P<level>WARNING|ERROR|CRITICAL): ')


def is_from_docstring(location):
    """True if a warning location points into the package's docstrings or the generated API pages"""
    if 'docstring of ' in location or '_autosummary' in location:
        return True
    path = location.split(':')[0]
    try:
        return Path(path).resolve().is_relative_to(PACKAGE_DIR)
    except (OSError, ValueError):
        return False


def main(log_path):
    docstring_warnings, page_warnings = Counter(), []
    for line in Path(log_path).read_text(encoding='utf-8').splitlines():
        match = RECORD_START.match(line)
        if not match:
            continue  # continuation line of a multi-line warning
        location = match.group('location')
        if is_from_docstring(location):
            module = re.search(r'docstring of ([\w.]+)', location)
            docstring_warnings[module.group(1) if module else location] += 1
        else:
            page_warnings.append(line)

    print(f'{sum(docstring_warnings.values())} warnings from docstrings in levisim (not failing the build):')
    for name, count in docstring_warnings.most_common():
        print(f'  {count:3d}  {name}')
    if page_warnings:
        print(f'\n{len(page_warnings)} warnings from the documentation pages:')
        print('\n'.join(f'  {line}' for line in page_warnings))
        return 1
    print('\nNo warnings from the documentation pages.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'docs/_build/warnings.log'))
