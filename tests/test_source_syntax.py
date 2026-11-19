"""
Every source file compiles with warnings as errors

Invalid escape sequences such as '\\D' in a normal string are a SyntaxWarning since Python 3.12 and
will be a SyntaxError in a future Python, so LaTeX in strings and docstrings must use raw strings.
"""

import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = sorted(
    path for folder in ('levisim', 'tests', 'docs') for path in (REPO_ROOT / folder).rglob('*.py')
    if '_build' not in path.parts
)


def test_source_files_found():
    assert len(SOURCE_FILES) > 40


@pytest.mark.parametrize('path', SOURCE_FILES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_compiles_without_warnings(path):
    source = path.read_text(encoding='utf-8')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compile(source, str(path), 'exec')
