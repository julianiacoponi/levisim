"""
levisim.core.wrappers.progress_bar resolves its dotted path against any argument of the call
"""
from types import SimpleNamespace

import pytest

from levisim.core.wrappers import progress_bar, resolve_variable_path


class Looper:
    def __init__(self):
        self.ω_range = list(range(7))
        self.nested = SimpleNamespace(points={'grid': [1, 2, 3]})

    @progress_bar('self.ω_range', enabled=True)
    def method(self):
        return sum(self.ω_range)

    @progress_bar('self.nested.points.grid', enabled=True)
    def nested_method(self):
        return len(self.nested.points['grid'])


@progress_bar('labels', enabled=True)
def plain_function(n, labels, scale=1):
    return n * scale * len(labels)


@progress_bar('items', enabled=True)
def function_with_default(items=('a', 'b')):
    return ''.join(items)


def test_resolve_variable_path():
    looper = Looper()
    assert resolve_variable_path({'self': looper}, 'self.ω_range') == list(range(7))
    assert resolve_variable_path({'self': looper}, 'self.nested.points.grid') == [1, 2, 3]
    assert resolve_variable_path({'labels': ['x1']}, 'labels') == ['x1']
    with pytest.raises(ValueError):
        resolve_variable_path({'self': looper}, 'labels')
    with pytest.raises(ValueError):
        resolve_variable_path({'self': looper}, 'self.missing')


def test_progress_bar_on_methods_plain_functions_and_defaults(capsys):
    looper = Looper()
    assert looper.method() == 21
    assert looper.nested_method() == 3
    assert plain_function(2, ['x1', 'y1', 'x2']) == 6
    assert plain_function(2, labels=['x1'], scale=3) == 6
    assert function_with_default() == 'ab'
    assert 'Looping through labels' in capsys.readouterr().err


def test_unresolvable_path_raises():
    @progress_bar('nonexistent', enabled=True)
    def f(x):
        return x
    with pytest.raises(ValueError):
        f(1)

    @progress_bar('nonexistent', enabled=False)
    def g(x):
        return x
    assert g(1) == 1
