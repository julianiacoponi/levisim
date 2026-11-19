"""
levisim.core.wrappers.progress_bar resolves its dotted path against any argument of the call
"""
from types import SimpleNamespace

import pytest

from levisim.core.logger import set_console_level
from levisim.core.wrappers import progress_bar, progress_bars_wanted, resolve_variable_path


@pytest.fixture(autouse=True)
def console_at_info():
    """Progress bars draw only when the console is at INFO or more verbose"""
    import levisim.core.logger  # noqa: F401
    import sys
    previous = sys.modules['levisim.core.logger']._console_level
    set_console_level('INFO')
    yield
    set_console_level(previous)


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


def test_progress_bars_follow_the_console_level(capsys):
    """At the default WARNING console no bar is drawn; at INFO one is"""
    set_console_level('WARNING')
    assert not progress_bars_wanted()
    assert Looper().method() == 21
    assert 'Looping through' not in capsys.readouterr().err
    set_console_level('INFO')
    assert progress_bars_wanted()
    assert Looper().method() == 21
    assert 'Looping through' in capsys.readouterr().err


@pytest.mark.parametrize('level, wanted', [
    ('TRACE', True), ('DEBUG', True), ('INFO', True), ('SUCCESS', False), ('WARNING', False), ('ERROR', False),
    (5, True), (20, True), (25, False), (30, False),
])
def test_progress_bars_are_drawn_at_info_or_more_verbose_only(level, wanted):
    """By name or by severity, the same rule: INFO (20) or more verbose; SUCCESS (25) is not"""
    import sys
    logger_module = sys.modules['levisim.core.logger']  # `levisim.core.logger` the attribute is loguru's logger
    previous = logger_module._console_level
    try:
        logger_module._console_level = level
        assert progress_bars_wanted() is wanted
    finally:
        logger_module._console_level = previous
