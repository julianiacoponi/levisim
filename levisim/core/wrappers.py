"""
Decorators for timing, profiling and showing a progress bar
"""

import time
import inspect
import cProfile
import pstats

from io import StringIO
from functools import wraps
from tqdm import tqdm

from levisim.core.logger import logger


__all__ = [
    'timer',
    'profiler',
    'progress_bar',
]


def _called_name(func, args):
    """
    The name a timed call is logged under: the function's qualified name, except for a method
    called on an instance, which is named after the instance's class (so that a method a class
    inherits from a base or mixin, e.g. `MultiTrap.homodyne_spectrum` from `MultiTrapBase`, is
    logged under the class it was called on)
    """
    if args and '.' in func.__qualname__ and not isinstance(args[0], type):
        cls = type(args[0])
        if any(func.__name__ in vars(klass) for klass in cls.__mro__):
            return f'{cls.__name__}.{func.__name__}'
    return func.__qualname__


def timer(enabled, debug=False):
    """
    A `perf_counter` timer decorator, which logs the run time of each call at DEBUG

    With `debug` it also logs the call's arguments at DEBUG. Both messages are formatted lazily,
    so they cost nothing when no sink takes DEBUG.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            if debug:
                logger.debug('"{}" called with args={!r}, kwargs={!r}', _called_name(func, args), args, kwargs)
            logger.debug('⏱️ "{}" executed in {:.4f} s', _called_name(func, args), end_time - start_time)
            return result
        return wrapper
    return decorator


def profiler(enabled):
    """
    A cProfile profiling decorator
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()

            # capture the profiling results
            result_stream = StringIO()
            stats = pstats.Stats(profiler, stream=result_stream)
            # sort by time
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats()

            print(result_stream.getvalue())
            return result
        return wrapper
    return decorator


def resolve_variable_path(bound_arguments, path):
    """
    Resolves a dotted `path` against a function call's bound arguments

    The first part names any argument of the call (`self` in a method, or e.g. `labels` in a
    plain function); each later part is an attribute lookup, or a key lookup if the object is a
    dict. So 'self.ω_range', 'labels' and 'mtrap.cavity.κ' all work.
    """
    parts = path.split('.')
    if parts[0] not in bound_arguments:
        raise ValueError(
            f'Variable path "{path}" could not be resolved:'
            f' "{parts[0]}" is not an argument of the call ({list(bound_arguments)})'
        )
    obj = bound_arguments[parts[0]]
    for part in parts[1:]:
        try:
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        except (AttributeError, KeyError):
            raise ValueError(f'Variable path "{path}" could not be resolved at "{part}"')
    return obj


def progress_bar_wanted_levels():
    """
    The console levels at which progress bars are drawn: INFO and anything more verbose

    SUCCESS (severity 25) is less verbose than INFO (20), so, like WARNING, it draws no bars; this
    matches the numeric rule in :func:`progress_bars_wanted` (severity <= 20).
    """
    return ('TRACE', 'DEBUG', 'INFO')


def progress_bars_wanted():
    """
    Whether progress bars should be drawn now, decided at call time

    True only if the console log level (see levisim.core.logger.set_console_level) is INFO or
    more verbose, so that the default WARNING console keeps notebooks and scripts free of bars.
    tqdm's own TQDM_DISABLE environment variable still switches them off regardless.
    """
    import sys
    module = sys.modules.get('levisim.core.logger')
    level = getattr(module, '_console_level', None)
    if level is None:
        return False
    if isinstance(level, int):
        return level <= 20
    return str(level).upper() in progress_bar_wanted_levels()


def progress_bar(loop_variable_path, enabled):
    """
    A decorator that shows a tqdm progress bar sized by an iterable the function loops over

    The iterable is found from `loop_variable_path`, a dotted path whose first part is the name
    of any argument of the decorated function (see `resolve_variable_path`): 'self.ω_range' for
    the MultiTrap spectrum methods, 'labels' for generate_determinant_and_adjugate_by_order.
    Defaults of unpassed arguments are applied before resolving.

    NOTE: a decorator cannot see inside the function's loop, so the bar is not updated per
    iteration: it opens with the total length, the function runs, and the bar is filled when it
    returns (it then reports the elapsed time and the mean time per item).

    :param loop_variable_path: the dotted path to the iterable (e.g. 'self.ω_range')
    :param enabled: Boolean to toggle the progress bar on or off
    """
    def decorator(func):
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled or not progress_bars_wanted():
                return func(*args, **kwargs)

            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            loop_variable = resolve_variable_path(bound.arguments, loop_variable_path)

            with tqdm(total=len(loop_variable), desc=f'Looping through {loop_variable_path}') as prog_bar:
                result = func(*args, **kwargs)
                prog_bar.update(len(loop_variable))

            return result

        return wrapper

    return decorator
