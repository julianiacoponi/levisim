import time
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


def timer(enabled, debug=False):
    """
    A `perf_counter` timer decorator
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
                msg = f'{args=}, {kwargs=}'
                print(msg), logger.debug(msg)
            msg = f'⏱️ Function "{func.__name__}" executed in {end_time - start_time:.4f} seconds'
            print(msg), logger.info(msg)
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


# TODO: this only works for self.ω_range at the moment...
def progress_bar(loop_variable_path, enabled):
    """
    A decorator to add a loading bar to a loop in a function
    The loop must iterate over an iterable stored in the function's local or nested attributes

    :param loop_variable_path: The dotted path to the loop variable (e.g., 'self.ω_range')
    :param enabled: Boolean to toggle the loading bar on or off
    """
    def resolve_variable_path(obj, path):
        """
        Resolve a dotted variable path to get the actual value
        """
        parts = path.split('.')
        # handle 'self' as the first part explicitly
        if parts[0] == 'self':
            # `obj` is `self` in class methods
            obj = obj
            # remove 'self' from the path
            parts = parts[1:]

        for part in parts:
            obj = getattr(obj, part)
        return obj

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            # resolve the loop variable path
            try:
                # args[0] is 'self'
                loop_variable = resolve_variable_path(args[0], loop_variable_path)
            except AttributeError:
                raise ValueError(f'Variable path "{loop_variable_path}" could not be resolved')

            # wrap the loop with a progress bar
            with tqdm(loop_variable, desc=f'Looping through {loop_variable_path}') as prog_bar:
                result = func(*args, **kwargs)
                prog_bar.update(len(loop_variable))

            return result

        return wrapper

    return decorator