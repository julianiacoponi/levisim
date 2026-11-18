"""
levisim's loguru logger, and where its records go

Three sinks are set up when levisim is imported:

console (stderr)
    WARNING and above by default, so notebooks and scripts show only problems. Set the level with
    the environment variable LEVISIM_LOG_LEVEL (e.g. ``INFO``, ``DEBUG``, ``TRACE``, or ``OFF``)
    before importing levisim, or at any time with :func:`set_console_level`. LEVISIM_LOG_TO_CONSOLE=1
    is the old switch for the same thing and still means LEVISIM_LOG_LEVEL=TRACE.
file
    DEBUG and above, one JSON object per line, rotated at 10 MB with the 5 newest rotated files kept,
    at ``$LEVISIM_LOG_DIR/levisim.jsonl`` (default ``~/.levisim/logs/levisim.jsonl``). Set
    LEVISIM_LOG_FILE=0 to switch it off. If the directory cannot be written, there is simply no
    file sink. :func:`current_log_file` gives the path in use.
Elasticsearch
    TRACE and above, to the ``loguru-logs`` index at LEVISIM_ELASTICSEARCH_URL (default
    http://localhost:9200), only when the optional ``elasticsearch`` package is installed. The first
    failed delivery disables it with one notice on stderr. ``levisim-logs`` (levisim.core.elk) runs
    Elasticsearch and Kibana locally and loads the log file into them, so nothing is lost while
    they are down.

The level meanings levisim's own messages follow are in docs/guide/logging.md.
"""
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

from loguru import logger

# NOTE: Elasticsearch is an optional convenience for shipping logs to a local ELK stack.
# `levisim` is a physics library and must stay importable without it, so this import is
# guarded and every use of the client below is conditional on it having succeeded.
try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None


__all__ = [
    'logger',
    'elastic_search_client',
    'elastic_search_index',
    'set_console_level',
    'current_log_file',
]


DEFAULT_CONSOLE_LEVEL = 'WARNING'
FILE_LEVEL = 'DEBUG'
ELASTICSEARCH_LEVEL = 'TRACE'
LOG_FILE_NAME = 'levisim.jsonl'
LOG_FILE_ROTATION = '10 MB'
LOG_FILE_RETENTION = 5  # rotated files kept, besides the live one

# override with LEVISIM_ELASTICSEARCH_URL to point at a non-default ELK stack
ELASTICSEARCH_URL = os.environ.get('LEVISIM_ELASTICSEARCH_URL', 'http://localhost:9200')

elastic_search_index = 'loguru-logs'
elastic_search_client = Elasticsearch(ELASTICSEARCH_URL) if Elasticsearch is not None else None

# tracks whether the Elasticsearch sink is still worth attempting - see send_to_elasticsearch
_elasticsearch_reachable = elastic_search_client is not None

# loguru handler ids of the sinks this module added, so they can be replaced without touching
# any sink a user added themselves
_handler_ids = {'console': None, 'file': None, 'elasticsearch': None}
_console_level = None
_log_file = None

_FALSE_STRINGS = {'', '0', 'false', 'no', 'off'}
_OFF_LEVELS = {'OFF', 'NONE'}


# ------------------------------------------------------------------------------------------------
# one flat JSON object per record, shared by the file and Elasticsearch sinks
# ------------------------------------------------------------------------------------------------
def log_record_to_dict(record):
    """
    Flattens a loguru record into the JSON-serialisable document levisim writes and indexes

    Keys: timestamp (ISO 8601 with UTC offset), level, level_no, message, name (the logging
    module's dotted path), module, function, line, file, process, thread, plus ``extra`` (anything
    bound with ``logger.bind``) and ``exception`` (the formatted traceback) when present.
    """
    document = {
        'timestamp': record['time'].isoformat(),
        'level': record['level'].name,
        'level_no': record['level'].no,
        'message': record['message'],
        'name': record['name'],
        'module': record['module'],
        'function': record['function'],
        'line': record['line'],
        'file': record['file'].name,
        'process': record['process'].id,
        'thread': record['thread'].name,
    }
    extra = {k: v for k, v in record['extra'].items() if not k.startswith('_levisim')}
    if extra:
        document['extra'] = extra
    if record['exception'] is not None:
        exc_type, exc_value, exc_traceback = record['exception']
        document['exception'] = ''.join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
    return document


def log_record_to_json(record):
    """The record as one line of JSON (no newline), deterministic for a given record"""
    return json.dumps(
        log_record_to_dict(record), ensure_ascii=False, sort_keys=True, default=str,
        separators=(',', ':'),
    )


def log_line_id(line):
    """
    The Elasticsearch document id of one JSON log line: its SHA-1

    The live Elasticsearch sink and ``levisim-logs ingest`` both use it, so a record that reached
    Elasticsearch live and is later ingested again from the file is stored once.
    """
    return hashlib.sha1(line.rstrip('\r\n').encode('utf-8')).hexdigest()


def _json_line_format(record):
    # loguru formats the string this returns with the record, so the JSON (full of braces) goes
    # in through `extra` rather than into the template itself
    record['extra']['_levisim_json'] = log_record_to_json(record)
    return '{extra[_levisim_json]}\n'


# ------------------------------------------------------------------------------------------------
# Elasticsearch sink
# ------------------------------------------------------------------------------------------------
def send_to_elasticsearch(message):
    """
    Sends one log record to the Elasticsearch client, as the same document the log file holds

    Silently stops attempting delivery after the first connection failure, so that an
    unreachable ELK stack produces one notice rather than a traceback per log call.
    """
    global _elasticsearch_reachable
    if not _elasticsearch_reachable:
        return

    line = log_record_to_json(message.record)
    try:
        elastic_search_client.index(
            index=elastic_search_index, id=log_line_id(line), document=json.loads(line),
        )
    except Exception as exception:
        _elasticsearch_reachable = False
        kept = (
            f'\n  Logs are still written to {_log_file};' if _log_file is not None else '\n '
        )
        print(
            f'levisim: Elasticsearch at {ELASTICSEARCH_URL} is unreachable'
            f' ({type(exception).__name__}), so the Elasticsearch log sink is now disabled.'
            f'{kept} start Elasticsearch with `levisim-logs up` (or `docker compose up -d`)'
            f' and `levisim-logs ingest` loads the log file into it.',
            file=sys.stderr,
        )


# ------------------------------------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------------------------------------
def _normalise_level(level):
    """An upper-case loguru level name or int, or None for 'off'; ValueError if unknown"""
    if level is None:
        return None
    if isinstance(level, str):
        level = level.strip()
        if level.lstrip('-').isdigit():
            level = int(level)
        else:
            level = level.upper()
            if level in _OFF_LEVELS:
                return None
            logger.level(level)  # raises ValueError for a name loguru does not know
    if isinstance(level, bool) or not isinstance(level, (int, str)):
        raise TypeError(f'a log level must be a level name or an int, not {level!r}')
    if isinstance(level, int) and level < 0:
        raise ValueError(f'a log level must be a non-negative int, not {level}')
    return level


def _console_level_from(environ):
    level = environ.get('LEVISIM_LOG_LEVEL', '').strip()
    if level:
        return level
    if environ.get('LEVISIM_LOG_TO_CONSOLE', '').strip().lower() not in _FALSE_STRINGS:
        return 'TRACE'
    return DEFAULT_CONSOLE_LEVEL


def default_log_dir(environ=None):
    """$LEVISIM_LOG_DIR, or ~/.levisim/logs"""
    environ = os.environ if environ is None else environ
    if environ.get('LEVISIM_LOG_DIR'):
        return Path(environ['LEVISIM_LOG_DIR']).expanduser()
    return Path.home() / '.levisim' / 'logs'


def _remove_handler(name):
    handler_id = _handler_ids[name]
    _handler_ids[name] = None
    if handler_id is not None:
        try:
            logger.remove(handler_id)
        except ValueError:  # already removed, e.g. by a user's logger.remove()
            pass


def set_console_level(level):
    """
    Sets the level of levisim's console (stderr) log sink, at any time

    Args:
        level: a loguru level name (case-insensitive: 'TRACE', 'DEBUG', 'INFO', 'SUCCESS',
            'WARNING', 'ERROR', 'CRITICAL'), an int severity, or None / 'OFF' for no console output

    Returns:
        the level now in use (None when off)

    Raises:
        ValueError: for a level loguru does not know (the console sink is left unchanged)

    The default is WARNING, or LEVISIM_LOG_LEVEL when that is set. The file and Elasticsearch sinks
    are unaffected.
    """
    global _console_level
    level = _normalise_level(level)
    _remove_handler('console')
    if level is not None:
        # sys.stderr is looked up now rather than at import, so a notebook or test that has
        # replaced it gets the messages
        _handler_ids['console'] = logger.add(sys.stderr, level=level)
    _console_level = level
    return level


def current_log_file():
    """The path of the JSON-lines log file being written, or None if there is no file sink"""
    return _log_file


def _add_file_sink(path, level=FILE_LEVEL, rotation=LOG_FILE_ROTATION, retention=LOG_FILE_RETENTION):
    """Adds the JSON-lines file sink at `path`; returns its handler id, or None if not writable"""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8'):
            pass
        return logger.add(
            str(path), level=level, format=_json_line_format, rotation=rotation,
            retention=retention, encoding='utf-8',
        )
    except Exception:  # unwritable directory, read-only home, odd path: no file sink, no noise
        return None


def _configure(environ=None):
    """
    (Re)builds levisim's three sinks from the environment variables described in the module docstring

    Called once at import. Sinks a user added to the shared loguru logger are left alone.
    """
    global _console_level, _log_file
    environ = os.environ if environ is None else environ

    # loguru's own default stderr sink (id 0) would print everything at DEBUG
    try:
        logger.remove(0)
    except ValueError:
        pass

    # console
    level = _console_level_from(environ)
    try:
        set_console_level(level)
    except (ValueError, TypeError):
        print(
            f'levisim: LEVISIM_LOG_LEVEL={level!r} is not a log level;'
            f' using {DEFAULT_CONSOLE_LEVEL} for the console.',
            file=sys.stderr,
        )
        set_console_level(DEFAULT_CONSOLE_LEVEL)

    # file
    _remove_handler('file')
    _log_file = None
    if environ.get('LEVISIM_LOG_FILE', '1').strip().lower() not in _FALSE_STRINGS:
        try:
            path = default_log_dir(environ) / LOG_FILE_NAME
        except Exception:  # no home directory to be found
            path = None
        if path is not None:
            _handler_ids['file'] = _add_file_sink(path)
            if _handler_ids['file'] is not None:
                _log_file = path

    # Elasticsearch
    _remove_handler('elasticsearch')
    if elastic_search_client is not None:
        _handler_ids['elasticsearch'] = logger.add(send_to_elasticsearch, level=ELASTICSEARCH_LEVEL)


# NOTE: nothing is logged here at import. A record at import would reach the Elasticsearch sink,
# whose first failed delivery prints a notice, on every `import levisim` without a running stack.
_configure()
