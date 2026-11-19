"""
Tests of levisim's log sinks (levisim/core/logger.py): console level, JSON-lines file, Elasticsearch

Most tests rebuild the sinks in-process from a given environment with logger._configure, and the
`sinks` fixture restores the real configuration afterwards. One test imports levisim in a fresh
interpreter, because the defaults are a claim about import time.
"""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import importlib

import pytest

from levisim.core.logger import current_log_file, log_line_id, logger, set_console_level

# NOTE: not `from levisim.core import logger`, nor `import levisim.core.logger as ...`: both give the
# loguru Logger, because levisim/core/__init__.py's wildcard import binds the name `logger` over
# the submodule attribute
log_module = importlib.import_module('levisim.core.logger')


REQUIRED_KEYS = {'timestamp', 'level', 'message', 'module', 'function', 'line'}


@pytest.fixture
def sinks():
    """Applies an environment to levisim's sinks; restores the configuration from os.environ after"""
    yield log_module._configure
    log_module._configure()


def read_json_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------------
# console
# ------------------------------------------------------------------------------------------------
def test_console_shows_warnings_but_not_info_by_default(sinks, tmp_path, capfd):
    sinks({'LEVISIM_LOG_DIR': str(tmp_path)})
    logger.info('quiet-info')
    logger.warning('loud-warning')
    err = capfd.readouterr().err
    assert 'loud-warning' in err
    assert 'quiet-info' not in err


@pytest.mark.parametrize('environment, shown, hidden', [
    ({'LEVISIM_LOG_LEVEL': 'info'}, 'info', 'debug'),
    ({'LEVISIM_LOG_LEVEL': 'DEBUG'}, 'debug', 'trace'),
    ({'LEVISIM_LOG_TO_CONSOLE': '1'}, 'trace', None),  # the old switch means TRACE
    ({'LEVISIM_LOG_TO_CONSOLE': '0'}, 'warning', 'info'),
    ({'LEVISIM_LOG_TO_CONSOLE': '1', 'LEVISIM_LOG_LEVEL': 'SUCCESS'}, 'success', 'info'),
])
def test_console_level_from_environment(sinks, tmp_path, capfd, environment, shown, hidden):
    sinks({'LEVISIM_LOG_DIR': str(tmp_path), **environment})
    getattr(logger, shown)(f'shown-{shown}')
    if hidden:
        getattr(logger, hidden)(f'hidden-{hidden}')
    err = capfd.readouterr().err
    assert f'shown-{shown}' in err
    if hidden:
        assert f'hidden-{hidden}' not in err


def test_console_off_from_environment(sinks, tmp_path, capfd):
    sinks({'LEVISIM_LOG_DIR': str(tmp_path), 'LEVISIM_LOG_LEVEL': 'OFF'})
    logger.critical('nothing-on-the-console')
    assert 'nothing-on-the-console' not in capfd.readouterr().err


def test_unknown_level_in_environment_falls_back_to_warning(sinks, tmp_path, capfd):
    sinks({'LEVISIM_LOG_DIR': str(tmp_path), 'LEVISIM_LOG_LEVEL': 'LOUD'})
    logger.info('quiet-info')
    logger.warning('loud-warning')
    err = capfd.readouterr().err
    assert "LEVISIM_LOG_LEVEL='LOUD' is not a log level" in err
    assert 'loud-warning' in err and 'quiet-info' not in err


def test_set_console_level_at_runtime(sinks, tmp_path, capfd):
    sinks({'LEVISIM_LOG_DIR': str(tmp_path)})
    assert set_console_level('debug') == 'DEBUG'
    logger.debug('now-debug-shows')
    assert set_console_level(None) is None
    logger.error('console-is-off')
    assert set_console_level('OFF') is None
    assert set_console_level(20) == 20  # loguru's INFO severity
    logger.info('int-level-works')
    err = capfd.readouterr().err
    assert 'now-debug-shows' in err
    assert 'console-is-off' not in err
    assert 'int-level-works' in err


def test_set_console_level_rejects_an_unknown_level_and_keeps_the_sink(sinks, tmp_path, capfd):
    sinks({'LEVISIM_LOG_DIR': str(tmp_path)})
    with pytest.raises(ValueError):
        set_console_level('LOUDER')
    logger.warning('still-there')
    assert 'still-there' in capfd.readouterr().err


def test_set_console_level_is_exported():
    assert 'set_console_level' in log_module.__all__
    assert 'current_log_file' in log_module.__all__
    import levisim.core
    assert levisim.core.set_console_level is set_console_level


# ------------------------------------------------------------------------------------------------
# file
# ------------------------------------------------------------------------------------------------
def test_file_sink_writes_flat_json_lines_from_debug(sinks, tmp_path):
    sinks({'LEVISIM_LOG_DIR': str(tmp_path)})
    path = tmp_path / 'levisim.jsonl'
    assert current_log_file() == path

    logger.trace('trace-not-in-file')
    logger.debug('debug-in-file')
    logger.bind(run=7).info('info-with-extra')
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception('error-with-traceback')

    records = read_json_lines(path)
    messages = [r['message'] for r in records]
    assert messages == ['debug-in-file', 'info-with-extra', 'error-with-traceback']
    for record in records:
        assert REQUIRED_KEYS <= set(record)
        assert record['module'] == 'test_logger'
        assert record['function'] == 'test_file_sink_writes_flat_json_lines_from_debug'
        assert isinstance(record['line'], int)
        assert '_levisim_json' not in record.get('extra', {})
    assert [r['level'] for r in records] == ['DEBUG', 'INFO', 'ERROR']
    assert records[1]['extra'] == {'run': 7}
    assert 'ZeroDivisionError' in records[2]['exception']
    assert records[0]['timestamp'][:4].isdigit() and 'T' in records[0]['timestamp']


@pytest.mark.parametrize('value', ['0', 'false', 'off', 'no'])
def test_file_sink_can_be_switched_off(sinks, tmp_path, value):
    sinks({'LEVISIM_LOG_DIR': str(tmp_path), 'LEVISIM_LOG_FILE': value})
    logger.warning('no-file')
    assert current_log_file() is None
    assert not (tmp_path / 'levisim.jsonl').exists()


def test_unwritable_log_directory_means_no_file_sink_and_no_noise(sinks, tmp_path, capfd):
    # a directory that cannot exist: its parent is a regular file
    blocker = tmp_path / 'not-a-directory'
    blocker.write_text('')
    sinks({'LEVISIM_LOG_DIR': str(blocker / 'logs')})
    logger.warning('still-on-the-console')
    assert current_log_file() is None
    err = capfd.readouterr().err
    assert 'still-on-the-console' in err
    assert 'Logging error' not in err and 'Traceback' not in err


@pytest.mark.skipif(os.name != 'posix' or (hasattr(os, 'geteuid') and os.geteuid() == 0),
                    reason='needs POSIX permissions, and root ignores them')
def test_read_only_log_directory_means_no_file_sink(sinks, tmp_path, capfd):
    read_only = tmp_path / 'read-only'
    read_only.mkdir()
    read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        sinks({'LEVISIM_LOG_DIR': str(read_only)})
        logger.warning('fine')
        assert current_log_file() is None
        assert 'Logging error' not in capfd.readouterr().err
    finally:
        read_only.chmod(stat.S_IRWXU)


def test_file_sink_rotates_and_keeps_only_the_newest_files(tmp_path):
    path = tmp_path / 'levisim.jsonl'
    handler_id = log_module._add_file_sink(path, rotation='2 KB', retention=2)
    assert handler_id is not None
    try:
        for i in range(200):
            logger.bind(i=i).debug('rotation filler message number {}', i)
    finally:
        logger.remove(handler_id)
    files = sorted(tmp_path.glob('levisim*.jsonl'))
    assert path in files
    assert 2 <= len(files) <= 3  # the live file plus at most `retention` rotated ones
    for file in files:
        for record in read_json_lines(file):
            assert record['message'].startswith('rotation filler message number')


def test_default_log_directory(monkeypatch, tmp_path):
    assert log_module.default_log_dir({'LEVISIM_LOG_DIR': str(tmp_path)}) == tmp_path
    monkeypatch.setenv('HOME', str(tmp_path))
    assert log_module.default_log_dir({}) == tmp_path / '.levisim' / 'logs'


# ------------------------------------------------------------------------------------------------
# Elasticsearch
# ------------------------------------------------------------------------------------------------
class FakeElasticsearch:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def index(self, **kwargs):
        if self.fail:
            raise ConnectionError('nothing listens')
        self.calls.append(kwargs)


def test_elasticsearch_sink_indexes_the_file_line_under_its_line_id(sinks, tmp_path, monkeypatch):
    sinks({'LEVISIM_LOG_DIR': str(tmp_path)})
    fake = FakeElasticsearch()
    monkeypatch.setattr(log_module, 'elastic_search_client', fake)
    monkeypatch.setattr(log_module, '_elasticsearch_reachable', True)
    handler_id = logger.add(log_module.send_to_elasticsearch, level='TRACE')
    try:
        logger.bind(run=3).info('same-record-twice')
    finally:
        logger.remove(handler_id)

    (line,) = (tmp_path / 'levisim.jsonl').read_text(encoding='utf-8').splitlines()
    (call,) = fake.calls
    assert call['index'] == 'loguru-logs'
    # the same id `levisim-logs ingest` gives the line, so ingesting the file later adds no duplicate
    assert call['id'] == log_line_id(line)
    assert call['document'] == json.loads(line)


def test_elasticsearch_sink_disables_itself_after_one_notice(monkeypatch, capfd):
    fake = FakeElasticsearch(fail=True)
    monkeypatch.setattr(log_module, 'elastic_search_client', fake)
    monkeypatch.setattr(log_module, '_elasticsearch_reachable', True)
    handler_id = logger.add(log_module.send_to_elasticsearch, level='TRACE')
    try:
        for i in range(3):
            logger.trace('to an absent Elasticsearch {}', i)
    finally:
        logger.remove(handler_id)
    err = capfd.readouterr().err
    assert err.count('is unreachable') == 1
    assert 'levisim-logs up' in err
    assert log_module._elasticsearch_reachable is False


def test_log_line_id_ignores_the_line_ending():
    assert log_line_id('{"a":1}\n') == log_line_id('{"a":1}') == log_line_id('{"a":1}\r\n')
    assert len(log_line_id('{}')) == 40


# ------------------------------------------------------------------------------------------------
# import-time defaults, in a fresh interpreter
# ------------------------------------------------------------------------------------------------
def test_fresh_import_defaults(tmp_path):
    repo_root = str(Path(__file__).resolve().parents[1])
    env = {k: v for k, v in os.environ.items() if not k.startswith('LEVISIM_')}
    env['PYTHONPATH'] = os.pathsep.join(p for p in (repo_root, env.get('PYTHONPATH')) if p)
    env['LEVISIM_LOG_DIR'] = str(tmp_path)
    script = (
        'import levisim\n'
        'from levisim.core.logger import logger, current_log_file\n'
        'logger.debug("DEBUG-MARK")\n'
        'logger.info("INFO-MARK")\n'
        'logger.warning("WARNING-MARK")\n'
        'print("LOG_FILE:" + str(current_log_file()))\n'
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'WARNING-MARK' in result.stderr
    assert 'INFO-MARK' not in result.stderr and 'DEBUG-MARK' not in result.stderr
    assert f'LOG_FILE:{tmp_path / "levisim.jsonl"}' in result.stdout
    messages = [r['message'] for r in read_json_lines(tmp_path / 'levisim.jsonl')]
    assert [m for m in messages if m.endswith('-MARK')] == ['DEBUG-MARK', 'INFO-MARK', 'WARNING-MARK']
