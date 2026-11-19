"""
Tests of `levisim-logs` (levisim/core/elk.py) without Elasticsearch or Kibana

No server can run here, so HTTP is replaced by fakes (`fake_http`, `fake_urlopen`), the archives
by small tarballs with the same layout, and the two programs by shell scripts that sleep. What is
tested: platform names and URLs, download with SHA-512 checks, safe unpacking, config files,
pid-file handling and process-group stops, `up`/`down` orchestration, bulk-ingest batching and
idempotence, and the Kibana data view.
"""
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from levisim.core import elk
from levisim.core.elk import ElkError, ElkPaths

log_module = importlib.import_module('levisim.core.logger')

VERSION = '9.5.6'


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def make_tarball(top, files):
    """A .tar.gz (bytes) holding `files` {relative path: (text, mode)} under the directory `top`"""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for relative, (text, mode) in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f'{top}/{relative}')
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def fake_archive(product, version=VERSION):
    return make_tarball(f'{product}-{version}', {
        f'bin/{product}': ('#!/bin/sh\nsleep 60\n', 0o755),
        f'config/{product}.yml': ('# shipped config\n', 0o644),
    })


class FakeResponse(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data))}


def serve(files):
    """A replacement for elk._urlopen serving {url: bytes}; counts the requests"""
    calls = []

    def urlopen(url, timeout=60):
        calls.append(url)
        if url not in files:
            raise elk.urllib.error.HTTPError(url, 404, 'Not Found', {}, None)
        return FakeResponse(files[url])
    urlopen.calls = calls
    return urlopen


def published(product, platform_name='linux-x86_64', data=None):
    """{archive url: bytes, checksum url: bytes} as artifacts.elastic.co would serve them"""
    url = elk.archive_url(product, VERSION, platform_name)
    data = fake_archive(product) if data is None else data
    checksum = f'{hashlib.sha512(data).hexdigest()}  {url.rsplit("/", 1)[-1]}\n'.encode()
    return {url: data, url + '.sha512': checksum}


def make_installation(root, scripts=None):
    """An unpacked installation with shell scripts in place of the two programs"""
    paths = ElkPaths(root, VERSION)
    for product in elk.PRODUCTS:
        executable = paths.executable(product)
        executable.parent.mkdir(parents=True)
        script = (scripts or {}).get(product, f'#!/bin/sh\necho "{product} started"\nsleep 60\n')
        executable.write_text(script)
        executable.chmod(0o755)
        (paths.home(product) / 'config').mkdir()
        (paths.home(product) / 'config' / f'{product}.yml').write_text('# shipped config\n')
    return paths


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


# ------------------------------------------------------------------------------------------------
# platform and URLs
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('system, machine, rosetta, expected', [
    ('Darwin', 'arm64', False, 'darwin-aarch64'),
    ('Darwin', 'x86_64', True, 'darwin-aarch64'),  # an x86_64 Python under Rosetta on Apple silicon
    ('Darwin', 'x86_64', False, 'darwin-x86_64'),
    ('Linux', 'x86_64', False, 'linux-x86_64'),
    ('Linux', 'AMD64', False, 'linux-x86_64'),
    ('Linux', 'aarch64', False, 'linux-aarch64'),
])
def test_detect_platform(monkeypatch, system, machine, rosetta, expected):
    monkeypatch.setattr(elk, '_running_under_rosetta', lambda: rosetta)
    assert elk.detect_platform(system, machine) == expected


def test_detect_platform_points_windows_to_docker():
    with pytest.raises(ElkError, match='docker compose up -d'):
        elk.detect_platform('Windows', 'AMD64')


def test_archive_url():
    assert elk.archive_url('elasticsearch', '9.5.6', 'darwin-aarch64') == (
        'https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.5.6-darwin-aarch64.tar.gz'
    )
    assert elk.archive_url('kibana', '9.5.6', 'linux-x86_64') == (
        'https://artifacts.elastic.co/downloads/kibana/kibana-9.5.6-linux-x86_64.tar.gz'
    )
    with pytest.raises(ValueError):
        elk.archive_url('logstash', '9.5.6', 'linux-x86_64')


# ------------------------------------------------------------------------------------------------
# download, checksum, unpacking, install
# ------------------------------------------------------------------------------------------------
def test_parse_sha512_file():
    digest = 'ab' * 64
    assert elk.parse_sha512_file(f'{digest}  kibana-9.5.6-linux-x86_64.tar.gz\n', 'kibana-9.5.6-linux-x86_64.tar.gz') == digest
    assert elk.parse_sha512_file(digest.upper()) == digest
    with pytest.raises(ElkError, match='not a SHA-512'):
        elk.parse_sha512_file('<html>not found</html>')
    with pytest.raises(ElkError, match='checksum file is for'):
        elk.parse_sha512_file(f'{digest}  other.tar.gz', 'kibana-9.5.6-linux-x86_64.tar.gz')


def test_download_verified_checks_the_digest_and_skips_a_verified_archive(monkeypatch, tmp_path):
    files = published('kibana')
    url = elk.archive_url('kibana', VERSION, 'linux-x86_64')
    urlopen = serve(files)
    monkeypatch.setattr(elk, '_urlopen', urlopen)

    archive = elk.download_verified(url, tmp_path, progress=False)
    assert archive.read_bytes() == files[url]
    assert urlopen.calls == [url + '.sha512', url]

    elk.download_verified(url, tmp_path, progress=False)  # already there and verified
    assert urlopen.calls[2:] == [url + '.sha512']


def test_download_verified_rejects_a_corrupt_download(monkeypatch, tmp_path):
    files = published('kibana')
    url = elk.archive_url('kibana', VERSION, 'linux-x86_64')
    data = bytearray(files[url])
    data[len(data) // 2] ^= 0xFF  # one byte changed after the checksum was published
    files[url] = bytes(data)
    monkeypatch.setattr(elk, '_urlopen', serve(files))
    with pytest.raises(ElkError, match='SHA-512 mismatch'):
        elk.download_verified(url, tmp_path, progress=False)
    assert list(tmp_path.iterdir()) == []  # neither the archive nor a .part file is left


def test_extract_archive_returns_the_top_directory(tmp_path):
    archive = tmp_path / 'a.tar.gz'
    archive.write_bytes(fake_archive('elasticsearch'))
    top = elk.extract_archive(archive, tmp_path / 'out')
    assert top == tmp_path / 'out' / f'elasticsearch-{VERSION}'
    assert os.access(top / 'bin' / 'elasticsearch', os.X_OK)


def test_extract_archive_refuses_paths_outside_the_destination(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        info = tarfile.TarInfo('../escaped.txt')
        info.size = 0
        tar.addfile(info, io.BytesIO(b''))
    archive = tmp_path / 'evil.tar.gz'
    archive.write_bytes(buffer.getvalue())
    with pytest.raises((ElkError, tarfile.TarError)):
        elk.extract_archive(archive, tmp_path / 'out')
    assert not (tmp_path / 'escaped.txt').exists()


def test_install_downloads_verifies_and_unpacks_both(monkeypatch, tmp_path):
    files = {**published('elasticsearch'), **published('kibana')}
    urlopen = serve(files)
    monkeypatch.setattr(elk, '_urlopen', urlopen)
    monkeypatch.setattr(elk, '_http', lambda method, url, *a, **k: (200 if url in files else 404, b''))

    paths = elk.install(VERSION, root=tmp_path, platform_name='linux-x86_64', progress=False)
    for product in elk.PRODUCTS:
        assert paths.is_installed(product)
    assert not paths.downloads.exists()  # archives removed after unpacking

    downloads = len(urlopen.calls)
    elk.install(VERSION, root=tmp_path, platform_name='linux-x86_64', progress=False)
    assert len(urlopen.calls) == downloads  # already installed: nothing fetched


def test_install_fails_before_downloading_when_an_archive_is_missing(monkeypatch, tmp_path):
    # as on an Intel Mac: Kibana 9.5.6 has a darwin-x86_64 archive, Elasticsearch has not
    files = published('kibana', 'darwin-x86_64')
    urlopen = serve(files)
    monkeypatch.setattr(elk, '_urlopen', urlopen)
    monkeypatch.setattr(elk, '_http', lambda method, url, *a, **k: (200 if url in files else 404, b''))
    with pytest.raises(ElkError) as error:
        elk.install(VERSION, root=tmp_path, platform_name='darwin-x86_64', progress=False)
    assert 'HTTP 404' in str(error.value)
    assert '9.5.4' in str(error.value) and 'docker compose up -d' in str(error.value)
    assert urlopen.calls == []


# ------------------------------------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------------------------------------
def test_write_configs(tmp_path):
    paths = make_installation(tmp_path / 'with space')
    es_config, kibana_config = elk.write_configs(paths)

    es_text = es_config.read_text()
    for setting in ('discovery.type: single-node', 'xpack.security.enabled: false', 'http.port: 9200',
                    'network.host: _local_'):
        assert setting in es_text
    assert json.dumps(str(paths.data('elasticsearch'))) in es_text
    kibana_text = kibana_config.read_text()
    for setting in ('server.port: 5601', 'server.host: "localhost"', 'elasticsearch.hosts: ["http://localhost:9200"]'):
        assert setting in kibana_text

    # the shipped files are kept once, and not overwritten by a second run
    for config in (es_config, kibana_config):
        original = config.with_name(config.name + '.orig')
        assert original.read_text() == '# shipped config\n'
    elk.write_configs(paths)
    assert es_config.with_name('elasticsearch.yml.orig').read_text() == '# shipped config\n'
    assert paths.data('kibana').is_dir() and paths.run.is_dir() and paths.logs.is_dir()


def test_written_configs_are_valid_yaml(tmp_path):
    yaml = pytest.importorskip('yaml')
    paths = make_installation(tmp_path / 'odd: path')
    es_config, kibana_config = elk.write_configs(paths)
    es = yaml.safe_load(es_config.read_text())
    assert es['xpack.security.enabled'] is False
    assert es['path.data'] == str(paths.data('elasticsearch'))
    assert yaml.safe_load(kibana_config.read_text())['elasticsearch.hosts'] == ['http://localhost:9200']


def test_docker_compose_file_uses_matching_multiarch_version_tags():
    text = (Path(__file__).resolve().parents[1] / 'docker-compose.yaml').read_text()
    assert 'docker.elastic.co/elasticsearch/elasticsearch:9.5.6' in text
    assert 'docker.elastic.co/kibana/kibana:9.5.6' in text
    assert 'sha256' not in text  # digests pin a single architecture
    assert elk.ELK_VERSION == '9.5.6'


# ------------------------------------------------------------------------------------------------
# pid files and processes
# ------------------------------------------------------------------------------------------------
def test_read_pid_handles_missing_garbage_dead_and_foreign_pids(monkeypatch, tmp_path):
    paths = ElkPaths(tmp_path, VERSION)
    paths.run.mkdir(parents=True)
    pid_file = paths.pid_file('kibana')

    assert elk.read_pid(paths, 'kibana') is None  # no file

    pid_file.write_text('not a pid')
    assert elk.read_pid(paths, 'kibana') is None
    assert not pid_file.exists()

    finished = subprocess.Popen([sys.executable, '-c', 'pass'])
    finished.wait()
    pid_file.write_text(str(finished.pid))
    assert elk.read_pid(paths, 'kibana') is None  # dead
    assert not pid_file.exists()

    pid_file.write_text(str(os.getpid()))  # alive, but this test, not Kibana
    monkeypatch.setattr(elk, '_process_command', lambda pid: f'{sys.executable} -m pytest')
    assert elk.read_pid(paths, 'kibana') is None
    assert not pid_file.exists()


def test_start_and_stop_a_process_group(tmp_path):
    paths = make_installation(tmp_path)
    pid = elk.start_process(paths, 'elasticsearch')
    try:
        assert paths.pid_file('elasticsearch').read_text().strip() == str(pid)
        assert elk.read_pid(paths, 'elasticsearch') == pid
        assert wait_for(lambda: 'elasticsearch started' in paths.out_file('elasticsearch').read_text())
    finally:
        assert elk.stop_process(paths, 'elasticsearch', timeout=10)
    assert not elk.pid_is_running(pid)
    assert not paths.pid_file('elasticsearch').exists()
    with pytest.raises(ProcessLookupError):  # the script's `sleep` child went too
        os.killpg(pid, 0)
    assert elk.stop_process(paths, 'elasticsearch') is False  # nothing left to stop


def test_stop_waits_for_children_that_outlive_the_launcher_and_kills_them(tmp_path, capsys):
    # like bin/elasticsearch, the launcher script exits on SIGTERM but leaves a child behind - here
    # one that ignores SIGTERM - so stopping must wait for the whole group, then SIGKILL the rest
    script = (
        '#!/bin/sh\n'
        "(trap '' TERM; echo 'child started'; exec sleep 60) &\n"
        'echo "elasticsearch started"\n'
        'wait\n'
    )
    paths = make_installation(tmp_path, scripts={'elasticsearch': script})
    pid = elk.start_process(paths, 'elasticsearch')
    assert wait_for(lambda: 'child started' in paths.out_file('elasticsearch').read_text())
    started = time.monotonic()
    assert elk.stop_process(paths, 'elasticsearch', timeout=1)
    assert time.monotonic() - started >= 1  # it waited for the straggler rather than returning at once
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
    assert 'sending SIGKILL' in capsys.readouterr().err


def test_start_process_needs_an_installation(tmp_path):
    with pytest.raises(ElkError, match='levisim-logs install'):
        elk.start_process(ElkPaths(tmp_path, VERSION), 'kibana')


# ------------------------------------------------------------------------------------------------
# up / down
# ------------------------------------------------------------------------------------------------
def test_up_starts_both_waits_prepares_and_ingests_then_down_stops_both(monkeypatch, tmp_path):
    paths = make_installation(tmp_path)
    started = lambda product: paths.pid_file(product).exists()  # noqa: E731 - "healthy once started"
    monkeypatch.setattr(elk, 'elasticsearch_health', lambda url=None, timeout=3: 'green' if started('elasticsearch') else None)
    monkeypatch.setattr(elk, 'kibana_level', lambda url=None, timeout=3: 'available' if started('kibana') else None)
    calls = []
    monkeypatch.setattr(elk, 'ensure_index', lambda url=None, index=None: calls.append('index') or True)
    monkeypatch.setattr(elk, 'ensure_data_view', lambda url=None: calls.append('data view') or elk.DATA_VIEW_ID)
    monkeypatch.setattr(elk, 'ingest', lambda url=None, **k: calls.append('ingest'))
    try:
        elk.up(VERSION, root=tmp_path, timeout=10)
        pids = {p: elk.read_pid(paths, p) for p in elk.PRODUCTS}
        assert all(pids.values())
        assert calls == ['index', 'data view', 'ingest']
        assert 'xpack.security.enabled: false' in (paths.home('elasticsearch') / 'config' / 'elasticsearch.yml').read_text()

        elk.up(VERSION, root=tmp_path, timeout=10)  # again: already running, nothing new started
        assert {p: elk.read_pid(paths, p) for p in elk.PRODUCTS} == pids
    finally:
        stopped = elk.down(VERSION, root=tmp_path, timeout=10)
    assert stopped == ['kibana', 'elasticsearch']
    assert not any(elk.pid_is_running(pid) for pid in pids.values())


def test_up_reports_a_program_that_exits_while_starting(monkeypatch, tmp_path):
    make_installation(tmp_path, scripts={'elasticsearch': '#!/bin/sh\necho "fatal: port taken"\nexit 1\n'})
    monkeypatch.setattr(elk, 'elasticsearch_health', lambda url=None, timeout=3: None)
    monkeypatch.setattr(elk, 'kibana_level', lambda url=None, timeout=3: None)
    with pytest.raises(ElkError, match='fatal: port taken'):
        elk.up(VERSION, root=tmp_path, timeout=30)


def test_up_uses_a_stack_that_is_already_serving(monkeypatch, tmp_path, capsys):
    make_installation(tmp_path)
    monkeypatch.setattr(elk, 'elasticsearch_health', lambda url=None, timeout=3: 'yellow')
    monkeypatch.setattr(elk, 'kibana_level', lambda url=None, timeout=3: 'available')
    monkeypatch.setattr(elk, 'ensure_index', lambda *a, **k: True)
    monkeypatch.setattr(elk, 'ensure_data_view', lambda *a, **k: elk.DATA_VIEW_ID)
    monkeypatch.setattr(elk, 'ingest', lambda *a, **k: None)
    monkeypatch.setattr(elk, 'start_process', lambda *a, **k: pytest.fail('nothing should be started'))
    elk.up(VERSION, root=tmp_path, timeout=1)
    assert capsys.readouterr().out.count('already serving') == 2


def test_up_needs_an_installation(tmp_path):
    with pytest.raises(ElkError, match='levisim-logs install'):
        elk.up(VERSION, root=tmp_path)


# ------------------------------------------------------------------------------------------------
# ingest
# ------------------------------------------------------------------------------------------------
class FakeElasticsearchHttp:
    """Just enough of Elasticsearch's REST API for ensure_index and ingest"""

    def __init__(self, reachable=True):
        self.reachable = reachable
        self.ids, self.batches, self.requests = set(), [], []

    def __call__(self, method, url, body=None, headers=None, timeout=10):
        self.requests.append((method, url))
        if not self.reachable:
            return None, b''
        if url.endswith('/_cluster/health'):
            return 200, b'{"status": "green"}'
        if '/_index_template/' in url or url.endswith('/_refresh'):
            return 200, b'{}'
        if method == 'HEAD':
            return 404, b''
        if method == 'PUT':
            return 200, b'{"acknowledged": true}'
        if url.endswith('/_bulk'):
            assert headers['Content-Type'] == 'application/x-ndjson'
            lines = body.splitlines()
            items = []
            for action, source in zip(lines[::2], lines[1::2]):
                doc_id = json.loads(action)['create']['_id']
                assert doc_id == log_module.log_line_id(source)
                json.loads(source)
                items.append({'create': {'status': 409 if doc_id in self.ids else 201}})
                self.ids.add(doc_id)
            self.batches.append(len(items))
            return 200, json.dumps({'errors': False, 'items': items}).encode()
        raise AssertionError(f'unexpected request {method} {url}')


def write_log_lines(path, count, start=0):
    with open(path, 'w', encoding='utf-8') as file:
        for i in range(start, start + count):
            file.write(json.dumps({'timestamp': f'2026-11-18T12:00:00.{i:06d}+00:00', 'level': 'DEBUG', 'message': f'm{i}'}) + '\n')


def test_ingest_batches_skips_malformed_lines_and_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv('LEVISIM_LOG_DIR', str(tmp_path))
    rotated, live = tmp_path / 'levisim.2026-11-17_10-00-00_000000.jsonl', tmp_path / 'levisim.jsonl'
    write_log_lines(rotated, 10)
    write_log_lines(live, 2500, start=10)
    with open(live, 'a') as file:
        file.write('\n{"truncated": \n')  # a blank line and an unfinished one
    os.utime(rotated, (1, 1))  # older

    fake = FakeElasticsearchHttp()
    monkeypatch.setattr(elk, '_http', fake)
    summary = elk.ingest(url='http://es:9200', batch_size=1000, quiet=True)
    assert summary == {'files': 2, 'lines': 2511, 'created': 2510, 'existing': 0, 'malformed': 1, 'failed': 0}
    assert fake.batches == [1000, 1000, 510]
    assert ('PUT', 'http://es:9200/_index_template/levisim-loguru-logs') in fake.requests

    again = elk.ingest(url='http://es:9200', batch_size=1000, quiet=True)
    assert again['created'] == 0 and again['existing'] == 2510


def test_ingest_also_splits_batches_by_size(monkeypatch, tmp_path):
    path = tmp_path / 'one.jsonl'
    write_log_lines(path, 50)
    fake = FakeElasticsearchHttp()
    monkeypatch.setattr(elk, '_http', fake)
    elk.ingest([path], url='http://es:9200', batch_size=1000, max_batch_bytes=2000, quiet=True)
    assert len(fake.batches) > 1 and sum(fake.batches) == 50


def test_ingest_needs_a_reachable_elasticsearch_and_existing_files(monkeypatch, tmp_path):
    path = tmp_path / 'one.jsonl'
    write_log_lines(path, 1)
    monkeypatch.setattr(elk, '_http', FakeElasticsearchHttp(reachable=False))
    with pytest.raises(ElkError, match='levisim-logs up'):
        elk.ingest([path], url='http://es:9200', quiet=True)
    with pytest.raises(ElkError, match='no such log file'):
        elk.ingest([tmp_path / 'missing.jsonl'], url='http://es:9200', quiet=True)


def test_log_files_are_found_oldest_first(monkeypatch, tmp_path):
    for name, mtime in (('levisim.jsonl', 300), ('levisim.b.jsonl', 200), ('levisim.a.jsonl', 100), ('other.jsonl', 1)):
        (tmp_path / name).write_text('')
        os.utime(tmp_path / name, (mtime, mtime))
    assert [p.name for p in elk.log_files(tmp_path)] == ['levisim.a.jsonl', 'levisim.b.jsonl', 'levisim.jsonl']
    assert elk.log_files(tmp_path / 'absent') == []


# ------------------------------------------------------------------------------------------------
# Kibana data view, status, command line
# ------------------------------------------------------------------------------------------------
def fake_kibana(existing=()):
    posted = []

    def http(method, url, body=None, headers=None, timeout=10):
        assert headers and headers.get('kbn-xsrf') == 'true'
        if method == 'GET' and url.endswith(f'/data_view/{elk.DATA_VIEW_ID}'):
            return (200, b'{}') if any(v['id'] == elk.DATA_VIEW_ID for v in existing) else (404, b'{}')
        if method == 'GET' and url.endswith('/api/data_views'):
            return 200, json.dumps({'data_view': list(existing)}).encode()
        if method == 'POST' and url.endswith('/api/data_views/data_view'):
            posted.append(json.loads(body))
            return 200, body.encode()
        raise AssertionError(f'unexpected request {method} {url}')
    http.posted = posted
    return http


def test_ensure_data_view_creates_one(monkeypatch):
    http = fake_kibana()
    monkeypatch.setattr(elk, '_http', http)
    assert elk.ensure_data_view('http://kb:5601') == elk.DATA_VIEW_ID
    (posted,) = http.posted
    assert posted['data_view']['title'] == 'loguru-logs*'
    assert posted['data_view']['timeFieldName'] == 'timestamp'


def test_ensure_data_view_reuses_one_with_the_same_title(monkeypatch):
    http = fake_kibana(existing=[{'id': 'made-by-hand', 'title': 'loguru-logs*'}])
    monkeypatch.setattr(elk, '_http', http)
    assert elk.ensure_data_view('http://kb:5601') == 'made-by-hand'
    assert http.posted == []


def test_open_prints_the_discover_url_even_with_kibana_down(monkeypatch, capsys):
    monkeypatch.setattr(elk, '_http', lambda *a, **k: (None, b''))
    opened = []
    monkeypatch.setattr(elk.webbrowser, 'open', opened.append)
    link = elk.open_kibana('http://kb:5601', browser=False)
    assert link.startswith('http://kb:5601/app/discover#/')
    assert f"dataViewId:'{elk.DATA_VIEW_ID}'" in link
    assert link in capsys.readouterr().out
    assert opened == []


def test_main_status_exit_code_and_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('LEVISIM_ELK_HOME', str(tmp_path / 'elk'))
    monkeypatch.setenv('LEVISIM_LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setattr(elk, '_http', lambda *a, **k: (None, b''))
    assert elk.main(['status']) == 3
    out = capsys.readouterr().out
    assert 'not installed' in out and 'no answer' in out
    assert elk.main(['ingest', str(tmp_path / 'missing.jsonl')]) == 1
    assert 'no such log file' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        elk.main([])  # a command is required


def test_module_runs_as_a_script():
    """`python -m levisim.core.elk --help` works (the console script calls the same main)"""
    repo_root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(p for p in (repo_root, env.get('PYTHONPATH')) if p)
    env['LEVISIM_LOG_FILE'] = '0'
    result = subprocess.run(
        [sys.executable, '-m', 'levisim.core.elk', '--help'], capture_output=True, text=True, timeout=300, env=env,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    for command in ('install', 'up', 'down', 'status', 'ingest', 'open'):
        assert command in result.stdout
