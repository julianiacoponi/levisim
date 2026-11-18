"""
``levisim-logs``: Elasticsearch and Kibana for levisim's logs, without Docker

Elastic publishes self-contained archives of both (each bundles its own Java or Node.js), so on
macOS (Apple silicon) and Linux (x86_64 and aarch64) the stack can run as two ordinary background
processes. This module downloads them, checks them against Elastic's SHA-512 sums, writes minimal
single-node configs (security off, bound to this machine only), starts and stops them, and loads
levisim's JSON-lines log file into Elasticsearch::

    levisim-logs install   # download and unpack into ~/.levisim/elk/<version>/ (~1 GB download)
    levisim-logs up        # start both, wait until healthy, load the log file, create the data view
    levisim-logs open      # print (and open) the Kibana Discover URL for the logs
    levisim-logs status    # installed? running? reachable? how many documents?
    levisim-logs ingest    # load the log file(s) again, e.g. after working with the stack down
    levisim-logs down      # stop both

``python -m levisim.core.elk ...`` is the same command. ``ingest``, ``open`` and ``status`` also
work against the Docker stack of docker-compose.yaml, which is the fallback where no native archive
exists (Intel Macs: Elasticsearch has published no darwin-x86_64 archive since 9.5.4; Windows).

Environment variables: LEVISIM_ELK_HOME (default ~/.levisim/elk), LEVISIM_ELK_VERSION (default
:data:`ELK_VERSION`), LEVISIM_ELASTICSEARCH_URL (default http://localhost:9200),
LEVISIM_KIBANA_URL (default http://localhost:5601), and LEVISIM_LOG_DIR for the log files
(see levisim.core.logger).

Only the standard library is used: the ``elasticsearch`` client package is not needed here.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from levisim.core.logger import (
    LOG_FILE_NAME,
    default_log_dir,
    elastic_search_index,
    log_line_id,
)


__all__ = [
    'ELK_VERSION',
    'ElkError',
    'ElkPaths',
    'detect_platform',
    'archive_url',
    'install',
    'write_configs',
    'up',
    'down',
    'status',
    'ingest',
    'ensure_data_view',
    'discover_url',
    'main',
]


ELK_VERSION = '9.5.6'
ARTIFACTS_URL = 'https://artifacts.elastic.co/downloads'
PRODUCTS = ('elasticsearch', 'kibana')
ELASTICSEARCH_PORT = 9200
KIBANA_PORT = 5601
DATA_VIEW_ID = 'levisim-loguru-logs'
DATA_VIEW_TITLE = f'{elastic_search_index}*'
DEFAULT_HEAP = '1g'

# explicit mapping for the log index, matching levisim.core.logger.log_record_to_dict
INDEX_MAPPING = {
    'mappings': {
        'properties': {
            'timestamp': {'type': 'date'},
            'level': {'type': 'keyword'},
            'level_no': {'type': 'integer'},
            'message': {'type': 'text'},
            'name': {'type': 'keyword'},
            'module': {'type': 'keyword'},
            'function': {'type': 'keyword'},
            'line': {'type': 'integer'},
            'file': {'type': 'keyword'},
            'process': {'type': 'integer'},
            'thread': {'type': 'keyword'},
            'exception': {'type': 'text'},
            # anything bound with logger.bind; flattened, so differing value types cannot conflict
            'extra': {'type': 'flattened'},
        }
    }
}

PLATFORM_ADVICE = {
    'darwin-x86_64': (
        'Elasticsearch has published no macOS x86_64 (Intel) archive since 9.5.4. Use the Docker'
        ' fallback (`docker compose up -d` in the repo root, see docker-compose.yaml), or install'
        ' the last version with both archives: `levisim-logs install --elk-version 9.5.4`.'
    ),
}


class ElkError(RuntimeError):
    """A failure the command line reports as one message, without a traceback"""


# ------------------------------------------------------------------------------------------------
# settings and paths
# ------------------------------------------------------------------------------------------------
def elasticsearch_url():
    return os.environ.get('LEVISIM_ELASTICSEARCH_URL', f'http://localhost:{ELASTICSEARCH_PORT}').rstrip('/')


def kibana_url():
    return os.environ.get('LEVISIM_KIBANA_URL', f'http://localhost:{KIBANA_PORT}').rstrip('/')


def default_elk_home():
    if os.environ.get('LEVISIM_ELK_HOME'):
        return Path(os.environ['LEVISIM_ELK_HOME']).expanduser()
    return Path.home() / '.levisim' / 'elk'


def default_version():
    return os.environ.get('LEVISIM_ELK_VERSION', ELK_VERSION)


@dataclass
class ElkPaths:
    """Where one version of the stack lives: ``<root>/<version>/{<product>-<version>, data, logs, run}``"""
    root: Path
    version: str

    @property
    def base(self):
        return Path(self.root) / self.version

    def home(self, product):
        """The unpacked archive (the directory holding bin/<product>)"""
        return self.base / f'{product}-{self.version}'

    def executable(self, product):
        return self.home(product) / 'bin' / product

    def is_installed(self, product):
        return self.executable(product).is_file()

    def data(self, product):
        return self.base / 'data' / product

    @property
    def logs(self):
        return self.base / 'logs'

    @property
    def run(self):
        return self.base / 'run'

    def pid_file(self, product):
        return self.run / f'{product}.pid'

    def out_file(self, product):
        """stdout and stderr of the product's process"""
        return self.logs / f'{product}.out'

    @property
    def downloads(self):
        return self.base / 'downloads'


def detect_platform(system=None, machine=None):
    """
    Elastic's archive platform name for this machine: darwin-aarch64, linux-x86_64 or linux-aarch64

    darwin-x86_64 is returned for an Intel Mac, for which no current Elasticsearch archive exists;
    a Python running under Rosetta on Apple silicon is recognised as darwin-aarch64.
    """
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == 'Darwin':
        if machine in ('arm64', 'aarch64') or _running_under_rosetta():
            return 'darwin-aarch64'
        return 'darwin-x86_64'
    if system == 'Linux':
        if machine in ('x86_64', 'amd64'):
            return 'linux-x86_64'
        if machine in ('aarch64', 'arm64'):
            return 'linux-aarch64'
    raise ElkError(
        f'levisim-logs runs the native Elasticsearch and Kibana archives on macOS and Linux only,'
        f' not on {system} {machine}. Use the Docker fallback: `docker compose up -d` in the repo'
        f' root (see docker-compose.yaml).'
    )


def _running_under_rosetta():
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'sysctl.proc_translated'], capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip() == '1'


def archive_url(product, version, platform_name):
    """https://artifacts.elastic.co/downloads/<product>/<product>-<version>-<platform>.tar.gz"""
    if product not in PRODUCTS:
        raise ValueError(f'unknown product {product!r}; expected one of {PRODUCTS}')
    return f'{ARTIFACTS_URL}/{product}/{product}-{version}-{platform_name}.tar.gz'


# ------------------------------------------------------------------------------------------------
# HTTP (module-level so that tests can replace them)
# ------------------------------------------------------------------------------------------------
def _urlopen(url, timeout=60):
    return urllib.request.urlopen(url, timeout=timeout)


def _http(method, url, body=None, headers=None, timeout=10):
    """
    One HTTP request: returns (status, body bytes), or (None, b'') if nothing answers at all
    """
    data = body.encode('utf-8') if isinstance(body, str) else body
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except (urllib.error.URLError, OSError):
        return None, b''


def _json_or_none(content):
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------------------------------------
# install
# ------------------------------------------------------------------------------------------------
def parse_sha512_file(text, expected_filename=None):
    """The hex digest from an Elastic ``.sha512`` file (``<digest>  <filename>``)"""
    parts = text.split()
    if not parts or len(parts[0]) != 128 or any(c not in '0123456789abcdef' for c in parts[0].lower()):
        raise ElkError(f'not a SHA-512 checksum file: {text[:200]!r}')
    if expected_filename is not None and len(parts) > 1 and parts[1].lstrip('*') != expected_filename:
        raise ElkError(f'checksum file is for {parts[1]!r}, not {expected_filename!r}')
    return parts[0].lower()


def sha512_of(path, chunk_size=1 << 20):
    digest = hashlib.sha512()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url, destination_dir, progress=True, chunk_size=1 << 20):
    """
    Downloads `url` into `destination_dir`, checking it against ``url + '.sha512'``

    An archive already there with the right checksum is not downloaded again. On a mismatch the
    partial file is deleted and ElkError is raised. Returns the archive's path.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rsplit('/', 1)[-1]
    target = destination_dir / filename

    try:
        with _urlopen(url + '.sha512') as response:
            expected = parse_sha512_file(response.read().decode('utf-8'), filename)
    except urllib.error.URLError as error:
        raise ElkError(f'could not fetch the checksum {url}.sha512: {error}') from error

    if target.is_file() and sha512_of(target) == expected:
        return target

    partial = target.with_name(target.name + '.part')
    digest = hashlib.sha512()
    try:
        with _urlopen(url) as response, open(partial, 'wb') as file:
            total = int(response.headers.get('Content-Length') or 0) if hasattr(response, 'headers') else 0
            done, last_reported = 0, -1
            for chunk in iter(lambda: response.read(chunk_size), b''):
                file.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress and total:
                    percent = int(100 * done / total)
                    if percent // 5 != last_reported:
                        last_reported = percent // 5
                        print(f'\r  {filename}: {percent:3d}% of {total / 1e6:.0f} MB', end='', file=sys.stderr)
            if progress and total:
                print(file=sys.stderr)
    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise ElkError(f'download of {url} failed: {error}') from error

    if digest.hexdigest() != expected:
        partial.unlink(missing_ok=True)
        raise ElkError(f'SHA-512 mismatch for {filename}: the download is corrupt or not the published file')
    partial.replace(target)
    return target


def extract_archive(archive, destination_dir):
    """Unpacks a .tar.gz into `destination_dir` safely; returns the top-level directory it created"""
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        tops = {Path(m.name).parts[0] for m in members if m.name not in ('.', './') and Path(m.name).parts}
        tops.discard('.')
        if hasattr(tarfile, 'data_filter'):
            # refuses absolute paths, '..', links out of the tree and device files
            try:
                tar.extractall(destination_dir, filter='data')
            except tarfile.FilterError as error:
                raise ElkError(f'unsafe member in {Path(archive).name}: {error}') from error
        else:  # pragma: no cover - Pythons without PEP 706's filters
            root = destination_dir.resolve()
            for member in members:
                if not (root / member.name).resolve().is_relative_to(root) or member.isdev():
                    raise ElkError(f'unsafe path in {archive}: {member.name}')
            tar.extractall(destination_dir)
    if len(tops) != 1:
        raise ElkError(f'{archive} does not unpack into a single directory: {sorted(tops)[:5]}')
    return destination_dir / tops.pop()


def install(version=None, root=None, platform_name=None, force=False, keep_archives=False, progress=True):
    """
    Downloads, verifies and unpacks Elasticsearch and Kibana `version` for this platform

    Both archives are checked to exist before either is downloaded, so an unavailable platform
    fails fast. Returns the ElkPaths of the installation.
    """
    version = version or default_version()
    paths = ElkPaths(Path(root) if root else default_elk_home(), version)
    platform_name = platform_name or detect_platform()

    to_install = [p for p in PRODUCTS if force or not paths.is_installed(p)]
    for product in PRODUCTS:
        if product not in to_install:
            print(f'{product} {version} is already installed in {paths.home(product)}')
    if not to_install:
        return paths

    for product in to_install:
        url = archive_url(product, version, platform_name)
        status_code, _ = _http('HEAD', url, timeout=30)
        if status_code != 200:
            advice = PLATFORM_ADVICE.get(platform_name, '')
            reason = 'no answer' if status_code is None else f'HTTP {status_code}'
            raise ElkError(f'no {product} {version} archive for {platform_name} ({reason} from {url}). {advice}'.strip())

    for product in to_install:
        url = archive_url(product, version, platform_name)
        print(f'downloading {url}')
        archive = download_verified(url, paths.downloads, progress=progress)
        print(f'  SHA-512 verified; unpacking into {paths.base}')
        if force and paths.home(product).exists():
            shutil.rmtree(paths.home(product))
        top = extract_archive(archive, paths.base)
        if top != paths.home(product):
            raise ElkError(f'{archive.name} unpacked into {top.name}, expected {paths.home(product).name}')
        if not keep_archives:
            archive.unlink()
    if paths.downloads.is_dir() and not any(paths.downloads.iterdir()):
        paths.downloads.rmdir()
    print(f'installed Elasticsearch and Kibana {version} in {paths.base}; start them with `levisim-logs up`')
    return paths


# ------------------------------------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------------------------------------
def _yaml_string(value):
    return json.dumps(str(value))  # a JSON string is a valid YAML scalar, whatever the path holds


def elasticsearch_config(paths, port=ELASTICSEARCH_PORT):
    return '\n'.join([
        '# Written by `levisim-logs up`: one Elasticsearch node for levisim\'s logs, reachable from',
        '# this machine only (loopback), with security off. The shipped file is elasticsearch.yml.orig.',
        'cluster.name: levisim-logs',
        'node.name: levisim-logs',
        f'path.data: {_yaml_string(paths.data("elasticsearch"))}',
        f'path.logs: {_yaml_string(paths.logs / "elasticsearch")}',
        'network.host: _local_',
        f'http.port: {port}',
        'discovery.type: single-node',
        'xpack.security.enabled: false',
        '# a nearly full laptop disk would otherwise make the index read-only',
        'cluster.routing.allocation.disk.threshold_enabled: false',
        '',
    ])


def kibana_config(paths, port=KIBANA_PORT, es_port=ELASTICSEARCH_PORT):
    return '\n'.join([
        '# Written by `levisim-logs up`: Kibana for levisim\'s logs, reachable from this machine only.',
        '# The shipped file is kibana.yml.orig.',
        'server.host: "localhost"',
        f'server.port: {port}',
        f'elasticsearch.hosts: ["http://localhost:{es_port}"]',
        f'path.data: {_yaml_string(paths.data("kibana"))}',
        '',
    ])


def write_configs(paths, es_port=ELASTICSEARCH_PORT, kibana_port=KIBANA_PORT):
    """
    Writes config/elasticsearch.yml and config/kibana.yml, keeping each shipped file once as .orig

    Returns the two paths written.
    """
    written = []
    for product, text in (
        ('elasticsearch', elasticsearch_config(paths, es_port)),
        ('kibana', kibana_config(paths, kibana_port, es_port)),
    ):
        config = paths.home(product) / 'config' / f'{product}.yml'
        config.parent.mkdir(parents=True, exist_ok=True)
        original = config.with_name(config.name + '.orig')
        if config.is_file() and not original.exists():
            shutil.copy2(config, original)
        config.write_text(text, encoding='utf-8')
        paths.data(product).mkdir(parents=True, exist_ok=True)
        written.append(config)
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.run.mkdir(parents=True, exist_ok=True)
    return written


# ------------------------------------------------------------------------------------------------
# processes
# ------------------------------------------------------------------------------------------------
def _process_command(pid):
    """The command line of `pid` from ps, or None if ps cannot tell"""
    try:
        result = subprocess.run(
            ['ps', '-ww', '-o', 'command=', '-p', str(pid)], capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # a process os.kill can see but ps cannot report on is left as unknown rather than foreign
    return result.stdout.strip() if result.returncode == 0 else None


def pid_is_running(pid, expect=None):
    """True if process `pid` exists (and, given `expect`, its command line contains that text)"""
    _reap(pid)  # an exited child of this process would otherwise linger as a zombie, and look alive
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, belongs to someone else
    if expect is None:
        return True
    command = _process_command(pid)
    return command is None or expect.lower() in command.lower()


def read_pid(paths, product):
    """
    The pid of our running `product`, or None; a stale pid file (dead or reused pid) is removed
    """
    pid_file = paths.pid_file(product)
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        if pid_file.exists():
            pid_file.unlink()
        return None
    if pid_is_running(pid, expect=product):
        return pid
    pid_file.unlink(missing_ok=True)
    return None


def start_process(paths, product, extra_env=None):
    """Starts bin/<product> in its own session, output to logs/<product>.out; returns the pid"""
    executable = paths.executable(product)
    if not executable.is_file():
        raise ElkError(f'{product} is not installed in {paths.base}; run `levisim-logs install` first')
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.run.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(extra_env or {})
    with open(paths.out_file(product), 'ab') as out:
        process = subprocess.Popen(
            [str(executable)], cwd=paths.home(product), env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True,  # its own process group, so `down` stops all of its children
        )
    paths.pid_file(product).write_text(f'{process.pid}\n')
    return process.pid


def stop_process(paths, product, timeout=30.0):
    """Stops our `product` (SIGTERM to its process group, SIGKILL after `timeout`); True if it was running"""
    pid = read_pid(paths, product)
    if pid is None:
        return False
    try:
        target = os.getpgid(pid)
        kill = os.killpg
    except ProcessLookupError:
        paths.pid_file(product).unlink(missing_ok=True)
        return False
    except OSError:
        target, kill = pid, os.kill
    try:
        kill(target, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and pid_is_running(pid):
        time.sleep(0.2)
    if pid_is_running(pid):
        try:
            kill(target, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(0.2)
        _reap(pid)
    paths.pid_file(product).unlink(missing_ok=True)
    return True


def _reap(pid):
    # an exited child of this very process stays a zombie, which os.kill still sees, until reaped
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass  # not our child (the usual case: up and down run in separate commands)


# ------------------------------------------------------------------------------------------------
# health
# ------------------------------------------------------------------------------------------------
def elasticsearch_health(url=None, timeout=3):
    """The cluster health status ('green', 'yellow', 'red'), or None if Elasticsearch does not answer"""
    status_code, content = _http('GET', f'{url or elasticsearch_url()}/_cluster/health', timeout=timeout)
    body = _json_or_none(content) if status_code == 200 else None
    return body.get('status') if isinstance(body, dict) else None


def kibana_level(url=None, timeout=3):
    """Kibana's overall status level ('available', 'degraded', ...), or None if it does not answer"""
    status_code, content = _http('GET', f'{url or kibana_url()}/api/status', timeout=timeout)
    body = _json_or_none(content)
    try:
        return body['status']['overall']['level']
    except (TypeError, KeyError):
        return None


def wait_until(check, timeout, interval=2.0, alive=None, what='service'):
    """Polls `check()` until truthy; stops early if `alive()` turns False. Returns check()'s value or None"""
    deadline = time.monotonic() + timeout
    while True:
        value = check()
        if value:
            return value
        if alive is not None and not alive():
            raise ElkError(f'{what} exited while starting')
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval)


def _tail(path, lines=15):
    try:
        return '\n'.join(Path(path).read_text(errors='replace').splitlines()[-lines:])
    except OSError:
        return ''


# ------------------------------------------------------------------------------------------------
# up / down / status
# ------------------------------------------------------------------------------------------------
def up(version=None, root=None, timeout=180.0, heap=DEFAULT_HEAP, run_ingest=True):
    """
    Starts Elasticsearch, then Kibana, waits for both, and prepares the index and the data view

    Anything already answering on the ports (e.g. the Docker stack) is used rather than started.
    Unless `run_ingest` is False, the log files are then loaded with :func:`ingest`.
    """
    paths = ElkPaths(Path(root) if root else default_elk_home(), version or default_version())
    for product in PRODUCTS:
        if not paths.is_installed(product):
            raise ElkError(f'{product} {paths.version} is not installed in {paths.base}; run `levisim-logs install` first')
    write_configs(paths)
    es_url, kb_url = f'http://localhost:{ELASTICSEARCH_PORT}', f'http://localhost:{KIBANA_PORT}'

    for product, url, health, extra_env in (
        ('elasticsearch', es_url, lambda: elasticsearch_health(es_url) in ('green', 'yellow'),
         {'ES_JAVA_OPTS': os.environ.get('ES_JAVA_OPTS', f'-Xms{heap} -Xmx{heap}')}),
        ('kibana', kb_url, lambda: kibana_level(kb_url) == 'available', {}),
    ):
        pid = read_pid(paths, product)
        if pid is not None:
            print(f'{product} is already running (pid {pid})')
        elif health():
            print(f'something is already serving {product} at {url}; using it')
            continue
        else:
            pid = start_process(paths, product, extra_env)
            print(f'started {product} (pid {pid}); waiting for {url} ...')
        try:
            ready = wait_until(health, timeout, alive=lambda: pid_is_running(pid), what=product)
        except ElkError:
            raise ElkError(
                f'{product} exited while starting. Last lines of {paths.out_file(product)}:\n'
                f'{_tail(paths.out_file(product))}'
            ) from None
        if not ready:
            raise ElkError(
                f'{product} did not become healthy within {timeout:.0f} s; it is left running.'
                f' See {paths.out_file(product)}, or `levisim-logs down` to stop it.'
            )
        print(f'{product} is up at {url}')

    ensure_index(es_url)
    data_view = ensure_data_view(kb_url)
    if run_ingest:
        ingest(url=es_url)
    if data_view:
        print(f'logs in Kibana: {discover_url(kb_url, data_view)}')
    return paths


def down(version=None, root=None, timeout=30.0):
    """Stops Kibana, then Elasticsearch, if levisim-logs started them; returns the products stopped"""
    paths = ElkPaths(Path(root) if root else default_elk_home(), version or default_version())
    stopped = []
    for product in reversed(PRODUCTS):
        if stop_process(paths, product, timeout=timeout):
            print(f'stopped {product}')
            stopped.append(product)
        else:
            print(f'{product} was not running (as far as levisim-logs knows)')
    return stopped


def status(version=None, root=None):
    """Prints what is installed, running and reachable; returns True if both services answer"""
    paths = ElkPaths(Path(root) if root else default_elk_home(), version or default_version())
    es_url, kb_url = elasticsearch_url(), kibana_url()
    es_health, kb_level = elasticsearch_health(es_url), kibana_level(kb_url)
    for product, url, answer in (('elasticsearch', es_url, es_health), ('kibana', kb_url, kb_level)):
        installed = f'installed in {paths.home(product)}' if paths.is_installed(product) else 'not installed'
        pid = read_pid(paths, product)
        running = f'running (pid {pid})' if pid else 'not started by levisim-logs'
        print(f'{product} {paths.version}: {installed}; {running}; {url}: {answer or "no answer"}')
    if es_health:
        status_code, content = _http('GET', f'{es_url}/{elastic_search_index}/_count')
        count = (_json_or_none(content) or {}).get('count') if status_code == 200 else None
        print(f'index {elastic_search_index}: {count if count is not None else "absent"} documents')
    files = log_files()
    if files:
        size = sum(f.stat().st_size for f in files)
        print(f'log files: {len(files)} in {files[0].parent} ({size / 1e6:.1f} MB)')
    else:
        print(f'log files: none in {default_log_dir()}')
    return bool(es_health) and bool(kb_level)


# ------------------------------------------------------------------------------------------------
# index, ingest, data view
# ------------------------------------------------------------------------------------------------
def ensure_index(url=None, index=elastic_search_index):
    """
    Puts an index template with levisim's mapping, and creates `index` if it does not exist

    The template (for ``<index>*``) also covers an index the live Elasticsearch sink creates first.
    Returns True if the index exists now.
    """
    url = url or elasticsearch_url()
    template = {'index_patterns': [f'{index}*'], 'priority': 200, 'template': INDEX_MAPPING}
    status_code, _ = _http(
        'PUT', f'{url}/_index_template/levisim-{index}', json.dumps(template),
        {'Content-Type': 'application/json'},
    )
    if status_code is None:
        return False
    status_code, _ = _http('HEAD', f'{url}/{index}')
    if status_code == 200:
        return True
    if status_code is None:
        return False
    status_code, content = _http(
        'PUT', f'{url}/{index}', json.dumps(INDEX_MAPPING), {'Content-Type': 'application/json'},
    )
    # 400 resource_already_exists_exception: created meanwhile by the live sink
    return status_code == 200 or b'resource_already_exists' in content


def log_files(log_dir=None):
    """The live log file and its rotated predecessors, oldest first"""
    log_dir = Path(log_dir) if log_dir else default_log_dir()
    stem, suffix = os.path.splitext(LOG_FILE_NAME)
    files = [p for p in log_dir.glob(f'{stem}*{suffix}') if p.is_file()] if log_dir.is_dir() else []
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))


def _bulk(url, index, batch):
    """Sends one _bulk request of (id, line) pairs; returns (created, existing, failures)"""
    body = ''.join(
        json.dumps({'create': {'_index': index, '_id': doc_id}}) + '\n' + line + '\n'
        for doc_id, line in batch
    )
    status_code, content = _http(
        'POST', f'{url}/_bulk', body, {'Content-Type': 'application/x-ndjson'}, timeout=120,
    )
    if status_code is None:
        raise ElkError(f'Elasticsearch at {url} stopped answering during the ingest')
    response = _json_or_none(content)
    if status_code != 200 or not isinstance(response, dict):
        raise ElkError(f'bulk request rejected (HTTP {status_code}): {content[:300]!r}')
    created = existing = 0
    failures = []
    for item in response.get('items', []):
        result = item.get('create') or item.get('index') or {}
        if result.get('status') in (200, 201):
            created += 1
        elif result.get('status') == 409:  # this line was indexed before, live or by an earlier ingest
            existing += 1
        else:
            failures.append(str(result.get('error', result))[:200])
    return created, existing, failures


def ingest(files=None, url=None, index=elastic_search_index, batch_size=1000, max_batch_bytes=5_000_000, quiet=False):
    """
    Loads JSON-lines log files into Elasticsearch, idempotently

    Every line gets the id :func:`levisim.core.logger.log_line_id` gives it, the same id the live
    Elasticsearch sink uses, so lines already indexed (live, or by an earlier ingest) are counted as
    existing rather than duplicated. Lines that are not JSON objects are skipped and counted.

    Args:
        files: paths to read; default the live log file and its rotated predecessors
        url: Elasticsearch URL; default LEVISIM_ELASTICSEARCH_URL or http://localhost:9200
        batch_size: documents per _bulk request
        max_batch_bytes: a request is also sent once its body reaches this size

    Returns:
        dict of counts: files, lines, created, existing, malformed, failed
    """
    url = (url or elasticsearch_url()).rstrip('/')
    files = [Path(f) for f in files] if files else log_files()
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise ElkError(f'no such log file: {", ".join(missing)}')
    summary = {'files': len(files), 'lines': 0, 'created': 0, 'existing': 0, 'malformed': 0, 'failed': 0}
    if not files:
        if not quiet:
            print(f'no log files to ingest in {default_log_dir()}')
        return summary
    if elasticsearch_health(url) is None:
        raise ElkError(f'Elasticsearch does not answer at {url}; start it with `levisim-logs up`')
    ensure_index(url, index)

    failures = []
    batch, batch_bytes = [], 0

    def flush():
        nonlocal batch, batch_bytes
        if batch:
            created, existing, failed = _bulk(url, index, batch)
            summary['created'] += created
            summary['existing'] += existing
            summary['failed'] += len(failed)
            failures.extend(failed[:3])
        batch, batch_bytes = [], 0

    for path in files:
        with open(path, encoding='utf-8', errors='replace') as file:
            for raw in file:
                line = raw.rstrip('\r\n')
                if not line.strip():
                    continue
                summary['lines'] += 1
                if not isinstance(_json_or_none(line), dict):
                    summary['malformed'] += 1
                    continue
                batch.append((log_line_id(line), line))
                batch_bytes += len(line) + 100
                if len(batch) >= batch_size or batch_bytes >= max_batch_bytes:
                    flush()
    flush()
    _http('POST', f'{url}/{index}/_refresh')
    if not quiet:
        print(
            f'ingested {summary["lines"]} lines from {summary["files"]} file(s) into {index}:'
            f' {summary["created"]} new, {summary["existing"]} already there,'
            f' {summary["malformed"]} not JSON, {summary["failed"]} failed'
        )
        for failure in failures[:3]:
            print(f'  e.g. {failure}', file=sys.stderr)
    return summary


def ensure_data_view(url=None, title=DATA_VIEW_TITLE, view_id=DATA_VIEW_ID):
    """
    Makes sure Kibana has a data view for `title` with time field ``timestamp``

    Returns the id of the data view (an existing one with the same title is reused), or None if
    Kibana does not answer.
    """
    url = url or kibana_url()
    headers = {'kbn-xsrf': 'true', 'Content-Type': 'application/json'}
    status_code, _ = _http('GET', f'{url}/api/data_views/data_view/{view_id}', headers=headers)
    if status_code is None:
        return None
    if status_code == 200:
        return view_id
    status_code, content = _http('GET', f'{url}/api/data_views', headers=headers)
    for view in (_json_or_none(content) or {}).get('data_view', []) if status_code == 200 else []:
        if view.get('title') == title:
            return view.get('id')
    body = {'data_view': {'id': view_id, 'title': title, 'name': 'levisim logs', 'timeFieldName': 'timestamp'}}
    status_code, content = _http('POST', f'{url}/api/data_views/data_view', json.dumps(body), headers)
    if status_code == 200:
        return view_id
    print(f'Kibana did not create the data view (HTTP {status_code}): {content[:200]!r}', file=sys.stderr)
    return None


def discover_url(url=None, view_id=DATA_VIEW_ID, since='now-24h'):
    """Kibana Discover on the levisim logs, newest first"""
    return (
        f'{url or kibana_url()}/app/discover#/'
        f"?_g=(time:(from:{since},to:now))"
        f"&_a=(dataSource:(dataViewId:'{view_id}',type:dataView),sort:!(!(timestamp,desc)))"
    )


def open_kibana(url=None, browser=True):
    """Creates the data view if Kibana answers, prints the Discover URL, and opens it in a browser"""
    url = url or kibana_url()
    view_id = ensure_data_view(url)
    if view_id is None:
        print(f'Kibana does not answer at {url} (start it with `levisim-logs up`); the URL will work once it does:', file=sys.stderr)
        view_id = DATA_VIEW_ID
    link = discover_url(url, view_id)
    print(link)
    if browser:
        webbrowser.open(link)
    return link


# ------------------------------------------------------------------------------------------------
# command line
# ------------------------------------------------------------------------------------------------
def _parser():
    parser = argparse.ArgumentParser(
        prog='levisim-logs',
        description='Run Elasticsearch and Kibana for levisim\'s logs without Docker, and load the log file into them.',
    )
    parser.add_argument('--elk-version', default=None, help=f'Elastic stack version (default {ELK_VERSION}, or LEVISIM_ELK_VERSION)')
    parser.add_argument('--home', default=None, help='install directory (default ~/.levisim/elk, or LEVISIM_ELK_HOME)')
    commands = parser.add_subparsers(dest='command', required=True, metavar='COMMAND')

    p = commands.add_parser('install', help='download, verify (SHA-512) and unpack Elasticsearch and Kibana')
    p.add_argument('--force', action='store_true', help='reinstall even if already unpacked')
    p.add_argument('--keep-archives', action='store_true', help='keep the downloaded .tar.gz files')

    p = commands.add_parser('up', help='start both in the background and wait until healthy')
    p.add_argument('--timeout', type=float, default=180.0, help='seconds to wait for each (default 180)')
    p.add_argument('--heap', default=DEFAULT_HEAP, help=f'Elasticsearch JVM heap (default {DEFAULT_HEAP})')
    p.add_argument('--no-ingest', action='store_true', help='do not load the log files once up')

    p = commands.add_parser('down', help='stop both')
    p.add_argument('--timeout', type=float, default=30.0, help='seconds to wait before SIGKILL (default 30)')

    commands.add_parser('status', help='what is installed, running and reachable (exit status 3 unless both answer)')

    p = commands.add_parser('ingest', help='bulk-load JSON-lines log files into the loguru-logs index')
    p.add_argument('files', nargs='*', help='files to load (default: the log file and its rotated predecessors)')
    p.add_argument('--batch-size', type=int, default=1000, help='documents per bulk request (default 1000)')

    p = commands.add_parser('open', help='create the Kibana data view and print (and open) the Discover URL')
    p.add_argument('--no-browser', action='store_true', help='only print the URL')
    return parser


def main(argv=None):
    """Entry point of the ``levisim-logs`` command; returns the exit status"""
    args = _parser().parse_args(argv)
    common = {'version': args.elk_version, 'root': args.home}
    try:
        if args.command == 'install':
            install(force=args.force, keep_archives=args.keep_archives, **common)
        elif args.command == 'up':
            up(timeout=args.timeout, heap=args.heap, run_ingest=not args.no_ingest, **common)
        elif args.command == 'down':
            down(timeout=args.timeout, **common)
        elif args.command == 'status':
            return 0 if status(**common) else 3
        elif args.command == 'ingest':
            ingest(files=args.files or None, batch_size=args.batch_size)
        elif args.command == 'open':
            open_kibana(browser=not args.no_browser)
    except ElkError as error:
        print(f'levisim-logs: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())
