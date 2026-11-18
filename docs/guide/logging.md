# Logging

`levisim` logs through [loguru](https://loguru.readthedocs.io), with one shared logger,
`levisim.core.logger.logger` (also importable as `from levisim import logger`). This page covers
where its records go, what each level is used for, how to read the log file, and how to browse
the logs in Kibana with or without Docker.

## Where the logs go

Importing `levisim` sets up three sinks:

| Sink | Levels | Default | Controlled by |
|---|---|---|---|
| console (stderr) | WARNING and above | on | `LEVISIM_LOG_LEVEL`, or {func}`~levisim.core.logger.set_console_level` at any time |
| file, JSON lines | DEBUG and above | on, at `~/.levisim/logs/levisim.jsonl` | `LEVISIM_LOG_DIR` (directory), `LEVISIM_LOG_FILE=0` (off) |
| Elasticsearch | TRACE and above | on if the `elasticsearch` package is installed | `LEVISIM_ELASTICSEARCH_URL` (default `http://localhost:9200`) |

So a notebook or script shows only problems, while the file keeps everything down to DEBUG for
later. The environment variables are read once, when `levisim` is imported; set them in the shell
(or in `os.environ` before the first `import levisim`).

- **Console.** `LEVISIM_LOG_LEVEL` takes any loguru level name (`TRACE`, `DEBUG`, `INFO`,
  `SUCCESS`, `WARNING`, `ERROR`, `CRITICAL`, case-insensitive) or `OFF`. The older switch
  `LEVISIM_LOG_TO_CONSOLE=1` still works and means `LEVISIM_LOG_LEVEL=TRACE`;
  `LEVISIM_LOG_LEVEL` wins if both are set. An unknown level gives a one-line notice and WARNING.
- **File.** The file is rotated at 10 MB and the five newest rotated files are kept, so it never
  takes more than about 60 MB. If the directory cannot be created or written, there is no file
  sink and no error. {func}`~levisim.core.logger.current_log_file` returns the path in use, or
  `None`.
- **Elasticsearch.** Only with `pip install "levisim[elk]"` (or `pip install elasticsearch`). Each
  record is sent as it is made. If Elasticsearch does not answer, the first failure prints one
  notice and disables the sink for the rest of the session. The file still has every record, and
  `levisim-logs ingest` (below) loads it once Elasticsearch is up.

To change the console level from a notebook:

```python
from levisim.core.logger import set_console_level

set_console_level('INFO')   # setup and progress milestones
set_console_level('DEBUG')  # also intermediate values
set_console_level(None)     # nothing on the console; the file is unaffected
```

Import it from `levisim.core.logger` as shown. The attribute `levisim.core.logger` is the loguru
logger object itself, because `levisim.core` re-exports `logger` under that name.

## What each level means

`levisim`'s own messages follow this convention:

| Level | Used for | Example |
|---|---|---|
| TRACE | per-frequency or inner-loop detail | one value of a hybridisation function at one $\omega$ |
| DEBUG | intermediate values useful when debugging: matrix shapes, chosen branches, parameters | the shape of the drift matrix; which method branch was taken |
| INFO | setup and progress milestones a user may want | a `MultiTrap` built; a spectrum computed by method X over N points; a simulation started or finished |
| SUCCESS | a requested result was produced, or a check passed | two methods agree to the requested tolerance |
| WARNING | suspicious input, ill-conditioning, fallbacks, deprecated usage | a near-singular matrix; a default substituted for a missing value |
| ERROR | a failure that is about to raise, or that is being swallowed | a file that cannot be read |

`print()` is kept only where printing is the documented output: the `info` properties, which print
`info_str`, and explicit `verbose=True` reports.

## Reading the log file

Each line is one JSON object with the keys `timestamp` (ISO 8601 with UTC offset), `level`,
`level_no` (loguru's severity: 5 TRACE, 10 DEBUG, 20 INFO, 25 SUCCESS, 30 WARNING, 40 ERROR),
`message`, `name` (the module's dotted path), `module`, `function`, `line`, `file`, `process` and
`thread`, plus `extra` (anything bound with `logger.bind(...)`) and `exception` (the traceback)
when present. With [`jq`](https://jqlang.github.io/jq/):

```sh
# follow the log as it is written, one readable line per record
tail -f ~/.levisim/logs/levisim.jsonl | jq -r '"\(.timestamp) \(.level) \(.name):\(.function):\(.line) \(.message)"'

# warnings and above
jq -c 'select(.level_no >= 30)' ~/.levisim/logs/levisim.jsonl

# the records of one module
jq -c 'select(.name == "levisim.multiphysics.multitrap")' ~/.levisim/logs/levisim.jsonl
```

or in Python, `pandas.read_json(path, lines=True)`. Rotated files sit next to the live one, as
`levisim.<date>_<time>.jsonl`. Several Python processes running at once (e.g. `pytest -n`)
append to the same file, and each rotates it on its own schedule, so records of one process can
end up spread over several rotated files.

## Kibana without Docker: `levisim-logs`

Elastic publishes self-contained archives of Elasticsearch and Kibana (each bundles its own Java or
Node.js runtime). The `levisim-logs` command, installed with `levisim`
(`python -m levisim.core.elk` is the same), runs them as two background processes:

```sh
levisim-logs install   # download Elasticsearch and Kibana 9.5.6 into ~/.levisim/elk/9.5.6/, check their SHA-512 sums, unpack
levisim-logs up        # start both, wait until healthy, load the log files, create the Kibana data view
levisim-logs open      # print the Kibana Discover URL for the logs, and open it in a browser
levisim-logs status    # installed? running? reachable? how many documents in the index?
levisim-logs ingest    # load the log files again, e.g. after working with the stack down
levisim-logs down      # stop both
```

- **`install`** checks that both archives exist for this machine before downloading either
  (0.95 GB on an Apple-silicon Mac, 1.3 GB on Linux x86_64; the archives are deleted once
  unpacked, and `--keep-archives` keeps them).
  An archive already downloaded and verified is not fetched again.
- **`up`** writes `config/elasticsearch.yml` (single node, security off, HTTP on port 9200 of the
  loopback interface only) and `config/kibana.yml` (port 5601, `localhost` only), keeping the
  shipped files as `*.yml.orig`. It then starts each program in its own process group, with output
  in `~/.levisim/elk/9.5.6/logs/<program>.out` and its pid in `~/.levisim/elk/9.5.6/run/`.
  Elasticsearch's JVM heap is 1 GB (`--heap`). If something already answers on 9200 or 5601
  (e.g. the Docker stack), `up` uses it rather than starting another. Kibana takes a minute or
  so to come up.
- **`ingest`** bulk-loads the live log file and its rotated predecessors into the `loguru-logs`
  index. Every line gets a fixed id, the SHA-1 of the line, and the live Elasticsearch sink uses the
  same id. Running it again, or ingesting records that already arrived live, adds nothing twice.
  This is how records made while the stack was down, or in sessions without the `elasticsearch`
  package, reach Kibana.
- **`open`** creates the `loguru-logs*` data view (time field `timestamp`) if Kibana answers, and
  prints the Discover URL. `--no-browser` only prints it.
- **`status`** exits with status 0 if both services answer, and 3 otherwise.

`ingest`, `open` and `status` talk to `LEVISIM_ELASTICSEARCH_URL` and `LEVISIM_KIBANA_URL`
(default `http://localhost:5601`), so they work with the Docker stack too. `LEVISIM_ELK_HOME`
and `LEVISIM_ELK_VERSION` change the install directory and the version (Elasticsearch and Kibana
always share one version). The command itself needs only the standard library. The `elasticsearch`
package is only for sending records live.

A Python session that started while Elasticsearch was down has already disabled its Elasticsearch
sink. Its records still go to the file, so run `levisim-logs ingest` to load them, or restart the
kernel to send new records live.

| Platform | Native (`levisim-logs`) | Docker |
|---|---|---|
| macOS, Apple silicon | yes (`darwin-aarch64`) | yes |
| macOS, Intel | no: Elasticsearch has published no `darwin-x86_64` archive since 9.5.4 (Kibana still has one) | yes |
| Linux x86_64 / aarch64 | yes | yes |
| Windows | no (the command manages POSIX processes) | yes |

On an Intel Mac, `levisim-logs install --elk-version 9.5.4` still works natively, with the last
version that has both archives. Otherwise use Docker.

## Docker fallback

`docker-compose.yaml` in the repository root runs the same two services (version 9.5.6,
multi-architecture images, security off, ports bound to `127.0.0.1`), with numbered instructions at
the top of the file. In short:

```sh
docker compose up -d          # from the repo root; older installs: docker-compose up -d
docker compose ps             # wait until both are "healthy"
levisim-logs ingest           # load the log file
levisim-logs open             # data view + Discover URL
docker compose down           # stop (add -v to delete the indexed logs too)
```

## Troubleshooting

- **`levisim-logs status`** says what is installed, running and answering.
- **A program exits while starting.** `up` prints the last lines of its `.out` file. The most
  common cause is a port already in use: check with `lsof -i :9200` (or `:5601`).
- **macOS refuses to run the bundled Java or Node.js.** Archives that `levisim-logs` downloads
  are not marked as quarantined. If you unpacked archives downloaded with a browser instead,
  remove the mark with `xattr -dr com.apple.quarantine ~/.levisim/elk`.
- **Nothing in Kibana.** Widen the time range (Discover opens on the last 24 hours), and check
  that `levisim-logs status` reports documents in `loguru-logs`.
- **Too much on the console.** The console shows WARNING and above unless `LEVISIM_LOG_LEVEL` (or
  the old `LEVISIM_LOG_TO_CONSOLE`) is set in the environment the notebook server was started
  from.
