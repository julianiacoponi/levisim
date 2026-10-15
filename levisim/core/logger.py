# NOTE: for instructions on how to see these logs locally, see `levisim/docker-compose.yaml`

import os
import sys
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
]


# override with LEVISIM_ELASTICSEARCH_URL to point at a non-default ELK stack
ELASTICSEARCH_URL = os.environ.get('LEVISIM_ELASTICSEARCH_URL', 'http://localhost:9200')

elastic_search_index = 'loguru-logs'
elastic_search_client = Elasticsearch(ELASTICSEARCH_URL) if Elasticsearch is not None else None

# tracks whether the Elasticsearch sink is still worth attempting - see send_to_elasticsearch
_elasticsearch_reachable = elastic_search_client is not None


# custom Loguru sink to send logs to Elasticsearch
def send_to_elasticsearch(message):
    """
    Formats the logger message to be sent to the Elasticsearch client

    Silently stops attempting delivery after the first connection failure, so that an
    unreachable ELK stack produces one warning rather than a traceback per log call.
    """
    global _elasticsearch_reachable
    if not _elasticsearch_reachable:
        return

    record = message.record
    log_entry = {
        'timestamp': record['time'].isoformat(),
        'level': record['level'].name,
        'message': record['message'],
        'file': record['file'].name,
        'line': record['line'],
    }
    try:
        elastic_search_client.index(index=elastic_search_index, body=log_entry)
    except Exception as exception:
        _elasticsearch_reachable = False
        print(
            f'levisim: Elasticsearch at {ELASTICSEARCH_URL} is unreachable'
            f' ({type(exception).__name__}), so the Elasticsearch log sink is now disabled.'
            f'\n  Start it with `docker-compose up -d`, or set LEVISIM_LOG_TO_CONSOLE=1'
            f' to log to stderr instead.',
            file=sys.stderr,
        )


# NOTE: this removes loguru's default stderr sink, so that the sinks below are the only ones
logger.remove()

# NOTE: logs go to Elasticsearch only, and are deliberately NOT printed to the console, so that
# they do not clutter ipython notebook output. Set LEVISIM_LOG_TO_CONSOLE=1 to get them on stderr
# as well - this replaces the previous workflow of commenting out the `logger.remove()` call here.
if os.environ.get('LEVISIM_LOG_TO_CONSOLE'):
    logger.add(sys.stderr, level=os.environ.get('LEVISIM_LOG_LEVEL', 'TRACE'))

if elastic_search_client is not None:
    logger.add(
        send_to_elasticsearch,
        # NOTE: comment this out to avoid logging TRACE level messages
        level='TRACE',
    )
    logger.trace(f'Added {elastic_search_client=} with {elastic_search_index=}')
    logger.info('Loguru logger started and linked to Elasticsearch sink!')
