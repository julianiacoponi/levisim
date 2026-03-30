# NOTE: for instructions on how to see these logs locally, see `levisim/docker-compose.yaml`

import sys
from loguru import logger
from elasticsearch import Elasticsearch


__all__ = [
    'logger',
    'elastic_search_client',
    'elastic_search_index',
]


# NOTE: this removes the default logger, and adds trace level logging to the new default
logger.remove()
logger.add(sys.stdout, level='TRACE')


# NOTE: this then removes the default stops logs from printing to stdout
# (i.e. appearing in ipython notebook output)
# therefore, comment it out to get logs printed locally :-)
logger.remove()


# configure Elasticsearch client
elastic_search_client = Elasticsearch('http://localhost:9200')
elastic_search_index = 'loguru-logs'


# custom Loguru sink to send logs to Elasticsearch
def send_to_elasticsearch(message):
    """
    Formats the logger message to be sent to the Elasticsearch client
    """
    record = message.record
    log_entry = {
        'timestamp': record['time'].isoformat(),
        'level': record['level'].name,
        'message': record['message'],
        'file': record['file'].name,
        'line': record['line'],
    }
    elastic_search_client.index(index=elastic_search_index, body=log_entry)


# add the Elasticsearch sink to Loguru
logger.add(
    send_to_elasticsearch,
    # NOTE: comment this out to avoid logging TRACE level messages
    level='TRACE',
)
logger.trace(f'Added {elastic_search_client=} with {elastic_search_index=}')
logger.info('Loguru logger started and linked to Elasticsearch sink!')
