import logging

LOG_LEVEL = logging.INFO  # Oder logging.DEBUG für mehr Details

def set_logger():
    logging.basicConfig(
        level=LOG_LEVEL,
        format='[%(levelname)s] %(message)s'
    )

set_logger()