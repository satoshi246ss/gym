import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

##ハンドラ取得
get_handler = logging.FileHandler('logfile/logger.log')
logger.addHandler(get_handler)


def do_something():
    logger.info('from logger_lesson2')
    logger.info('from logger_lesson2_debug')