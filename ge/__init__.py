import sys
import logging


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)

file_handler = logging.StreamHandler(open('log.txt', encoding="utf-8", mode="w"))
file_handler.setFormatter(formatter)

logger = logging.getLogger('ge')
logger.setLevel(logging.INFO)
if len(logger.handlers) == 0:
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
