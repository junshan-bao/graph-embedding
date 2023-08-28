import sys
import logging


sys.stdout.reconfigure(encoding='utf-8')
file = open('log.txt', encoding="utf-8", mode="w")
logging.basicConfig(
    stream=file,
    datefmt='%d-%m-%Y %H%M%S',
    format='%(asctime)s %(name)s: %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('ge')
if len(logger.handlers) == 0:
    logger.addHandler(logging.StreamHandler(sys.stdout))