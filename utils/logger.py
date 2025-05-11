
# logger.py

import logging

def get_logger(name="horse_ai"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(ch)
    return logger
