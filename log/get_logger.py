# -*- coding:UTF-8 -*-
"""
Filename:
Function:
Author:
Create:
"""

import logging
import logging.handlers
import os
import sys
from config.global_config import PROJECT_DATA_DIR, PROJECT_NAME
from util.dateutil import DateUtil

file_path = os.path.split(os.path.realpath(__file__))[0]
path1 = os.path.dirname(file_path)
path1 = os.path.dirname(path1)
sys.path.append(path1)

LOG_PATH = os.path.join(PROJECT_DATA_DIR, "log")


def get_logger(path):
    log_file = os.path.join(LOG_PATH, path)
    logger = logging.getLogger(PROJECT_NAME)  # 程序顶级目录的名字
    fmt = '[%(asctime)s] - %(filename)s:%(lineno)s - %(name)s - %(message)s'

    formatter = logging.Formatter(fmt)  # 实例化formatter
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)  # 实例化handler
    handler.setFormatter(formatter)  # 为handler添加formatter
    logger.addHandler(handler)  # 为logger添加handler

    logger.setLevel(logging.DEBUG)
    consoleHandle = logging.StreamHandler()
    consoleHandle.setFormatter(formatter)
    logger.addHandler(consoleHandle)

    return logger


G_LOG = get_logger("log_%s" % DateUtil.get_relative_delta_time_str())

if __name__ == "__main__":
    logger = get_logger("log_20170523")
    logger.info('first info message')
    logger.debug('first debug message')
    logger.debug('-----------')
