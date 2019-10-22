#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python2.7
@author: zhangmeng
@file: global_config.py
@time: 2019/10/22
"""

import os
import getpass
import platform

plat = ""
if platform.system().lower() == 'windows':
    plat = "windows"
elif platform.system().lower() == 'linux':
    plat = "linux"

assert plat in ("linux", "windows"), "platform is not linux or windows!"

PROJECT_NAME = "text_cnn_demo"

if plat == "windows":
    os.chdir("..")
    PROJECT_DIR = os.getcwd()
    PROJECT_DATA_DIR = os.path.join("E:\project_data", PROJECT_NAME)
    ORDER_PATH = os.path.join(PROJECT_DATA_DIR, "raw_data", "raw_order_info")

else:
    user = getpass.getuser()
    PROJECT_DIR = "/home/%s/%s" % (user, PROJECT_NAME)
    PROJECT_DATA_DIR = os.path.join("/home/%s/" % user, "project_data", PROJECT_NAME)
    ORDER_PATH = "/home/online/project_data/public_data/python_fetch_data/raw_order_info"


assert os.path.isdir(PROJECT_DATA_DIR), "%s不存在！" % PROJECT_DATA_DIR
assert os.path.isdir(PROJECT_DIR), "%s不存在！" % PROJECT_DIR
assert os.path.isdir(ORDER_PATH), "%s不存在！" % ORDER_PATH
