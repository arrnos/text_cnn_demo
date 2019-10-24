#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python2.7
@author: zhangmeng
@file: transfer_text_feature_2_tf_record.py
@time: 2019/10/23
"""

from data_process import tf_recorder
from config.global_config import *


def main():
    import sys

    start_date = sys.argv[1]
    end_date = sys.argv[2]
    raw_feature_file_path = os.path.join(FEATURE_PATH, "wechat_basic_feature")
    tf_record_file_path = os.path.join(TF_RECORD_PATH, "wechat_basic_feature")
    tf_recorder.TFRecorder().transfer_texts_2_tfRecord_default(start_date, end_date, raw_feature_file_path,
                                                               tf_record_file_path)


if __name__ == '__main__':
    main()
