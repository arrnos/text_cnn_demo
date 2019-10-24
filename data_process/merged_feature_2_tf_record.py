#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python2.7
@author: zhangmeng
@file: merged_feature_2_tf_record.py
@time: 2019/10/23
"""
"""
将/t分割的merged_feature_file文件转换为tf_record，并存储到指定目录。
支持功能：
（1）负采样：负样本的采样比率
（2）自定义特征列，指定label name
（3）特征值支持多维数组，一维定长向量，一维不定长向量
（4）支持padding
"""
from config.global_config import *
from util.dateutil import DateUtil
from data_process import data_helper, tf_recorder
import pandas as pd
import numpy as np
import tensorflow as tf
from log.get_logger import G_LOG as log

def transfer_txt_2_tfRecord(raw_feature_file, tf_record_file, data_info_csv_path, column_names, label_name,
                            need_feature_cols, negative_ratio=None, var_length_cols=None, col_preprocess_func=None):
    """
    :param raw_feature_file:  原始txt文件路径, 特征用/t分割
    :param tf_record_file: 要生成的tf_record文件路径
    :param data_info_csv_path: tfRecord 属性信息存储路径
    :param column_names: 每列对应的name
    :param label_name:  label name
    :param negative_ratio: 负样本按多少比率采样
    :param need_feature_cols: 考虑的特征列，按顺序
    :param var_length_cols: 指定不定长一维向量的特征名
    :param col_preprocess_func: 预处理函数字典，可以为某些列指定预处理函数，如字符串转为词索引
    """
    # 参数检验
    assert label_name in column_names and label_name not in need_feature_cols
    assert len(need_feature_cols) == len(set(need_feature_cols) & set(column_names))

    df_data = pd.read_csv(raw_feature_file, sep="\t", encoding="utf-8")
    df_data.columns = column_names
    df_data.dropna(axis=0, inplace=True)

    if negative_ratio:
        assert NUM_CLASS == 2, "负采样目前支持2分类！"
        # 采样后的索引列
        sample_idx_ls = [True if x or np.random.random() < negative_ratio else False for x in
                         df_data[label_name] == 1]
        df_data = df_data[sample_idx_ls]
    log.info("正样本：负样本=%s:%s" % (df_data[label_name].sum(), len(df_data) - df_data[label_name].sum()))

    if col_preprocess_func:
        for feature_name, func in col_preprocess_func.items():
            df_data[feature_name] = df_data[feature_name].apply(func)
    # del 不需要的col
    drop_cols = [x for x in column_names if x not in [label_name] + need_feature_cols]
    df_data.drop(drop_cols, axis=1, inplace=True)

    examples = []
    for i in range(len(df_data)):
        example = dict(df_data.iloc[i])
        examples.append(example)
    log.info("examples 已生成，examples[0]:", examples[0])

    tf_recorder.TFrecorder().writer(tf_record_file, data_info_csv_path, examples, var_features=var_length_cols)


def transfer_multi_day(start_date, end_date, raw_feature_floder_name, tf_record_folder_name, column_names,
                       label_name, need_feature_cols, negative_ratio=None, var_length_cols=None,
                       col_preprocess_func=None):
    """
    参数见：transfer_txt_2_tfRecord
    """
    raw_feature_file = os.path.join(FEATURE_PATH, raw_feature_floder_name, raw_feature_floder_name + "_%s")
    tf_record_file_path = os.path.join(TF_RECORD_PATH, tf_record_folder_name)
    if not os.path.isdir(tf_record_file_path):
        os.makedirs(tf_record_file_path)
    tf_record_file = os.path.join(tf_record_file_path, tf_record_folder_name + "_%s.tfrecord")
    data_info_csv_path = os.path.join(TF_RECORD_PATH, tf_record_folder_name, "data_info.csv")

    date_ls = DateUtil.get_every_date(start_date, end_date)
    for date in date_ls:
        log.info(date)
        transfer_txt_2_tfRecord(raw_feature_file % date, tf_record_file % date, data_info_csv_path, column_names,
                                label_name, need_feature_cols, negative_ratio, var_length_cols, col_preprocess_func)


def main():
    import sys

    dataHelper = data_helper.DataHelper()
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    raw_feature_floder_name = "wechat_basic_feature"
    tf_record_folder_name = sys.argv[3]
    negative_ratio = None if sys.argv[4] == "None" else float(sys.argv[4])
    log.info("负采样率：", negative_ratio)

    column_names = ["label", "opp_id", "acc_id", "create_time", "today_student_chat_num", "today_teacher_chat_num",
                    "today_total_chat_num", "hist_student_chat_num", "hist_teacher_chat_num", "hist_total_chat_num",
                    "chat_content"]
    label_name = "label"
    need_features_cols = ["today_student_chat_num", "today_teacher_chat_num", "today_total_chat_num",
                          "hist_student_chat_num", "hist_teacher_chat_num", "hist_total_chat_num", "chat_content"]
    var_length_cols = ["chat_content"]
    col_preprocess_func = {
        "chat_content": lambda text: dataHelper.transform_single_text_2_vector(text, SEQUENCE_MAX_LEN),
        "label": lambda x: tf.one_hot(x, NUM_CLASS).numpy().astype(np.int64)  # label onehot
    }
    transfer_multi_day(start_date, end_date, raw_feature_floder_name, tf_record_folder_name, column_names,
                       label_name, need_features_cols, negative_ratio=negative_ratio,
                       var_length_cols=var_length_cols, col_preprocess_func=col_preprocess_func)


if __name__ == '__main__':
    main()
