from model.TextCnn import TextCnn
import tensorflow as tf
from tensorflow import keras
import argparse
import pprint
from data_process import data_helper
from data_process import tf_recorder
import datetime
import os
import pandas as pd
import numpy as np
from config.global_config import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
np.set_printoptions(threshold=np.inf)


def evaluate_model(model_saved_path, test_tf_record_path, start_date, end_date):
    assert os.path.exists(model_saved_path) and os.path.isdir(test_tf_record_path)

    # 加载验证数据
    tfRecorder = tf_recorder.TFRecorder()
    feature_ls = ["chat_content"]
    label_name = "label"
    padding = ({"chat_content": [SEQUENCE_MAX_LEN]}, [None])

    valid_dataset = tfRecorder.get_dataset_from_path(test_tf_record_path, feature_ls, label_name=label_name,
                                                     start_date=start_date, end_date=end_date,
                                                     batch_size=args.batch_size, padding=padding)

    model = keras.models.load_model(model_saved_path)
    result = model.evaluate(valid_dataset)
    print("evaluate result for %s:\n" % os.path.basename(test_tf_record_path), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate model..")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("-model_saved_path", "--model_saved_path", type=str, default="", help="model_saved_path")
    parser.add_argument("-valid_tf_record_path", "--valid_tf_record_path", type=str, default="",
                        help="valid_tf_record_path")
    parser.add_argument("-start_date", "--start_date", type=str, default="20190801", help="start_date")
    parser.add_argument("-end_date", "--end_date", type=str, default="20190930", help="end_date")
    parser.add_argument("-local_mode", "--local_mode", type=bool, default=True, help="local_mode")

    args = parser.parse_args()
    print("Argument:", args, "\n")

    assert os.path.isdir(args.model_saved_path)
    assert os.path.isdir(args.valid_tf_record_path)
    assert args.start_date <= args.end_date and "" not in [args.start_date, args.end_date]

    if args.local_mode:
        evaluate_model("E:\\project_data\\text_cnn_demo\\result\\20191030-11-44\\Model_Saved",
                       "E:\\project_data\\text_cnn_demo\\tf_record\\wechat_basic_feature", "20190503", "20190503")
    else:
        evaluate_model(args.model_saved_path, args.valid_tf_record_path, args.start_date, args.end_date)
