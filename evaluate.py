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

from data_process.make_bench_mark_data import chat_num_ls

benchmark_file_base = "total_chat_num"
chat_num_ls = chat_num_ls


def evaluate_model(model_saved_path, test_tf_record_path, start_date, end_date, batch_size=265):
    assert os.path.exists(model_saved_path) and os.path.isdir(test_tf_record_path)

    # 加载验证数据
    tfRecorder = tf_recorder.TFRecorder()
    feature_ls = ["chat_content"]
    label_name = "label"
    padding = ({"chat_content": [SEQUENCE_MAX_LEN]}, [None])

    valid_dataset = tfRecorder.get_dataset_from_path(test_tf_record_path, feature_ls, label_name=label_name,
                                                     start_date=start_date, end_date=end_date,
                                                     batch_size=batch_size, padding=padding)

    model = keras.models.load_model(model_saved_path)
    result = model.evaluate(valid_dataset)
    print("\nEvaluate result for %s:\n" % os.path.basename(test_tf_record_path), result)


def test_benchmark(model_path, start_date, end_date, test_tf_record_folder_name, file_base, chat_num_ls):
    for tmp_num in map(str, chat_num_ls):
        test_tf_record_path = os.path.join(TF_RECORD_PATH, test_tf_record_folder_name, "%s_%s" % (file_base, tmp_num))
        assert os.path.isdir(test_tf_record_path)
        evaluate_model(model_path, test_tf_record_path, start_date, end_date)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate model..")
    parser.add_argument("-model_saved_path", "--model_saved_path", type=str, default="", help="model_saved_path")
    parser.add_argument("-test_tf_record_path", "--test_tf_record_path", type=str, default="",
                        help="test_tf_record_path")
    parser.add_argument("-start_date", "--start_date", type=str, default="20190801", help="start_date")
    parser.add_argument("-end_date", "--end_date", type=str, default="20190930", help="end_date")

    args = parser.parse_args()
    print("Argument:", args, "\n")

    assert os.path.exists(args.model_saved_path)
    assert os.path.isdir(args.test_tf_record_path)
    assert args.start_date <= args.end_date and "" not in [args.start_date, args.end_date]

    file_base = "total_chat_num"
    test_benchmark(args.model_saved_path, args.start_date, args.end_date, args.test_tf_record_path, file_base,
                   chat_num_ls)
