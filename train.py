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


def training(train_dataset, valid_dataset, vocab_size, epochs, model_saved_path):
    model = TextCnn(args.feature_size, args.embedding_size, vocab_size, args.classes_num, args.filter_num,
                    args.filter_list, args.drop_out_ratio)
    model.compile(tf.keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC(), keras.metrics.binary_accuracy])
    model.summary()

    # history = model.fit(train_dataset,validation_data = valid_dataset,use_multiprocessing=True,validation_steps=100,steps_per_epoch=200)
    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs,
                        class_weight={0: 1., 1: args.pos_sample_weight})
    print("Save model:", model_saved_path)
    keras.models.save_model(model, model_saved_path)

    print(history.history)
    return model


def testing(model, valid_dataset):
    pass
    # pred = model.predict(valid_dataset.take(5))
    # print(pred)


def prepare_dataset(date_ls, train_tf_record_folder_name, valid_tf_record_folder_name):
    # date_ls = ["20190406","20190713","20190719","20190725"]
    assert len(date_ls) == 4
    assert date_ls[0] < date_ls[1] < date_ls[2] < date_ls[3]
    print("加载数据集...\n训练集：%s-%s\n验证集：%s-%s" % (date_ls[0], date_ls[1], date_ls[2], date_ls[3]))

    tfRecorder = tf_recorder.TFRecorder()
    train_tf_record_path = os.path.join(TF_RECORD_PATH, train_tf_record_folder_name)
    valid_tf_record_path = os.path.join(TF_RECORD_PATH, valid_tf_record_folder_name)
    feature_ls = ["chat_content"]
    label_name = "label"
    padding = ({"chat_content": [SEQUENCE_MAX_LEN]}, [None])
    train_dataset = tfRecorder.get_dataset_from_path(train_tf_record_path, feature_ls, label_name=label_name,
                                                     start_date=date_ls[0], end_date=date_ls[1],
                                                     batch_size=args.batch_size, padding=padding)
    valid_dataset = tfRecorder.get_dataset_from_path(valid_tf_record_path, feature_ls, label_name=label_name,
                                                     start_date=date_ls[2], end_date=date_ls[3],
                                                     batch_size=args.batch_size, padding=padding)
    return train_dataset, valid_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="textCnn model..")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("-epochs", "--epochs", type=int, default=20, help="epochs")
    parser.add_argument("-feature_size", "--feature_size", type=int, default=300, help="feature_size")
    parser.add_argument("-embedding_size", "--embedding_size", type=int, default=32, help="embedding_size")
    parser.add_argument("-classes_num", "--classes_num", type=int, default=2, help="classes_num")
    parser.add_argument("-filter_num", "--filter_num", type=int, default=16, help="filter_num")
    parser.add_argument("-filter_list", "--filter_list", type=str, default="3,4,5,6", help="filter_list")
    parser.add_argument("-drop_out_ratio", "--drop_out_ratio", type=float, default=0.8, help="drop_out_ratio")
    parser.add_argument("-validation_split", "--validation_split", type=float, default=0.1, help="validation_split")
    parser.add_argument("-result_dir", "--result_dir", type=str, default=os.path.join(PROJECT_DATA_DIR, "result"),
                        help="result_dir")
    parser.add_argument("-train_start_date", "--train_start_date", type=str, default="20190406",
                        help="train_start_date")
    parser.add_argument("-train_end_date", "--train_end_date", type=str, default="20190713", help="train_end_date")
    parser.add_argument("-test_start_date", "--test_start_date", type=str, default="20190719", help="test_start_date")
    parser.add_argument("-test_end_date", "--test_end_date", type=str, default="20190725", help="test_end_date")
    parser.add_argument("-train_tf_record_folder_name", "--train_tf_record_folder_name", type=str,
                        default="wechat_basic_feature_full", help="train_tf_record_folder_name")
    parser.add_argument("-valid_tf_record_folder_name", "--valid_tf_record_folder_name", type=str,
                        default="wechat_basic_feature_full", help="valid_tf_record_folder_name")
    parser.add_argument("-pos_sample_weight", "--pos_sample_weight", type=float,
                        default=40.0, help="pos_sample_weight")

    args = parser.parse_args()
    print("Argument:", args, "\n")

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H-%M")
    log_path = os.path.join(args.result_dir, timestamp, "logs")
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    print("prepare train data and test data..")
    date_ls = [args.train_start_date, args.train_end_date, args.test_start_date, args.test_end_date]
    train_dataset, valid_dataset = prepare_dataset(date_ls, args.train_tf_record_folder_name,
                                                   args.valid_tf_record_folder_name)

    print("Training")
    model_saved_path = os.path.join(args.result_dir, timestamp, "Model_Saved")
    vocab_size = data_helper.DataHelper().vocab_size
    model = training(train_dataset, valid_dataset, vocab_size + 1, args.epochs, model_saved_path)

    print("Testing")
    testing(model, valid_dataset)
