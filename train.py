import argparse
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config.global_config import *
from data_process import data_helper
from data_process import tf_recorder
from evaluate import test_benchmark
from model.TextCnn import TextCnn

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
np.set_printoptions(threshold=np.inf)


def training(train_dataset, valid_dataset, vocab_size, epochs, model_saved_path, log_path):
    model = TextCnn(args.feature_size, args.embedding_size, vocab_size, args.classes_num, args.filter_num,
                    args.filter_list, args.drop_out_ratio)
    model.compile(tf.keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC(), keras.metrics.BinaryAccuracy(), keras.metrics.Precision(),
                           keras.metrics.Recall()])
    model.summary()

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
        # Write TensorBoard logs to logs path, terminal use: tensorboard --logdir=...
        keras.callbacks.TensorBoard(log_dir=log_path)
    ]

    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=callbacks)

    print("\nLog path:", log_path)
    print("\nSave model:", model_saved_path)
    keras.models.save_model(model, model_saved_path, save_format='h5')

    print(history.history)
    return model


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
    parser.add_argument("-epochs", "--epochs", type=int, default=40, help="epochs")
    parser.add_argument("-feature_size", "--feature_size", type=int, default=300, help="feature_size")
    parser.add_argument("-embedding_size", "--embedding_size", type=int, default=32, help="embedding_size")
    parser.add_argument("-classes_num", "--classes_num", type=int, default=2, help="classes_num")
    parser.add_argument("-filter_num", "--filter_num", type=int, default=16, help="filter_num")
    parser.add_argument("-filter_list", "--filter_list", type=str, default="3,4,5,6", help="filter_list")
    parser.add_argument("-drop_out_ratio", "--drop_out_ratio", type=float, default=0.5, help="drop_out_ratio")
    parser.add_argument("-validation_split", "--validation_split", type=float, default=0.1, help="validation_split")
    parser.add_argument("-result_dir", "--result_dir", type=str, default=os.path.join(PROJECT_DATA_DIR, "result"),
                        help="result_dir")
    parser.add_argument("-train_start_date", "--train_start_date", type=str, default="20190406",
                        help="train_start_date")
    parser.add_argument("-train_end_date", "--train_end_date", type=str, default="20190713", help="train_end_date")
    parser.add_argument("-valid_start_date", "--valid_start_date", type=str, default="20190719",
                        help="valid_start_date")
    parser.add_argument("-valid_end_date", "--valid_end_date", type=str, default="20190725", help="valid_end_date")
    parser.add_argument("-test_start_date", "--test_start_date", type=str, default="20190801", help="test_start_date")
    parser.add_argument("-test_end_date", "--test_end_date", type=str, default="20190930", help="test_end_date")
    parser.add_argument("-train_tf_record_folder_name", "--train_tf_record_folder_name", type=str,
                        default="wechat_basic_feature", help="train_tf_record_folder_name")
    parser.add_argument("-valid_tf_record_folder_name", "--valid_tf_record_folder_name", type=str,
                        default="wechat_basic_feature", help="valid_tf_record_folder_name")
    parser.add_argument("-test_tf_record_folder_name", "--test_tf_record_folder_name", type=str,
                        default="wechat_basic_feature_bench_mark", help="test_tf_record_folder_name")
    parser.add_argument("-pos_sample_weight", "--pos_sample_weight", type=float,
                        default=40.0, help="pos_sample_weight")

    parser.add_argument("-is_test", "--is_test", type=bool, default=False, help="is_test")

    args = parser.parse_args()
    print("\nArgument:", args, "\n")

    # Prepare 结果输出目录
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H-%M")
    log_path = os.path.join(args.result_dir, timestamp, "logs")
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    print("\nPrepare train data and test data..")
    date_ls = [args.train_start_date, args.train_end_date, args.valid_start_date, args.valid_end_date]
    train_dataset, valid_dataset = prepare_dataset(date_ls, args.train_tf_record_folder_name,
                                                   args.valid_tf_record_folder_name)

    print("\nTraining")
    model_saved_path = os.path.join(args.result_dir, timestamp, "saved_model")
    vocab_size = data_helper.DataHelper().vocab_size
    model = training(train_dataset, valid_dataset, vocab_size + 1, args.epochs, model_saved_path, log_path)

    print("\nTesting")
    from data_process.make_bench_mark_data import chat_num_ls

    file_base = "total_chat_num"
    chat_num_ls = chat_num_ls
    test_benchmark(model_saved_path, args.test_start_date, args.test_end_date, args.test_tf_record_folder_name, file_base,
         chat_num_ls)
