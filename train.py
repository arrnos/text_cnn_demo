from model.TextCnn import TextCnn
import tensorflow as tf
from tensorflow import keras
import argparse
import pprint
from data_preprocess import data_helper
import datetime
import os


def training(x, y, vocab_size, model_saved_path):
    model = TextCnn(args.feature_size, args.embedding_size, vocab_size, args.classes_num, args.filter_num,
                    args.filter_list, args.drop_out_ratio)
    model.compile(tf.keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC(), keras.metrics.Accuracy()])
    model.summary()
    history = model.fit(x, y, args.batch_size, args.epochs, validation_split=args.validation_split)

    print("Save model:", model_saved_path)
    keras.models.save_model(model, model_saved_path)

    print(history.history)


def testing():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="textCnn model..")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("-epochs", "--epochs", type=int, default=2, help="epochs")
    parser.add_argument("-feature_size", "--feature_size", type=int, default=150, help="feature_size")
    parser.add_argument("-embedding_size", "--embedding_size", type=int, default=32, help="embedding_size")
    parser.add_argument("-classes_num", "--classes_num", type=int, default=2, help="classes_num")
    parser.add_argument("-filter_num", "--filter_num", type=int, default=16, help="filter_num")
    parser.add_argument("-filter_list", "--filter_list", type=str, default="3,4,5", help="filter_list")
    parser.add_argument("-drop_out_ratio", "--drop_out_ratio", type=float, default=0.5, help="drop_out_ratio")
    parser.add_argument("-validation_split", "--validation_split", type=float, default=0.1, help="validation_split")
    parser.add_argument("-result_dir", "--result_dir", type=str, default="./project_data/result", help="result_dir")
    args = parser.parse_args()
    print("Argument:", args, "\n")

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H-%M")
    log_path = os.path.join(args.result_dir, timestamp, "logs")
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    print("prepare train data and test data..")
    x_train, y_train, vocab_size = data_helper.preprocess("./project_data/raw_data/feature_file_20190720",
                                                          os.path.join(args.result_dir, timestamp, "vocab.json"),
                                                          args.feature_size)
    y_train = tf.one_hot(y_train,args.classes_num)

    print("Training")
    model_saved_path = os.path.join(args.result_dir, timestamp, "Model_Saved")
    training(x_train, y_train, vocab_size + 1, model_saved_path)

    print("Testing")
    testing()
