from model.TextCnn import TextCnn
import tensorflow as tf
from tensorflow import keras
import argparse
import pprint
from data_process import data_helper
import datetime
import os
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
np.set_printoptions(threshold=np.inf)


def training(train_dataset,valid_dataset, vocab_size,epochs, model_saved_path):
    model = TextCnn(args.feature_size, args.embedding_size, vocab_size, args.classes_num, args.filter_num,
                    args.filter_list, args.drop_out_ratio)
    model.compile(tf.keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC(), keras.metrics.binary_accuracy])
    model.summary()

    #history = model.fit(train_dataset,validation_data = valid_dataset,use_multiprocessing=True,validation_steps=100,steps_per_epoch=200)
    history = model.fit(train_dataset,validation_data = valid_dataset,epochs=epochs)
    print("Save model:", model_saved_path)
    keras.models.save_model(model, model_saved_path)

    print(history.history)
    return model

def testing(model,valid_dataset):
    pass
    #pred = model.predict(valid_dataset.take(5))
    #print(pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="textCnn model..")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("-epochs", "--epochs", type=int, default=3, help="epochs")
    parser.add_argument("-feature_size", "--feature_size", type=int, default=300, help="feature_size")
    parser.add_argument("-embedding_size", "--embedding_size", type=int, default=32, help="embedding_size")
    parser.add_argument("-classes_num", "--classes_num", type=int, default=2, help="classes_num")
    parser.add_argument("-filter_num", "--filter_num", type=int, default=16, help="filter_num")
    parser.add_argument("-filter_list", "--filter_list", type=str, default="3,4,5,6", help="filter_list")
    parser.add_argument("-drop_out_ratio", "--drop_out_ratio", type=float, default=0.8, help="drop_out_ratio")
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
    

    train_dataset = data_helper.load_dataset_from_tfRecord(start_date="20190406",end_date="20190713",batch_size=args.batch_size,epochs=args.epochs)
    valid_dataset = data_helper.load_dataset_from_tfRecord(start_date="20190719",end_date="20190725",batch_size=args.batch_size,epochs=args.epochs)
    
    print("Training")
    model_saved_path = os.path.join(args.result_dir, timestamp, "Model_Saved")
    vocab_size = data_helper.DataHelper().vocab_size
    model = training(train_dataset,valid_dataset, vocab_size + 1,args.epochs, model_saved_path)

    print("Testing")
    testing(model,valid_dataset)
