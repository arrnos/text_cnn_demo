#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python2.7
@author: zhangmeng
@file: data_helper.py
@time: 2019/10/08
"""

import re
import pandas as pd
from tensorflow import keras
import numpy as np
import json
import codecs
import os
from data_process import tf_recorder

vocab_path = "/home/zhangmeng/text_cnn_demo/project_data/vocab.json"
tf_record_path = "/home/zhangmeng/text_cnn_demo/project_data/tf_record"
data_info_csv_path = os.path.join(tf_record_path, "data_info_csv")
raw_data_path = "/home/zhangmeng/text_cnn_demo/project_data/raw_feature"
sequence_max_len = 300
num_classes = 2


class DataHelper(object):
    def __init__(self, vocab_path=vocab_path):
        self.vocab_dict = None
        if vocab_path and os.path.isfile(vocab_path):
            self.vocab_dict = json.load(codecs.open(vocab_path, "r", encoding="utf-8"))
            self.vocab_size = len(self.vocab_dict)
            print("词典数：", self.vocab_size)

    def text_preprocess(self, text):
        """
        Clean and segment the text.
        Return a new text.
        """
        text = re.sub(r"[\d+\s+\.!\/_,?=\$%\^\)*\(\+\"\'\+——！:；，。？、~@#%……&*（）·¥\-\|\\《》〈〉～]",
                      "", text)
        text = re.sub("[<>]", "", text)
        text = re.sub("[a-zA-Z0-9]", "", text)
        text = re.sub(r"\s", "", text)
        if not text:
            return ''
        return ' '.join(string for string in text)

    def preprocess(self, data_file, vocab_file, padding_size, test=False):
        """
        Text to sequence, compute vocabulary size, padding sequence.
        Return sequence and label.
        """
        print("Loading data from {} ...".format(data_file))

        x_text, y = [], []
        with codecs.open(data_file, "r", "utf-8") as fin:
            for line in fin:
                arr = line.strip().split("\t")
                if len(arr) < 2:
                    continue
                x_text.append(self.text_preprocess(arr[1]))
                y.append(int(arr[0]))

        if not test:
            # Texts to sequences
            text_preprocesser = keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
            text_preprocesser.fit_on_texts(x_text)
            x = text_preprocesser.texts_to_sequences(x_text)
            word_dict = text_preprocesser.word_index
            json.dump(word_dict, open(vocab_file, 'w'), ensure_ascii=False)
            vocab_size = len(word_dict)
            # max_doc_length = max([len(each_text) for each_text in x])
            x = keras.preprocessing.sequence.pad_sequences(x, maxlen=padding_size,
                                                           padding='post', truncating='post')
            print("Vocabulary size: {:d}".format(vocab_size))
            print("Shape of train data: {}".format(np.shape(x)))
            return x, y, vocab_size
        else:
            word_dict = json.load(open(vocab_file, 'r'))
            vocabulary = word_dict.keys()
            x = [[word_dict[each_word] if each_word in vocabulary else 1 for each_word in each_sentence.split()] for
                 each_sentence in x_text]
            x = keras.preprocessing.sequence.pad_sequences(x, maxlen=padding_size,
                                                           padding='post', truncating='post')
            print("Shape of test data: {}\n".format(np.shape(x)))
            return x, y

    def transform_single_text_2_vector(self, text, senquence_max_len):
        x = [self.vocab_dict[each_word] if each_word in self.vocab_dict else 1 for each_word in text]
        return np.array(x[:senquence_max_len], dtype=np.int64)

    def prepare_vocab_dict(self, raw_data_path=raw_data_path, vocab_file=vocab_path):
        text_preprocesser = keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
        filenames = [os.path.join(raw_data_path, x) for x in os.listdir(raw_data_path)]
        for data_file in filenames:
            print(data_file)
            x_text = []
            with codecs.open(data_file, "r", "utf-8") as fin:
                for line in fin:
                    arr = line.strip().split("\t")
                    if len(arr) < 2:
                        continue
                    x_text.append(self.text_preprocess(arr[1]))
            text_preprocesser.fit_on_texts(x_text)
        word_dict = text_preprocesser.word_index
        json.dump(word_dict, open(vocab_file, 'w', encoding="utf-8"))
        word_dict = json.load(open(vocab_path, 'r', encoding='utf-8'))
        print("vocab dumps finished! word num:", len(word_dict))


def write_raw_data_2_tfRecord(raw_feature_files, tf_record_path, dataHelper, senquence_max_len=500):
    def label_onehot(x):
        one_hot = [0] * num_classes
        one_hot[x] = 1
        return np.array(one_hot, dtype=np.int64)

    for file_i in raw_feature_files:
        df = pd.read_csv(file_i, sep="\t", header=None, names=["label", "text"])
        df.dropna(axis=0, inplace=True)

        df["text"] = df["text"].apply(lambda x: dataHelper.transform_single_text_2_vector(x, senquence_max_len))
        df["label"] = df["label"].apply(lambda x: label_onehot(x))
        data = {df.columns[i]: df[df.columns[i]].values for i in range(len(df.columns))}

        data_ls = [{k: data[k][i] for k in data.keys()} for i in range(len(df))]

        tf_record_file = os.path.join(tf_record_path, os.path.basename(file_i) + ".tfrecord")
        tf_recorder.TFrecorder().writer(tf_record_file, data_info_csv_path, data_ls, var_features=["text"])


def load_dataset_from_tfRecord(start_date=None,end_date=None,tf_record_file_path=tf_record_path, senquence_max_len=sequence_max_len,batch_size=256,epochs=2):
    tf_recorder_files = [os.path.join(tf_record_file_path, x) for x in os.listdir(tf_record_file_path) if
                         x.endswith(".tfrecord")]
    if start_date and end_date:
        tf_recorder_files = [os.path.join(tf_record_file_path, x) for x in os.listdir(tf_record_file_path) if
                         x.endswith(".tfrecord") and start_date<=x.replace(".tfrecord","").split("_")[-1]<=end_date]
    data_set = tf_recorder.TFrecorder().get_dataset(tf_recorder_files, data_info_csv_path, batch_size=batch_size, epoch=epochs,
                                                    padding=([sequence_max_len], [None]))

    return data_set


if __name__ == '__main__':
    import sys
    print(sys.path)
    # helper = DataHelper()
    # helper.prepare_vocab_dict()
    # helper = DataHelper()
    # print("vocab词典数：", len(helper.vocab_dict))
    # raw_data_files = [os.path.join(raw_data_path, x) for x in os.listdir(raw_data_path) if x.startswith("feature_file")]
    # write_raw_data_2_tfRecord(raw_data_files, tf_record_path, helper, senquence_max_len=sequence_max_len)

    # data_set = load_dataset_from_tfRecord(tf_record_path, sequence_max_len)

    # for i in data_set.take(5):
    #     print(i)
