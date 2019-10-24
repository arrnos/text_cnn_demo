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
from data_process import tf_recorder
from config.global_config import *
from random import sample

wechat_full_sentence_data = os.path.join(RAW_DATA_PATH, "wechat_full_sentence_data")


class DataHelper(object):
    def __init__(self, vocab_path=VOCAB_PATH):
        if vocab_path and os.path.isfile(vocab_path):
            self.vocab_dict = json.load(codecs.open(vocab_path, "r", encoding="utf-8"))
            self.vocab_size = len(self.vocab_dict)
            print("词典数：", self.vocab_size)
        else:
            print("[Warning] %s 不存在，正在重新加载..." % vocab_path)
            self.prepare_vocab_dict()

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
        return np.array(x[-senquence_max_len:], dtype=np.int64)

    def prepare_vocab_dict(self, raw_data_path=wechat_full_sentence_data, vocab_file=VOCAB_PATH):
        text_preprocesser = keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
        filenames = [os.path.join(raw_data_path, x) for x in os.listdir(raw_data_path)]
        if len(filenames) > 120:
            filenames = sorted(np.random.choice(filenames, 120))
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
        word_dict = json.load(open(VOCAB_PATH, 'r', encoding='utf-8'))
        self.vocab_dict = word_dict
        self.vocab_size = len(self.vocab_dict)
        print("vocab dumps finished! word num:", self.vocab_size)


if __name__ == '__main__':
    import sys

    print(sys.path)
    helper = DataHelper()
