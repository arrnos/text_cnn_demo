# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from config.global_config import *
from util.dateutil import DateUtil


class TFRecorder(object):
    '''
    helper function for write and read TFrecord files
    
    Args:
        path: where to write and read
        
    '''

    def _data_info_fn(self, one_example, var_features):

        data_info = pd.DataFrame(columns=['name', 'type', 'shape', 'isbyte', 'length_type', 'default'])
        i = 0
        for key in one_example:
            value = one_example[key]
            dtype = str(value.dtype)
            shape = value.shape
            if len(shape) > 1:
                data_info.loc[i] = {'name': key,
                                    'type': dtype,
                                    'shape': shape,
                                    'isbyte': True,
                                    'length_type': 'fixed',
                                    'default': np.NaN}
                i += 1
                data_info.loc[i] = {'name': key + '_shape',
                                    'type': 'int64',
                                    'shape': (len(shape),),
                                    'isbyte': False,
                                    'length_type': 'fixed',
                                    'default': np.NaN}
                i += 1
            else:
                data_info.loc[i] = {'name': key,
                                    'type': dtype if "U" not in dtype else "str",  # 支持字符串类型
                                    'shape': shape,
                                    'isbyte': False if "U" not in dtype else True,
                                    'length_type': "fixed" if key not in var_features else "var",
                                    'default': np.NaN}
                i += 1
        return data_info

    def _feature_writer(self, df, value, features):
        '''
        Writes a single feature in features
        Args:
            value: an array : the value of the feature to be written

        Note:
            the tfrecord type will be as same as the numpy dtype
            if the feature's rank >= 2, the shape (type: int64) will also be added in features
            the name of shape info is: name+'_shape'
        Raises:
            TypeError: Type is not one of ('int64', 'float32')
        '''
        name = df['name']
        isbyte = df['isbyte']
        length_type = df['length_type']
        default = df['default']
        dtype = df['type']
        shape = df['shape']

        # get the corresponding type function
        if isbyte:
            if dtype == "str":
                feature_typer = lambda x: tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.encode()]))
            else:
                feature_typer = lambda x: tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.tostring()]))
        else:
            if dtype in ['int64', np.int64]:
                feature_typer = lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=x))
            elif dtype in ['float32', np.float32]:
                feature_typer = lambda x: tf.train.Feature(float_list=tf.train.FloatList(value=x))
            else:
                raise TypeError("Type needs one of {'int64','float32'}, but given %s for %s" % (dtype, name))
        # check whether the input is (1D-array)
        # if the input is a scalar, convert it to list
        if dtype == "str":
            features[name] = feature_typer(value)
        # if the rank of input array >=2, flatten the input and save shape info
        elif len(shape) > 1:
            features[name] = feature_typer(value.reshape(-1))
            # write shape info
            features['%s_shape' % name] = tf.train.Feature(int64_list=tf.train.Int64List(value=shape))
        elif len(shape) == 0 and dtype != "str":  # scalar 需要转换为list
            features[name] = feature_typer([value])
        elif len(shape) == 1:
            features[name] = feature_typer(value)

    def _writer(self, tf_records_file_path, data_info_csv_path, examples, var_features=()):
        print("\nWrite tfRecord data : %s\n" % tf_records_file_path)

        if not os.path.isfile(data_info_csv_path):
            self.data_info = self._data_info_fn(examples[0], var_features)
            self.data_info.to_csv(data_info_csv_path, index=False)
        else:
            self.data_info = pd.read_csv(data_info_csv_path, dtype={'isbyte': bool})
            self.data_info['shape'] = self.data_info['shape'].apply(
                lambda s: [int(i) for i in s[1:-1].split(',') if i != ''])

        self.path = tf_records_file_path

        self.num_example = len(examples)
        self.num_feature = len(self.data_info)
        writer = tf.io.TFRecordWriter('%s' % self.path)
        for e in np.arange(self.num_example):
            features = {}
            for f in np.arange(self.num_feature):
                feature_name = self.data_info.loc[f]['name']
                if '_shape' not in feature_name:
                    self._feature_writer(self.data_info.loc[f], examples[e][feature_name], features)

            tf_features = tf.train.Features(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            tf_serialized = tf_example.SerializeToString()
            writer.write(tf_serialized)
        writer.close()

        print('%s examples has been written to %s' % (self.num_example, self.path))

    def _create_parser(self, data_info, feature_list, label_name, reshape):

        names = data_info['name']
        types = data_info['type']
        shapes = data_info['shape']
        isbytes = data_info['isbyte']
        defaults = data_info['default']
        length_types = data_info['length_type']

        # 验证参数
        for feature_name in feature_list:
            assert feature_name in names.values, "col：%s 不存在tfRecord文件中，tf_cols:%s, 请检查！ " % (
                feature_name, names.values)

        if label_name:
            assert label_name in names.values, "col：%s 不存在tfRecord文件中，请检查！ " % label_name

        if reshape is None:
            reshape = {}

        def parser(example_proto):

            def specify_features():
                specified_features = {}
                for i in np.arange(len(names)):
                    # which type
                    if isbytes[i]:
                        t = tf.string
                        s = ()
                    else:
                        t = types[i]
                        s = shapes[i]
                    # has default_value?
                    if defaults[i] == np.NaN:
                        d = np.NaN
                    else:
                        d = defaults[i]
                    # length varies
                    if length_types[i] == 'fixed':
                        specified_features[names[i]] = tf.io.FixedLenFeature(s, t)
                    elif length_types[i] == 'var':
                        specified_features[names[i]] = tf.io.VarLenFeature(t)
                    else:
                        raise TypeError("length_type is not one of 'var', 'fixed'")
                return specified_features

            # decode each parsed feature and reshape
            def decode_reshape(parsed_example):
                # store all decoded&shaped features
                final_features = {}
                for i in np.arange(len(names)):
                    # exclude shape info
                    if '_shape' not in names[i]:
                        # decode
                        if isbytes[i] and types[i] != "str":
                            decoded_value = tf.io.decode_raw(parsed_example[names[i]], types[i])
                        else:
                            # Varlen value needs to be converted to dense format
                            if length_types[i] == 'var':
                                decoded_value = tf.sparse.to_dense(parsed_example[names[i]])
                            else:
                                decoded_value = parsed_example[names[i]]
                        # reshape
                        if '%s_shape' % names[i] in parsed_example.keys():
                            tf_shape = parsed_example['%s_shape' % names[i]]
                            decoded_value = tf.reshape(decoded_value, tf_shape)
                        elif names[i] in reshape.keys():
                            if len(reshape[names[i]]) > 0:
                                decoded_value = tf.reshape(decoded_value, reshape[names[i]])
                        final_features[names[i]] = decoded_value

                return final_features

            # create a dictionary to specify how to parse each feature 
            specified_features = specify_features()
            # parse all features of an example
            parsed_example = tf.io.parse_single_example(example_proto, specified_features)
            final_features = decode_reshape(parsed_example)
            if label_name:
                result = ({x: final_features[x] for x in feature_list}, final_features[label_name])
            else:
                result = {x: final_features[x] for x in feature_list}
            return result

        return parser

    def _get_dataset(self, tf_record_files, data_info_csv_path, feature_list, label_name=None, shuffle=True,
                     shuffle_buffer=10000, batch_size=1, padding=None, reshape=None, prefetch_buffer=1000):
        """
        从文件列表解析每个tfRecord文件，返回dataset。
        :param tf_record_files: 文件路径list
        :param data_info_csv_path: 解析字典csv
        :param feature_list: 需要读取的特征列表，不能包括label
        :param label_name: 默认为None,即文件内容只存在特征数据，没有类别标签。如果指定label_name，则将label_name列视为label列
        """
        self.filenames = tf_record_files

        print('\nRead tfRecord data from %s x %s\n' % (tf_record_files[0], len(tf_record_files)))
        data_info = pd.read_csv(data_info_csv_path, dtype={'isbyte': bool})
        data_info['shape'] = data_info['shape'].apply(lambda s: [int(i) for i in s[1:-1].split(',') if i != ''])
        print("Data Info:\n", data_info)

        dataset = tf.data.TFRecordDataset(self.filenames)

        self.parse_function = self._create_parser(data_info, feature_list, label_name, reshape)
        self.dataset = dataset.map(self.parse_function, num_parallel_calls=None)

        self.dataset_raw = self.dataset.prefetch(prefetch_buffer)
        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=shuffle_buffer)
        if padding is not None:
            self.dataset = self.dataset.padded_batch(batch_size, padded_shapes=padding)
        else:
            self.dataset = self.dataset.batch(batch_size)

        return self.dataset.map(lambda x,y:(x,y[:,1]))

    def get_dataset_from_path(self, tf_record_file_path, feature_list, start_date=None, end_date=None, label_name=None,
                              batch_size=1, padding=None, shuffle=True, shuffle_buffer=10000, reshape=None,
                              prefetch_buffer=1000):
        """
        从tf_record文件夹路径，读取取其目录下的tfRecord文件
        :param feature_list: 需要读取的特征列表，不能包括label
        :param start_date: 根据起止时间筛选tfRecord的文件数量，默认为None
        :param end_date: 根据起止时间筛选tfRecord的文件数量，默认为None
        :param label_name: 默认为None,即文件内容只存在特征数据，没有类别标签。如果指定label_name，则将label_name列视为label列
        其他参数见：_get_dataset
        """

        # 参数验证
        assert feature_list, "feature list不能为空!"
        if label_name:
            assert label_name not in feature_list

        tf_recorder_files = [os.path.join(tf_record_file_path, x) for x in os.listdir(tf_record_file_path) if
                             x.endswith(".tfrecord")]

        data_info_csv_path = os.path.join(tf_record_file_path, "data_info.csv")
        assert os.path.isfile(data_info_csv_path), "文件不存在，无法解析dataset！%s" % data_info_csv_path

        if start_date and end_date:
            tf_recorder_files = [os.path.join(tf_record_file_path, x) for x in os.listdir(tf_record_file_path) if
                                 x.endswith(".tfrecord") and start_date <= x.replace(".tfrecord", "").split("_")[
                                     -1] <= end_date]

        data_set = self._get_dataset(tf_recorder_files, data_info_csv_path, feature_list, label_name=label_name,
                                     batch_size=batch_size, padding=padding, shuffle=shuffle,
                                     shuffle_buffer=shuffle_buffer, reshape=reshape,
                                     prefetch_buffer=prefetch_buffer)

        return data_set

    def transfer_single_text_2_tfRecord(self, raw_feature_file, tf_record_file, data_info_csv_path, column_names,
                                        label_name, need_feature_cols, negative_ratio=None, var_length_cols=None,
                                        col_preprocess_func=None):
        """
        处理一个文件，将其转换为tfRecord格式
        :param raw_feature_file:  原始txt文件路径, 特征用"/t"分割
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
        # 聊天数据可能存在缺失，字符串为""
        df_data.dropna(axis=0, inplace=True)
        if (df_data.size == 0):
            log.warn("去除缺失值后，没有聊天样本存在！无法生成tfRecord文件：%s" % tf_record_file)
            return

        if negative_ratio:
            assert NUM_CLASS == 2, "负采样目前支持2分类！"
            # 采样后的索引列
            sample_idx_ls = [True if x or np.random.random() < negative_ratio else False for x in
                             df_data[label_name] == 1]
            df_data = df_data[sample_idx_ls]
        pos_sample_num = df_data[label_name].sum()
        neg_sample_num = len(df_data) - pos_sample_num
        print("正样本:%s，负样本：%s 比例:%.1f" % (
            pos_sample_num, neg_sample_num, np.nan if pos_sample_num == 0 else neg_sample_num / pos_sample_num))

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
        print("examples 已生成，examples[0]\n%s" % examples[0])

        self._writer(tf_record_file, data_info_csv_path, examples, var_features=var_length_cols)

    def transfer_texts_2_tfRecord(self, start_date, end_date, raw_feature_file_path, tf_record_file_path,
                                  column_names,
                                  label_name, need_feature_cols, negative_ratio=None, var_length_cols=None,
                                  col_preprocess_func=None):

        """
        根据起止时间，将指定目录下的text文件，都存储为tfRecord格式
        参数见：transfer_single_text_2_tfRecord
        """
        raw_feature_folder_name = os.path.basename(raw_feature_file_path)
        tf_record_folder_name = os.path.basename(tf_record_file_path)

        raw_feature_file = os.path.join(raw_feature_file_path, raw_feature_folder_name + "_%s")
        tf_record_file = os.path.join(tf_record_file_path, tf_record_folder_name + "_%s.tfrecord")
        data_info_csv_path = os.path.join(tf_record_file_path, "data_info.csv")

        if not os.path.isdir(tf_record_file_path):
            os.makedirs(tf_record_file_path)

        date_ls = DateUtil.get_every_date(start_date, end_date)
        for date in date_ls:
            print(date)
            self.transfer_single_text_2_tfRecord(raw_feature_file % date, tf_record_file % date, data_info_csv_path,
                                                 column_names,
                                                 label_name, need_feature_cols, negative_ratio, var_length_cols,
                                                 col_preprocess_func)

    def transfer_texts_2_tfRecord_default(self, start_date, end_date, raw_feature_file_path, tf_record_file_path,
                                          column_names=None, need_features_cols=None, var_length_cols=None,
                                          col_preprocess_func=None, label_name=None):
        """
        带默认参数设置的转换util,样本类型适用于wechat_basic_feature这种。
        """
        negative_ratio = None

        if not label_name:
            label_name = "label"

        if not column_names:
            column_names = ["label", "opp_id", "acc_id", "create_time", "today_student_chat_num",
                            "today_teacher_chat_num", "today_total_chat_num", "hist_student_chat_num",
                            "hist_teacher_chat_num", "hist_total_chat_num", "chat_content"]
        if not need_features_cols:
            need_features_cols = ["today_student_chat_num", "today_teacher_chat_num", "today_total_chat_num",
                                  "hist_student_chat_num", "hist_teacher_chat_num", "hist_total_chat_num",
                                  "chat_content"]
        if not var_length_cols:
            var_length_cols = ["chat_content"]

        if not col_preprocess_func:
            from data_process.data_helper import DataHelper
            dataHelper = DataHelper()
            col_preprocess_func = {
                "chat_content": lambda text: dataHelper.transform_single_text_2_vector(text, SEQUENCE_MAX_LEN),
                "label": lambda x: tf.one_hot(x, NUM_CLASS).numpy().astype(np.int64)  # label onehot
            }

        self.transfer_texts_2_tfRecord(start_date, end_date, raw_feature_file_path, tf_record_file_path,
                                       column_names, label_name, need_features_cols, negative_ratio=negative_ratio,
                                       var_length_cols=var_length_cols, col_preprocess_func=col_preprocess_func)
