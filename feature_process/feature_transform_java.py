# -*- coding:UTF-8 -*-
"""
Filename:
Function:
Author:
Create:
"""

import os
from config.global_config import PROJECT_DIR


def build_feature_map(raw_train_feature, train_feature_map_file, index=1):
    cmd = "java -cp {$project_dir}/jar/Feature_Transform_fat.jar feature_process.FeatureMap %s %s %s".replace("{$project_dir}", PROJECT_DIR)
    cmd = cmd % (raw_train_feature, train_feature_map_file, index)
    os.system(cmd)


def transform_libsvm_feature(raw_feature_file, train_feature_map_file, libsvm_file):
    cmd = "java -cp {$project_dir}/jar/Feature_Transform_fat.jar feature_process.LibsvmFeature %s %s %s".replace("{$project_dir}", PROJECT_DIR)
    cmd = cmd % (raw_feature_file, train_feature_map_file, libsvm_file)
    os.system(cmd)


def transform_libsvm_feature_multithread(raw_feature_file, train_feature_map_file, libsvm_file, thread_number):
    cmd = "java -cp {$project_dir}/jar/personalizedAlignmentClassify_fat.jar feature_process.LibsvmFeature %s %s %s %s".replace("{$project_dir}", PROJECT_DIR)
    cmd = cmd % (raw_feature_file, train_feature_map_file, libsvm_file, thread_number)
    os.system(cmd)


