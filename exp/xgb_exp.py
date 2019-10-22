# -*- coding:UTF-8 -*-
import os
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
from datetime import datetime


HOME_DIR = os.environ["HOME"]


def train_model(libsvm_file, dump_file, model_file):
    dtrain = xgb.DMatrix(libsvm_file)

    param = {
        "max_depth": 6,
        "eta": 0.3,
        "silent": 1,
        "objective": "binary:logistic",
        "nthread": 10,
        "eval_metric": "auc",
        "tree_method": "exact"
    }
    num_round = 100
    watchlist = [(dtrain, 'eval_train')]
    bst = xgb.train(param, dtrain, num_round, watchlist)
    bst.dump_model(dump_file)
    bst.save_model(model_file)


def test_model(libsvm_file, model_file, score_file, print_result=True):
    dtest = xgb.DMatrix(libsvm_file)
    y = dtest.get_label()
    bst = xgb.Booster(model_file=model_file)
    bst.save_model(model_file)
    preds = bst.predict(dtest)
    with open(score_file, 'w') as fout:
        for index, score in enumerate(list(preds)):
            fout.write("%s\t%s\n" % ('{:.9f}'.format(score), y[index]))

    if print_result:
        if y.shape[0] > 1:
            fpr, tpr, _ = roc_curve(y, preds, pos_label=1)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = "nan"
        print "roc_auc:", roc_auc
        sorted_index = preds.argsort()[::-1]
        sorted_results = y[sorted_index]
        observation_points = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, sorted_index.shape[0]/2, sorted_index.shape[0]]
        observation_points = [i for i in observation_points if i <= sorted_index.shape[0]]
        for i in observation_points:
            if i <= sorted_index.shape[0]:
                print sorted_results[:i].sum(), i
        top_5_percentage = int(sorted_index.shape[0] * 0.05)
        print "%s\t%s" % (sorted_results[:top_5_percentage].sum(), top_5_percentage)


def build_feature_map(raw_train_feature, train_feature_map_file, index=1):
    from ..feature_process.feature_transform_java import build_feature_map
    build_feature_map(raw_train_feature, train_feature_map_file, index)


def transform_libsvm_feature(raw_feature_file, train_feature_map_file, libsvm_file, thread_number=8):
    from ..feature_process.feature_transform_java import transform_libsvm_feature_multithread
    transform_libsvm_feature_multithread(raw_feature_file, train_feature_map_file, libsvm_file, thread_number)


def merge_data_file(input_dir, file_name_base, start_date, end_date, output_file):
    start_file_name = file_name_base % start_date
    end_file_name = file_name_base % end_date
    file_list = os.listdir(input_dir)
    file_list.sort()
    if len(file_list) <= 0:
        raise Exception("not existing file to merge")
    filter_list = [os.path.join(input_dir, file_name) for file_name in file_list if
                   start_file_name <= file_name <= end_file_name]
    cmd = 'cat %s > %s' % (" ".join(filter_list), output_file)
    os.system(cmd)


def print_nice_model(train_feature_map_file, dump_raw_file, dump_nice_file):
    from ..feature_process.print_nice_model import print_nice_model
    print_nice_model(train_feature_map_file, dump_raw_file, dump_nice_file)


def generate_candidate_file(merged_raw_data, test_score_file, merged_score_file, sorted_merged_score_file):
    score_file = test_score_file + "_tmp"
    cmd = '''awk -F"\t" '{print $1}' %s > %s ''' % (test_score_file, score_file)
    print "Execute cmd :\n" + cmd
    os.system(cmd)
    cmd = 'paste %s %s > %s' % (score_file, merged_raw_data, merged_score_file)
    print "Execute cmd :\n" + cmd
    os.system(cmd)
    cmd = 'sort -k 1nr %s > %s' % (merged_score_file, sorted_merged_score_file)
    print "Execute cmd :\n" + cmd
    os.system(cmd)


def xgb_exp(train_start_date, train_end_date, test_start_date, test_end_date):
    xgb_exp_dir = os.path.join(HOME_DIR, "project_data/xgb", "xgb_exp", "xgb_exp_%s" % datetime.now().strftime("%Y%m%d"))
    if not os.path.exists(xgb_exp_dir):
        print "mkdir %s" % xgb_exp_dir
        os.makedirs(xgb_exp_dir)

    wechat_tf_feature_dir = os.path.join(HOME_DIR, "project_data/xgb", "feature_file/wechat_tf_feature")
    benchmark_label_data_dir = os.path.join(HOME_DIR, "project_data/xgb", "benchmark_label_data")

    feature_file_name_base = "wechat_tf_feature_%s"
    benchmark_data_file_name_base = "benchmark_label_data_%s"

    merged_train_feature = os.path.join(xgb_exp_dir, "merged_train_feature")
    merged_test_feature = os.path.join(xgb_exp_dir, "merged_test_feature")
    merged_benchmark_data = os.path.join(xgb_exp_dir, "merged_benchmark_data")

    train_feature_map = os.path.join(xgb_exp_dir, "train_feature_map")
    train_libsvm_feature_file = os.path.join(xgb_exp_dir, "train_libsvm_feature_file")
    test_libsvm_feature_file = os.path.join(xgb_exp_dir, "test_libsvm_feature_file")

    raw_dump_file = os.path.join(xgb_exp_dir, "raw_dump_file")
    nice_dump_file = os.path.join(xgb_exp_dir, "nice_dump_file")
    model_file = os.path.join(xgb_exp_dir, "model_file")

    score_file = os.path.join(xgb_exp_dir, "score_file")
    paste_score_file = os.path.join(xgb_exp_dir, "paste_score_file")
    sorted_paste_score_file = os.path.join(xgb_exp_dir, "sorted_paste_score_file")

    print "merge feature and benchmark data..."
    merge_data_file(wechat_tf_feature_dir, feature_file_name_base, train_start_date, train_end_date, merged_train_feature)
    merge_data_file(wechat_tf_feature_dir, feature_file_name_base, test_start_date, test_end_date, merged_test_feature)
    merge_data_file(benchmark_label_data_dir, benchmark_data_file_name_base, test_start_date, test_end_date, merged_benchmark_data)

    print "build feature map..."
    build_feature_map(merged_train_feature, train_feature_map)

    print "transform feature to libsvm..."
    transform_libsvm_feature(merged_train_feature, train_feature_map, train_libsvm_feature_file)
    transform_libsvm_feature(merged_test_feature, train_feature_map, test_libsvm_feature_file)

    print "train model..."
    train_model(train_libsvm_feature_file, raw_dump_file, model_file)

    print "transform nice dump file..."
    print_nice_model(train_feature_map, raw_dump_file, nice_dump_file)

    print "test model..."
    test_model(test_libsvm_feature_file, model_file, score_file)

    print "generate test sore file..."
    generate_candidate_file(merged_benchmark_data, score_file, paste_score_file, sorted_paste_score_file)


def main():
    import sys
    train_start_date = sys.argv[1]
    train_end_date = sys.argv[2]
    test_start_date = sys.argv[3]
    test_end_date = sys.argv[4]

    xgb_exp(train_start_date, train_end_date, test_start_date, test_end_date)

if __name__ == "__main__":
    main()



