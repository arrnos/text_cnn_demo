# -*- coding:UTF-8 -*-
"""
学员说话为观测点
"""
import os
import copy
import codecs
import json
import time
from datetime import datetime, timedelta
from collections import OrderedDict
from util.aggregation_data import aggregation_data
from util.dateutil import DateUtil

home_dir = os.environ["HOME"]
PAST_DAYS_LIMIT = 14
LABEL_DAYS_LIMIT = 5

def load_wechat_2_dict(start_date, end_date):
    date_ls = DateUtil.get_every_date(start_date, end_date)
    dict_wechat = OrderedDict()  #

    wechat_segment_data_dir = os.path.join(home_dir, "project_data/xgb",
                                           "raw_data/aggregated_wechat_segment_data")
    wechat_segment_data = os.path.join(wechat_segment_data_dir, "aggregated_wechat_segment_data_%s")

    print(date_ls)
    for tmp_date in date_ls:
        print(tmp_date)
        dict_wechat[tmp_date] = OrderedDict()
        with codecs.open(wechat_segment_data % tmp_date, "r", "utf-8") as fin:
            for line in fin:
                arr = line.strip().split("\t")
                if len(arr) != 2:
                    continue
                opp_id = arr[0].strip()
                chat_ls = arr[1].strip()
                try:
                    chat_ls = json.loads(chat_ls)
                except Exception as e:
                    print(e)
                    continue
                dict_wechat[tmp_date][opp_id] = chat_ls
    return dict_wechat


def update_wechat_dict(dict_wechat, del_date, add_date):
    del dict_wechat[del_date]

    wechat_segment_data_dir = os.path.join(home_dir, "project_data/xgb",
                                           "raw_data/aggregated_wechat_segment_data")
    wechat_segment_data = os.path.join(wechat_segment_data_dir, "aggregated_wechat_segment_data_%s" % add_date)

    dict_wechat[add_date] = OrderedDict()
    with codecs.open(wechat_segment_data, "r", "utf-8") as fin:
        for line in fin:
            arr = line.strip().split("\t")
            if len(arr) != 2:
                continue

            opp_id = arr[0].strip()
            chat_ls = arr[1].strip()
            try:
                chat_ls = json.loads(chat_ls)
            except Exception as e:
                print(2)
                continue
            dict_wechat[add_date][opp_id] = chat_ls
    return dict_wechat


def load_applied_order_2_dict(data_file):
    dict_data = dict()
    with codecs.open(data_file, "r", "utf-8") as fin:
        for line in fin:
            arr = line.strip().split("\t")
            if len(arr) != 21:
                continue

            opp_id = arr[3].strip()
            payment_time = arr[14].strip()

            if opp_id not in dict_data:
                dict_data[opp_id] = list()
            dict_data[opp_id].append(payment_time)

    for tmp_opp, tmp_ls in dict_data.items():
        dict_data[tmp_opp] = sorted(tmp_ls)
    return dict_data


def judge_label(dict_order, opp, create_time):
    label = "0"
    create_time = datetime.strptime(create_time, "%Y-%m-%d %H:%M:%S")
    if opp in dict_order:
        payment_time = min(dict_order[opp])
        payment_time = datetime.strptime(payment_time, "%Y-%m-%d %H:%M:%S")
        if (payment_time - create_time).total_seconds() / (60.0 * 60 * 24) < LABEL_DAYS_LIMIT:
            label = "1"
    return label


def get_one_day_wechat_tf_feature(date, dict_wechat):
    applied_order_data_dir = os.path.join(home_dir, "project_data/xgb", "raw_data/applied_order")
    middle_dir = os.path.join(home_dir, "project_data/xgb", "middle_file")
    label_date = (datetime.strptime(date, "%Y%m%d") + timedelta(days=LABEL_DAYS_LIMIT)).strftime("%Y%m%d")

    hist_applied_order_data = aggregation_data(applied_order_data_dir, middle_dir,
                                               "applied_order_",
                                               """{0} <= '%s'""" % date,
                                               file_pattern_filter="applied_order_")

    future_applied_order_data = aggregation_data(applied_order_data_dir, middle_dir,
                                                 "applied_order_",
                                                 """'%s'<= {0} <='%s'""" % (date, label_date),
                                                 file_pattern_filter="applied_order_")

    print("load history applied order to filter applied samples...")
    dict_hist_applied_order = load_applied_order_2_dict(hist_applied_order_data)

    print("load future applied order to judge sample label...")
    dict_future_applied_order = load_applied_order_2_dict(future_applied_order_data)

    benchmark_label_data = os.path.join(home_dir, "project_data/xgb",
                                        "benchmark_label_data", "benchmark_label_data_%s" % date)
    wechat_tf_feature_file = os.path.join(home_dir, "project_data/xgb",
                                          "feature_file/wechat_tf_feature", "wechat_tf_feature_%s" % date)

    past_tf_dict = OrderedDict()  # 过去N天会话的词频统计（不区分学员和咨询师）
    for tmp_date, tmp_dc in dict_wechat.items():
        if tmp_date == date:
            continue
        for tmp_opp, tmp_dc_ls in tmp_dc.items():
            if tmp_opp not in past_tf_dict:
                past_tf_dict[tmp_opp] = OrderedDict()
            for tmp in tmp_dc_ls:
                chat_record_seg = tmp["chat_record"]
                if chat_record_seg.strip() == "":
                    continue
                seg_words = chat_record_seg.strip().split(" ")
                for word in seg_words:
                    if word not in past_tf_dict[tmp_opp]:
                        past_tf_dict[tmp_opp][word] = 0
                    past_tf_dict[tmp_opp][word] += 1

    past_student_dialogue_dict = dict()  # 过去N天学员会话次数
    past_account_dialogue_dict = dict()   # 过去N天咨询师会话次数
    for tmp_date, tmp_dc in dict_wechat.items():
        if tmp_date == date:
            continue
        for tmp_opp, tmp_dc_ls in tmp_dc.items():
            if tmp_opp not in past_student_dialogue_dict:
                past_student_dialogue_dict[tmp_opp] = 0
            if tmp_opp not in past_account_dialogue_dict:
                past_account_dialogue_dict[tmp_opp] = 0
            for tmp in tmp_dc_ls:
                send_type = tmp["send_type"]
                if send_type == "1":
                    past_student_dialogue_dict[tmp_opp] += 1
                elif send_type == "0":
                    past_account_dialogue_dict[tmp_opp] += 1

    with codecs.open(benchmark_label_data, "w", "utf-8") as fout1, codecs.open(wechat_tf_feature_file, "w", "utf-8") as fout2:
        for tmp_opp, tmp_dc_ls in dict_wechat[date].items():

            dict_word_tf = OrderedDict()
            student_dialogue = 0
            account_dialogue = 0
            for tmp_dc in tmp_dc_ls:
                create_time = tmp_dc["create_time"]
                receive_time = tmp_dc["receive_time"]
                account = tmp_dc["account"]
                chat_record_seg = tmp_dc["chat_record"]
                send_type = tmp_dc["send_type"]

                if tmp_opp in dict_hist_applied_order and min(dict_hist_applied_order[tmp_opp]) < create_time:  # 历史已经成单（这里不考虑成多单的情况）
                    continue

                if chat_record_seg.strip() == "":
                    continue

                seg_words = chat_record_seg.strip().split(" ")
                for word in seg_words:  # 计算当天该机会对应会话的tf
                    if word not in dict_word_tf:
                        dict_word_tf[word] = 0
                    dict_word_tf[word] += 1

                if send_type == "0":  # 咨询师会话不作为样本
                    account_dialogue += 1
                    continue
                else:
                    student_dialogue += 1

                # print("judge label...")
                label = judge_label(dict_future_applied_order, tmp_opp, create_time)

                # 综合past days的该机会的会话词频
                if tmp_opp in past_tf_dict:
                    statistic_tf_dict = copy.deepcopy(past_tf_dict[tmp_opp])
                    for tmp_word, tmp_tf in dict_word_tf.items():
                        if tmp_word not in statistic_tf_dict:
                            statistic_tf_dict[tmp_word] = 0
                        statistic_tf_dict[tmp_word] += tmp_tf
                else:
                    statistic_tf_dict = copy.deepcopy(dict_word_tf)

                result_str = label + " " + "account_%s:1" % account + " "
                for tmp_word, tmp_tf in statistic_tf_dict.items():
                    result_str += "TF_" + tmp_word + ":" + str(tmp_tf) + " "
                fout1.write(str(student_dialogue + past_student_dialogue_dict.get(tmp_opp, 0)) + "\t" +
                            str(account_dialogue + past_account_dialogue_dict.get(tmp_opp, 0)) + "\t" +
                            label + "\t" + tmp_opp + "\t" +
                            account + "\t" + create_time + "\t" + receive_time + "\t" + chat_record_seg + "\n")
                fout2.write(result_str.strip() + "\n")

    delete_flag = True
    if delete_flag:
        cmd = "rm %s" % (" ".join([hist_applied_order_data, future_applied_order_data]))
        print(cmd)
        os.system(cmd)


def get_hist_wechat_tf_feature(start_date, end_date):
    date_ls = DateUtil.get_every_date(start_date, end_date)

    print("initial past n day wechat segment data...")
    s_date = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=PAST_DAYS_LIMIT)).strftime("%Y%m%d")
    t_date = start_date
    print("load past days wechat segment data...")
    dict_wechat = load_wechat_2_dict(s_date, t_date)

    print("extract tf feature...")
    for tmp_date in date_ls:
        print("extract %s tf feature..." % tmp_date)
        start_time = time.time()
        get_one_day_wechat_tf_feature(tmp_date, dict_wechat)
        print("extract {0} wechat tf feature cost time:{1}".format(tmp_date, time.time()-start_time))

        del_date = (datetime.strptime(tmp_date, "%Y%m%d") - timedelta(days=PAST_DAYS_LIMIT)).strftime("%Y%m%d")
        add_date = (datetime.strptime(tmp_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        if add_date <= end_date:
            print("update past days wechat segment data [del %s, add %s]..." % (del_date, add_date))
            dict_wechat = update_wechat_dict(dict_wechat, del_date, add_date)
            print("=======" * 3)


def main():
    import sys
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    get_hist_wechat_tf_feature(start_date, end_date)

if __name__ == "__main__":
    main()
