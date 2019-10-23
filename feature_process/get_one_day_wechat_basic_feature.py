# -*- coding:UTF-8 -*-
"""

"""
import codecs
import json
from config.global_config import *
from feature_process.get_one_day_wechat_full_sentence_data import clear_sentence, stat_sentence
from util.dateutil import DateUtil
from collections import defaultdict
from data_load.load_order_data import load_multi_day_order
import datetime
from log.get_logger import G_LOG as log


HISTORY_ORDER_DELTA_DAY = 30
FUTURE_ORDER_DELTA_DAY = 5
HISTORY_WECHAT_RECORD_DELTA_DAY = 5

def load_hist_wechat_record_dict(date):
    wechat_dict = defaultdict(list)
    start_date = DateUtil.get_relative_delta_time_str(date, day=-HISTORY_WECHAT_RECORD_DELTA_DAY)
    end_date = DateUtil.get_relative_delta_time_str(date, -1)
    date_ls = sorted(DateUtil.get_every_date(start_date, end_date))  # 时间在后的聊天追加在后面
    wechat_full_sentence_data_dir = os.path.join(PROJECT_DATA_DIR, "raw_data", "wechat_full_sentence_data")

    wechat_full_sentence_data_file = os.path.join(wechat_full_sentence_data_dir,
                                                  "wechat_full_sentence_data_%s")
    for date in date_ls:
        log.info(date)
        wechat_full_sentence_data = wechat_full_sentence_data_file % date
        with codecs.open(wechat_full_sentence_data, 'r', 'utf-8') as fin:
            for line in fin:
                arr = line.strip().split("\t")
                if len(arr) != 5:
                    continue
                opp_id, student_chat_num, teacher_chat_num, all_chat_num, chat_content = arr
                student_chat_num, teacher_chat_num, all_chat_num = int(student_chat_num), int(teacher_chat_num), int(
                    all_chat_num)

                if opp_id not in wechat_dict:
                    wechat_dict[opp_id] = {"stat_info": [0, 0, 0], "chat_content": ""}
                wechat_dict[opp_id]["chat_content"] = wechat_dict[opp_id]["chat_content"] + chat_content
                wechat_dict[opp_id]["stat_info"] = [x + y for x, y in
                                                    zip([student_chat_num, teacher_chat_num, all_chat_num],
                                                        wechat_dict[opp_id]["stat_info"])]
    return wechat_dict


def judge_label(order_time, create_time):
    label = "0"
    if order_time:
        create_time = datetime.datetime.strptime(create_time, "%Y-%m-%d %H:%M:%S")
        order_time = datetime.datetime.strptime(order_time, "%Y-%m-%d %H:%M:%S")
        delta_day = (order_time - create_time).total_seconds() // (60.0 * 60 * 24)
        if 0 <= delta_day < FUTURE_ORDER_DELTA_DAY:
            label = "1"
        if delta_day < 0:  # 聊天之前今日已成单，去除该样本
            label = "-1"
    return label


def get_one_day_wechat_basic_feature(date):
    log.info("get %s wechat basic feature..." % date)
    wechat_basic_feature_dir = os.path.join(PROJECT_DATA_DIR, "feature_file", "wechat_basic_feature")
    aggregated_wechat_data_dir = os.path.join(PROJECT_DATA_DIR, "raw_data", "aggregated_wechat_data")

    wechat_basic_feature_data = os.path.join(wechat_basic_feature_dir, "wechat_basic_feature_%s" % date)
    aggregated_wechat_data = os.path.join(aggregated_wechat_data_dir, "aggregated_wechat_data_%s" % date)
    log.info("prepare hist wechat chat dict...")
    hist_wechat_chat_dict = load_hist_wechat_record_dict(date)

    log.info("prepare hist and future order dict...")
    hist_order_dict = load_multi_day_order(DateUtil.get_relative_delta_time_str(date, day=-HISTORY_ORDER_DELTA_DAY),
                                           DateUtil.get_relative_delta_time_str(date, day=-1))
    future_order_dict = load_multi_day_order(date,
                                             DateUtil.get_relative_delta_time_str(date, day=FUTURE_ORDER_DELTA_DAY))

    log.info("start 2 gen wechat basic feature...")
    with codecs.open(aggregated_wechat_data, "r", "utf-8") as fin, \
            codecs.open(wechat_basic_feature_data, "w", "utf-8") as fout:
        for line in fin:
            arr = line.strip().split("\t")
            if len(arr) != 2:
                continue

            opp_id = arr[0].strip()
            chat_ls = arr[1].strip()
            try:
                chat_ls = json.loads(chat_ls, encoding="utf-8")
            except Exception as e:
                log.info(e)
                continue
            if opp_id in hist_order_dict:
                continue

            order_time = future_order_dict.get(opp_id, None)  # 是否最近成单
            hist_wechat_chat = hist_wechat_chat_dict.get(opp_id, None)  # 是否有历史聊天记录

            # 剖析每个对话，学生说话则触发一次样本生成
            for idx, chat_dict in enumerate(chat_ls):
                send_type = chat_dict["send_type"]
                create_time = chat_dict["create_time"]
                accout = chat_dict["account"]

                if send_type == "0":  # 老师会话不做样本选取点
                    continue

                label = judge_label(order_time, create_time)
                if label == "-1":
                    continue

                cleared_chat_sentence = clear_sentence(chat_ls[:idx + 1])
                chat_stat_ls = stat_sentence(chat_ls[:idx + 1])
                hist_chat_stat_ls = [0, 0, 0]

                if hist_wechat_chat:  # 拼接历史聊天信息
                    hist_chat_stat_ls = hist_wechat_chat["stat_info"]
                    cleared_chat_sentence = hist_wechat_chat["chat_content"] + cleared_chat_sentence

                today_stat_str = "\t".join(map(str, chat_stat_ls))
                hist_stat_str = "\t".join(map(str, hist_chat_stat_ls))
                result = "\t".join(
                    [label, opp_id, accout, create_time, today_stat_str, hist_stat_str, cleared_chat_sentence])
                fout.write(result + "\n")
    log.info("finished, write feature to file : %s" % wechat_basic_feature_data)


def main():
    import sys
    # date = sys.argv[1]
    date = "20190503"
    get_one_day_wechat_basic_feature(date)


if __name__ == "__main__":
    main()
