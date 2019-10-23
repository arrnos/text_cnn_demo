# -*- coding:UTF-8 -*-
"""

"""
import codecs
import json
from config.global_config import *


def clear_sentence(chat_data):
    sentence_ls = []
    for chat_dict in chat_data:
        chat_record = chat_dict["chat_record"]
        send_type = chat_dict["send_type"]
        chinese_sentence = "".join([word for word in chat_record if u'\u4e00' <= word <= u'\u9fff'])
        # 超长语句截断
        if len(chinese_sentence) > 70:
            chinese_sentence = chinese_sentence[:20] + chinese_sentence[-20:]
        sentence_ls.append(chinese_sentence)
    return "".join(sentence_ls)


def get_one_day_wechat_full_sentence(date):
    aggregated_wechat_data_dir = os.path.join(PROJECT_DATA_DIR, "raw_data", "aggregated_wechat_data")
    wechat_full_sentence_data_dir = os.path.join(PROJECT_DATA_DIR, "raw_data",
                                                 "aggregated_wechat_full_sentence_data")

    aggregated_wechat_data = os.path.join(aggregated_wechat_data_dir, "aggregated_wechat_data_%s" % date)
    wechat_full_sentence_data = os.path.join(wechat_full_sentence_data_dir,
                                             "aggregated_wechat_full_sentence_data_%s" % date)

    print("segment wechat text...")
    with codecs.open(aggregated_wechat_data, "r", "utf-8") as fin, \
            codecs.open(wechat_full_sentence_data, "w", "utf-8") as fout:
        for line in fin:
            arr = line.strip().split("\t")
            if len(arr) != 2:
                continue

            opp_id = arr[0].strip()
            chat_ls = arr[1].strip()
            try:
                chat_ls = json.loads(chat_ls, encoding="utf-8")
            except Exception as e:
                print(e)
                continue

            cleared_chat_sentence = clear_sentence(chat_ls)
            chat_stat_ls = stat_sentence(chat_ls)

            stat_str = "\t".join(map(str, chat_stat_ls))
            fout.write(opp_id + "\t" + stat_str + "\t" + cleared_chat_sentence + "\n")


def stat_sentence(chat_ls):
    student_chat_num = 0
    teacher_chat_num = 0
    for chat_dict in chat_ls:
        type = chat_dict["send_type"]
        if type == "0":
            teacher_chat_num += 1
        else:
            student_chat_num += 1
    all_chat_num = student_chat_num + teacher_chat_num
    return [student_chat_num, teacher_chat_num, all_chat_num]


def main():
    import sys
    date = sys.argv[1]
    # date = "20190405"
    get_one_day_wechat_full_sentence(date)


if __name__ == "__main__":
    main()
