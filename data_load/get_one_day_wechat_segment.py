# -*- coding:UTF-8 -*-
"""

"""
import codecs
import json
import os
from config.global_config import *
import jieba

STOP_WORDS_FILE = os.path.join(PROJECT_DIR, "util/stopword.dic")


def load_stop_words():
    stop_words = set()
    with codecs.open(STOP_WORDS_FILE, "r", "utf-8") as fin:
        for line in fin:
            stop_words.add(line.strip())
    return stop_words


def seg_sentence(chat_data, stop_words):
    seg_words = jieba.cut(chat_data, cut_all=False)
    return [word for word in seg_words if u'\u4e00' <= word <= u'\u9fff' and len(word) > 1 and word not in stop_words]


def get_one_day_wechat_segment(date):
    aggregated_wechat_data_dir = os.path.join(PROJECT_DATA_DIR, "raw_data", "aggregated_wechat_data")
    wechat_segment_data_dir = os.path.join(PROJECT_DATA_DIR, "raw_data", "aggregated_wechat_segment_data")

    aggregated_wechat_data = os.path.join(aggregated_wechat_data_dir, "aggregated_wechat_data_%s" % date)
    wechat_segment_data = os.path.join(wechat_segment_data_dir, "aggregated_wechat_segment_data_%s" % date)

    print("load stop words...")
    stop_words = load_stop_words()

    print("segment wechat text...")
    with codecs.open(aggregated_wechat_data, "r", "utf-8") as fin, codecs.open(wechat_segment_data, "w",
                                                                               "utf-8") as fout:
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

            for tmp_dc in chat_ls:
                chat_record = tmp_dc["chat_record"]
                seg_words = seg_sentence(chat_record, stop_words)
                seg_str = " ".join(seg_words)
                tmp_dc["chat_record"] = seg_str
            json_str = json.dumps(chat_ls, encoding="utf-8", ensure_ascii=False, sort_keys=False)
            fout.write(opp_id + "\t" + json_str + "\n")


def main():
    import sys
    date = sys.argv[1]
    get_one_day_wechat_segment(date)


if __name__ == "__main__":
    main()
