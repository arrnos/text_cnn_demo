# -*- coding:UTF-8 -*-
"""

"""
import os
import codecs
import json
from config.global_config import *


def get_one_day_aggregated_wechat_data(date):
    """
    :param date:
    :return:
    """
    wechat_data_file = os.path.join("/home/yanxin/project_data/public_data/raw_data/raw_wechat_record_data",
                                    "weChatMsg_%s" % date, "000000_0")
    aggregated_wechat_data_file = os.path.join(PROJECT_DATA_DIR,"raw_data/aggregated_wechat_data", "aggregated_wechat_data_%s" % date)

    dict_chatContent = dict()
    with codecs.open(wechat_data_file, "r", "utf-8") as fin, codecs.open(aggregated_wechat_data_file, "w", "utf-8") as fout:
        for line in fin:
            arr = line.strip().split("\001")
            if len(arr) != 7:
                continue

            oid = arr[0].strip()
            account = arr[2].strip()
            send_type = arr[3].strip()
            receive_time = arr[4].strip()
            create_time = arr[5].strip()
            chat_record = arr[6].strip()

            if oid not in dict_chatContent:
                dict_chatContent[oid] = list()
            dict_chatContent[oid].append({"account": account,"send_type": send_type, "receive_time": receive_time, "create_time": create_time, "chat_record": chat_record})

        # dump json to file
        for tmp_oid, tmp_dc_ls in dict_chatContent.items():
            # sort by receive time in same student chat text
            tmp_dc_ls = sorted(tmp_dc_ls, key=lambda it: it["create_time"])
            json_str = json.dumps(tmp_dc_ls, encoding="utf-8", ensure_ascii=False, sort_keys=False)
            fout.write(tmp_oid + "\t" + json_str + "\n")


def main():
    import sys
    date = sys.argv[1]
    get_one_day_aggregated_wechat_data(date)


if __name__ == "__main__":
    main()
