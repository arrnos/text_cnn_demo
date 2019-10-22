#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python2.7
@author: zhangmeng
@file: load_order_data.py
@time: 2019/08/01
"""
import codecs

from config.global_config import *
from util.dateutil import DateUtil


def load_one_day_order(date):
    print("load %s order data.." % date)
    order_raw_file = os.path.join(ORDER_PATH, "raw_order_info_%s" % date)
    rs_dict = {}
    with codecs.open(order_raw_file) as fin:
        for line in fin:
            line_array = line.strip().split("\t")
            opp_id = line_array[3]
            payment_time = line_array[11]
            rs_dict[opp_id] = payment_time
    return rs_dict


def load_multi_day_order(start_date, end_date):
    multi_day_dict = {}
    date_ls = DateUtil.get_every_date(start_date, end_date)
    sorted(date_ls)  # 降序排列，如果出现多次，用最小的订单时间覆盖
    for date in date_ls:
        one_day_dict = load_one_day_order(date)
        multi_day_dict.update(one_day_dict)
    return multi_day_dict


if __name__ == '__main__':
    load_multi_day_order("20190411", "20190428")
