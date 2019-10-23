# -*- coding:UTF-8 -*-
"""

"""
import time

from data_load.get_one_day_wechat_segment import get_one_day_wechat_segment
from util.dateutil import DateUtil


def get_hist_wechat_segment(start_date, end_date):
    date_ls = DateUtil.get_every_date(start_date, end_date)

    for tmp_date in date_ls:
        start_time = time.time()
        get_one_day_wechat_segment(tmp_date)
        print("{0} wechat segment cost time: {1}".format(tmp_date, time.time() - start_time))


def main():
    import sys
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    get_hist_wechat_segment(start_date, end_date)


if __name__ == "__main__":
    main()
