# -*- coding:UTF-8 -*-
"""

"""
from feature_process.get_one_day_wechat_basic_feature import get_one_day_wechat_basic_feature
from util.dateutil import DateUtil
from log.get_logger import G_LOG as log


def get_hist_wechat_basic_feature(start_date, end_date):
    date_ls = DateUtil.get_every_date(start_date, end_date)
    for tmp_date in date_ls:
        log.info("extract %s basic feature..." % tmp_date)
        get_one_day_wechat_basic_feature(tmp_date)


def main():
    import sys
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    get_hist_wechat_basic_feature(start_date, end_date)


if __name__ == "__main__":
    main()
