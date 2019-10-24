# -*- coding:UTF-8 -*-
"""

"""
from util.dateutil import DateUtil
from data_load.get_one_day_aggregated_wechat_data import get_one_day_aggregated_wechat_data


def get_his_aggregated_wechat_data(start_date, end_date):
    date_ls = DateUtil.get_every_date(start_date, end_date)

    for tmp_date in date_ls:
        print(tmp_date)
        get_one_day_aggregated_wechat_data(tmp_date)


def main():
    import sys
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    get_his_aggregated_wechat_data(start_date, end_date)


if __name__ == "__main__":
    main()
