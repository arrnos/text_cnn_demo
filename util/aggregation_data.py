# -*- coding:UTF-8 -*-
"""

"""
import os


class AggregationException(Exception):
    pass


def aggregation_data(input_dir, output_dir, out_put_prefix, time_expr, file_pattern_filter=None):
    file_name_list = os.listdir(input_dir)
    if file_pattern_filter:
        filter_name_list = [i for i in file_name_list if eval(time_expr.format("'{0}'".format(i.split("_")[-1])))
                            and i.startswith(file_pattern_filter)]
    else:
        filter_name_list = [i for i in file_name_list if eval(time_expr.format("'{0}'".format(i.split("_")[-1])))]
    filter_name_list.sort()
    if not filter_name_list:
        raise AggregationException("can't aggregate to a legal file, please check the path and condition")
    output_file_name = out_put_prefix + filter_name_list[0].split("_")[-1] + "_" + filter_name_list[-1].split("_")[-1]
    output_file_name = os.path.join(output_dir, output_file_name)
    cmd = "cat %s > %s" % (
        " ".join([os.path.join(input_dir, file_name) for file_name in filter_name_list]), output_file_name)
    # print("Execute cmd :\n" + cmd)
    os.system(cmd)
    return output_file_name
