import codecs


def load_dict(file_name, splitor="\t"):
    dict_return = {}
    with codecs.open(file_name, 'r', "utf-8") as fin:
        for line in fin:
            arr = line.split(splitor)
            dict_return[arr[1].strip()] = arr[0].strip()
    return dict_return


def load_dict_key_first(file_name, splitor="\t"):
    dict_return = {}
    with codecs.open(file_name, 'r', "utf-8") as fin:
        for line in fin:
            arr = line.split(splitor)
            dict_return[arr[0].strip()] = arr[1].strip()
    return dict_return
