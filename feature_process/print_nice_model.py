import sys
import codecs
import re
from util import load_dict as load_dict


def print_nice_model(feature_index_file, model_dump_file, model_dump_nice_file):
    dict_feature_index = load_dict.load_dict(feature_index_file)
    pattern = re.compile("f[0-9]+")
    with codecs.open(model_dump_file, 'r', 'utf-8') as fin, codecs.open(model_dump_nice_file, 'w', 'utf-8') as fout:
        for line in fin:
            matcher = pattern.search(line)
            if matcher is None:
                fout.write(line)
                continue
            feature_index = "%s" % (int(matcher.group()[1:]))
            name = dict_feature_index[feature_index]
            line = line.replace(matcher.group(), name)
            fout.write(line)


def main():
    feature_index_file = sys.argv[1]
    model_dump_file = sys.argv[2]
    model_dump_nice_file = sys.argv[3]
    print_nice_model(feature_index_file, model_dump_file, model_dump_nice_file)

if __name__ == "__main__":
    main()
