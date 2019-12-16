import os


def get_feature_num(feature_num_file):
    if not os.path.exists(feature_num_file):
        return 0
    else:
        fp = open(feature_num_file)
        for line in fp:
            item = line.strip().split("=")
            if item[0] == "feature_num":
                return int(item[1])
        return 0
