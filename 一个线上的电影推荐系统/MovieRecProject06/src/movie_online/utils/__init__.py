class SimpleMapping(object):
    """
    定义的一个简单的映射表
    """

    def __init__(self, path):
        self.mapping = {}
        with open(path, 'r', encoding='utf-8') as reader:
            for line in reader:
                arr = line.strip().split("\t")
                self.mapping[arr[0]] = int(arr[1])

    def get(self, key) -> int:
        value = self.mapping.get(str(key))
        if value is None:
            return self.mapping['unk']
        else:
            return value

    def in_mapping(self, key):
        return str(key) in self.mapping

    def size(self) -> int:
        return len(self.mapping)
