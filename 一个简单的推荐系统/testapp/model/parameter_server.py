from pathlib import Path

import numpy as np

class Singleton(type):
    _instance = {}
    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instance:
            Singleton._instance[cls] = type.__call__(cls, *args, **kwargs)
        return Singleton._instance[cls]


class ParameterSever(metaclass=Singleton):

    def __init__(self, hidden_dim):
        self.param_server = dict()
        self.hidden_dim = hidden_dim

    def pull(self, keys):
        values = []
        for key in keys:
            value = []
            for k in key:
                if k not in self.param_server:
                    self.param_server[k] = np.random.rand(self.hidden_dim)
                value.append(self.param_server[k])
            values.append(value)
        return np.array(values, dtype=np.float32)

    def push(self, keys, values):
        for i in range(len(keys)):
            for j in range(len(keys[i])):
                self.param_server[keys[i][j]] = values[i][j]

    def delete(self, keys):
        for i in range(len(keys)):
            for j in range(len(keys[i])):
                self.param_server.pop(keys[i][j])

    def save(self, file_path):
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "w") as f:
            for key, value in self.param_server.items():
                value_str = ",".join(["{:.8f}".format(val) for val in value])
                f.write("{}\t{}\n".format(key, value_str))