import numpy as np
from sklearn.metrics import roc_auc_score


class ScoreUtil(object):

    def __init__(self):
        self.loss = []
        self.label = []
        self.prediction = []

    def reset(self):
        self.loss = []
        self.prediction = []
        self.label = []

    def add(self, loss, label, prediction):
        self.loss.append(loss)
        self.label.extend(label.flatten().tolist())
        self.prediction.extend(prediction.flatten().tolist())

    def get_score(self):
        return {
            "loss_num": len(self.loss),
            "loss": np.array(self.loss).mean(),
            "auc_num": len(self.label),
            "auc": roc_auc_score(self.label, self.prediction) if len(
                self.label) > 0 else 0,
            "pcoc": sum(self.prediction) / sum(self.label)
        }

    def to_string(self, result):
        return "loss: {}({}), auc: {}({}), pcoc: {}".format(
            result["loss"], result["loss_num"], result["auc"],
            result["auc_num"], result["pcoc"])


    def __str__(self):
        return self.to_string(self.get_score())