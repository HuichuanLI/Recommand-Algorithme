# -*- coding: utf-8 -*-
# @Author : Zip
# @Time   : 2020/7/20|11:45
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import roc_auc_score


class AUCUtil(object):
    """Summary
    Attributes:
        ground_truth (list): Description
        loss (list): Description
        prediction (list): Description
    """

    def __init__(self):
        """Summary
        """
        self.reset()

    def add(self, loss, g=np.array([]), p=np.array([])):
        """Summary
        Args:
            loss (TYPE): Description
            g (TYPE): Description
            p (TYPE): Description
        """
        self.loss.append(loss)
        self.ground_truth += g.flatten().tolist()
        self.prediction += p.flatten().tolist()

    def calc(self):
        """Summary
        No Longer Returned:
            TYPE: Description
        Returns:
            TYPE: Description
        """
        return {
            "loss_num": len(self.loss),
            "loss": np.array(self.loss).mean(),
            "auc_num": len(self.ground_truth),
            "auc": roc_auc_score(self.ground_truth, self.prediction) if len(self.ground_truth) > 0 else 0,
            "pcoc": sum(self.prediction) / sum(self.ground_truth)
        }

    def calc_str(self):
        """Summary
        Returns:
            TYPE: Description
        """
        res = self.calc()
        return "loss: %f(%d), auc: %f(%d), pcoc: %f" % (
            res["loss"], res["loss_num"],
            res["auc"], res["auc_num"],
            res["pcoc"]
        )

    def reset(self):
        """Summary
        """
        self.loss = []
        self.ground_truth = []
        self.prediction = []
