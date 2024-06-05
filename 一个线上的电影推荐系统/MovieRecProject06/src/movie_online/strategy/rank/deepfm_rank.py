# -*- coding: utf-8 -*-

from .base import ModelBaseRank


class DeepFMRank(ModelBaseRank):
    def __init__(self):
        super(DeepFMRank, self).__init__("deepfm", "deepfm_rank")
