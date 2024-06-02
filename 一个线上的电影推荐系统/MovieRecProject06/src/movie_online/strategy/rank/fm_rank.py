# -*- coding: utf-8 -*-

from .base import ModelBaseRank


class FMRank(ModelBaseRank):
    def __init__(self):
        super(FMRank, self).__init__("fm", "fm_rank")
