# -*- coding: utf-8 -*-
from .base import ModelBaseRank


class BPRRank(ModelBaseRank):
    def __init__(self):
        super(BPRRank, self).__init__('bpr', 'bpr_rank')
