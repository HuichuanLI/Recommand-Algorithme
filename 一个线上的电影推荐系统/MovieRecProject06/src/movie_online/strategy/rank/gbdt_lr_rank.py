# -*- coding: utf-8 -*-
from .base import ModelBaseRank


class GBDTLrRank(ModelBaseRank):
    def __init__(self):
        super(GBDTLrRank, self).__init__('gbdt_lr', 'gbdt_lr_rank')
