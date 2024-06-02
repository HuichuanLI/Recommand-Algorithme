# -*- coding: utf-8 -*-

from .base import ModelBaseRank


class LrRank(ModelBaseRank):
    def __init__(self):
        super(LrRank, self).__init__('lr', 'lr_rank')
