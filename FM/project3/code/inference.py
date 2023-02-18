# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import deepfm
import dssm

# load model
fmInf = deepfm.DEEPFM()
fmInf.load_model()
print(fmInf.imported)
test_ds = fmInf.init_dataset("../data/test", is_train=False)
test_ds = iter(test_ds)

for i in range(3):
    ds = next(test_ds)
    label = ds.pop("ctr")
    print(ds)
    res = fmInf.infer(ds)
    print(res)
