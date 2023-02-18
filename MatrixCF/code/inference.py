# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import matrixcf

# load model
mcf = matrixcf.MCF()
mcf.load_model()
print(mcf.imported)
test_ds = mcf.init_dataset("../data/test", is_train=False)
test_ds = iter(test_ds)

for i in range(3):
    ds = next(test_ds)
    label = ds.pop("ctr")
    print(ds)
    res = mcf.infer(ds)
    print(res)
