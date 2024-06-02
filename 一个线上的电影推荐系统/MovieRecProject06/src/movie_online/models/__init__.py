# -*- coding: utf-8 -*-
from typing import Tuple, Dict, Any, Optional


class BasePredictor(object):
    def __init__(self, model_dir: str, model_version: str):
        self.model_dir = model_dir
        self.model_version = model_version

    def predict(self, **kwargs) -> Optional[Tuple[Dict[str, Any], Any]]:
        raise NotImplementedError("当前子类未实现模型预测方法!!!")
