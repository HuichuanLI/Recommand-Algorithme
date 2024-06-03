# -*- coding: utf-8 -*-
import os
from typing import Dict, Any, Tuple, Optional

from . import BasePredictor
from .bpr import BPRLocalPredictor
from .fm import FMUserSideLocalPredictor, FMSpuSideLocalPredictor, FMLocalPredictor
from .gbdt_lr import GBDTLrLocalPredictor
from .lr import LrLocalPredictor
from .. import logger
from ..config import global_config


class ModelService(object):
    name_to_predictor_mapping = {
        # 内部保存的具体处理对象，格式是嵌套的{}
        # register_name: {
        #     model_version1: predictor1,
        #     model_version2: predictor2,
        # }
    }
    name_to_predictor_cls_mapping = {
        'fm_user': ("fm", FMUserSideLocalPredictor),  # 第一个表示模型存储的文件夹名称，当前必须为fm；第二个表示处理器类对象
        'fm_spu': ("fm", FMSpuSideLocalPredictor),
        'fm_rank': ("fm", FMLocalPredictor),
        'lr_rank': ("lr", LrLocalPredictor),
        "gbdt_lr_rank": ("gbdt_lr", GBDTLrLocalPredictor),
        "bpr_rank": ("bpr", BPRLocalPredictor),
    }

    @staticmethod
    def _get_model_version(model_name: str, model_version: Optional[str]) -> str:
        _model_version = model_version or 'last'
        _model_version = _model_version.lower()
        if _model_version in ['last', 'longest']:
            _dir = os.path.join(global_config.model_root_dir, model_name)
            _dirs = os.listdir(_dir)
            _dirs = sorted(_dirs)  # 字符串排序，默认为升序排列
            if _model_version == 'last':  # 最近的版本
                _model_version = _dirs[-1]
            elif _model_version == 'longest':  # 最老的一个版本
                _model_version = _dirs[0]
            else:
                raise ValueError(f"_model_version异常:{_model_version}")
        return _model_version

    @staticmethod
    def _get_predictor(model_register_name: str, model_version: Optional[str] = None) -> BasePredictor:
        if model_register_name not in ModelService.name_to_predictor_mapping:
            ModelService.name_to_predictor_mapping[model_register_name] = {}
        version2predictor = ModelService.name_to_predictor_mapping[model_register_name]

        # 1. 获取模型名称以及执行器cls对象
        model_name, model_predictor_cls = ModelService.name_to_predictor_cls_mapping[model_register_name]

        # 2. 获取明确的版本字符串
        model_version = ModelService._get_model_version(model_name, model_version)
        if model_version in version2predictor:
            return version2predictor[model_version]
        else:
            # TODO: 模型处理器创建这块儿的代码逻辑，存在多线程异常 --> 有可能会针对一个版本创建多个predictor预测器
            # 3. 创建predictor
            _dir = os.path.join(global_config.model_root_dir, model_name, model_version)
            predictor = model_predictor_cls(model_dir=_dir, model_version=model_version)
            # 4. 保存
            version2predictor[model_version] = predictor
            return predictor

    @staticmethod
    def fetch_predict_result(
            model_register_name: str, model_version: Optional[str] = None, **kwargs
    ) -> Optional[Tuple[Dict[str, Any], Any]]:
        """
        获取模型预测结果
        :param model_register_name: 给定模型注册名称字符串，必须全局唯一
        :param model_version: 给定期望的模型版本，可以给定名称的版本字符串，
            也可以给定None、last、longest，None默认为last，last表示获取最新版本，longest表示获取最老版本 --> 排序来决定
        :param kwargs: 进行模型预测操作时候对应的入参
        :return: 第一个返回的是元数据信息，第二个就是模型预测结果
        """
        if model_register_name not in ModelService.name_to_predictor_cls_mapping:
            logger.warn(f"当前不支持模型:{model_register_name}")
            return None
        predictor = ModelService._get_predictor(model_register_name, model_version)
        return predictor.predict(**kwargs)

    @staticmethod
    def list_models():
        result = {}
        for k, v in ModelService.name_to_predictor_mapping.items():
            result[k] = list(v.keys())
        return result
