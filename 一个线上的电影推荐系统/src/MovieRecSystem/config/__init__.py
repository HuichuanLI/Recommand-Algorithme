import json
import os
from typing import List, Optional

import yaml


class AttributeNotFound(KeyError):
    pass


class ConfigNoFound(KeyError):
    pass


class RedisConfig(object):
    def __init__(self, name: str):
        arg_check(name, ['host', 'port', 'db'])
        self.cfg = __GLOBAL_CONFIGS__[name]
        append_default_config_values(
            self.cfg,
            {
                'password': None,
                'socket_timeout': 3000,
                'socket_connect_timeout': 3000
            }
        )

    def __str__(self):
        _cfg = self.cfg.copy()
        if 'password' in _cfg:
            _cfg['password'] = '***'
        else:
            _cfg['password'] = None
        return json.dumps(_cfg, ensure_ascii=False)


__GLOBAL_CONFIGS__ = {}
redis_config: Optional[RedisConfig] = None


def arg_check(config_name: str, arg_names: List[str]):
    if config_name not in __GLOBAL_CONFIGS__:
        raise ConfigNoFound(f"配置参数:'{config_name}'没有给定。")
    config = __GLOBAL_CONFIGS__[config_name]
    no_in_args = list(filter(lambda x: x not in config, arg_names))
    if len(no_in_args) != 0:
        raise AttributeNotFound(f'"{";".join(no_in_args)}" 在配置项:{config_name}中未发现有效配置项!')


def append_default_config_values(cfg: dict, mapping):
    for key in mapping:
        if key not in cfg:
            cfg[key] = mapping[key]


def __load_all_config():
    """
    以字典的形式返回所有配置信息
    :return:
    """
    _env = os.environ.get('MOVIE_REC_ENV')
    config_root_dir = os.environ.get("MOVIE_REC_CONFIG_DIR")
    if config_root_dir is None:
        # 当前文件所在的文件夹就是默认路径
        config_root_dir = os.path.dirname(os.path.abspath(__file__))
    if _env is None:
        with open(os.path.join(config_root_dir, 'config.yaml'), 'r', encoding='utf-8') as reader:
            _env = yaml.full_load(reader)['env'].strip()
    _config_path = os.path.join(config_root_dir, f'config_{_env}.yaml')
    if not os.path.exists(_config_path):
        raise ValueError(f"配置文件未找到:{_config_path}")
    with open(_config_path, 'r', encoding='utf-8') as reader:
        __GLOBAL_CONFIGS__.update(yaml.full_load(reader))


def init_config():
    """
    进行参数初始化操作
    :return:
    """
    from ..utils.logger_util import logger

    global redis_config
    # 加载yaml配置对象
    __load_all_config()
    # 解析配置基础特征信息
    redis_config = RedisConfig('redis')
    logger.info(f"redis Config is:{redis_config}")
