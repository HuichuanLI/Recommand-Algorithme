import json
import os
from typing import List, Optional

import yaml


class AttributeNotFound(KeyError):
    pass


class ConfigNoFound(KeyError):
    pass


class GlobalConfig(object):
    def __init__(self, name: str):
        arg_check(name, ['model_root_dir'])
        self.cfg = __GLOBAL_CONFIGS__[name]

    @property
    def model_root_dir(self):
        return self.cfg['model_root_dir']

    def __str__(self):
        _cfg = self.cfg.copy()
        return json.dumps(_cfg, ensure_ascii=False)


class RedisConfig(object):
    def __init__(self, name: str):
        arg_check(name, ['host', 'port', 'db'])
        self.cfg = __GLOBAL_CONFIGS__[name]
        # 检查配置是否存在，如果不存在，进行填充
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


class MysqlConfig(object):
    def __init__(self, name: str):
        arg_check(name, ['creator', 'host', 'user', 'password', 'database'])
        self.cfg = __GLOBAL_CONFIGS__[name]
        append_default_config_values(
            self.cfg,
            {
                'connect_timeout': 20,
                'read_timeout': 5,
                'write_timeout': 5
            }
        )

    def __str__(self):
        _cfg = self.cfg.copy()
        if 'password' in _cfg:
            _cfg['password'] = '***'
        else:
            _cfg['password'] = None
        return json.dumps(_cfg, ensure_ascii=False)


#
#
# class Neo4jConfig(object):
#     def __init__(self, name: str):
#         arg_check(name, ['profile', 'user', 'password'])
#         self.cfg = __GLOBAL_CONFIGS__[name]
#
#     @property
#     def profile(self):
#         return self.cfg['profile']
#
#     @property
#     def auth(self):
#         return self.cfg['user'], self.cfg['password']
#
#     def __str__(self):
#         _cfg = self.cfg.copy()
#         if 'password' in _cfg:
#             _cfg['password'] = '***'
#         else:
#             _cfg['password'] = None
#         return json.dumps(_cfg, ensure_ascii=False)


__GLOBAL_CONFIGS__ = {}
redis_config: Optional[RedisConfig] = None
mysql_config: Optional[MysqlConfig] = None
# neo4j_config: Optional[Neo4jConfig] = None
global_config: Optional[GlobalConfig] = None


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

    global redis_config, mysql_config, global_config, neo4j_config
    # 加载yaml配置对象
    __load_all_config()
    # 解析global配置基础特征信息
    global_config = GlobalConfig('global')
    logger.info(f"global Config is:{global_config}")
    # 解析redis配置基础特征信息
    redis_config = RedisConfig('redis')
    logger.info(f"redis Config is:{redis_config}")
    # # 解析mysql配置基础特征信息
    mysql_config = MysqlConfig('mysql')
    logger.info(f"mysql config is:{mysql_config}")
    # # 解析mysql配置基础特征信息
    # neo4j_config = Neo4jConfig('neo4j')
    # logger.info(f"neo4j config is:{neo4j_config}")
