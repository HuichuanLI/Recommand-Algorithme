"""
定义策略元数据对象
"""

from typing import List, Optional


class SceneMeta(object):
    def __init__(self, record: Optional[dict] = None):
        # 场景名称
        self.name = "测试"
        self.weight = 1.0
        # 召回策略名称字符串
        self.recalls: List[str] = [
            "hots", "location_news", "location_hots", "category_hots", "category_news",
            # "sim_category_hots", "sim_category_news",
            "u2i:user_cf:mf:item_cf", "i2i:views:mf:item_cf",
            "i2i:mf:item_cf",
            # "dssm",
            "fm",
            # "dummy", "fm",
            # "news", "hots",
            # "loc_news", "loc_hots"
            "u2i:user_cf:mf:item_cf",
            "user_views_i2i:mf:item_cf"
        ]
        # 过滤策略名称字符串
        self.filters: List[str] = [
            "user_views_filter:10", 'user_blacklist_filter', 'user_nolike_category_filter'
        ]
        # 排序策略名称字符串
        self.ranks: List[str] = [
            # "deepfm", "new_item2:1.6"
            "dummy", "recall_weighting"
        ]
        # 重排序策略名称字符串
        self.reranks: List[str] = []
        # 基于参数初始化
        self.__init_with_record(record)

    def __init_with_record(self, record):
        if record is None or len(record) == 0:
            return

        def _split(_v):
            if _v is None:
                return []
            return list(map(lambda t: t.strip(), _v.split(",")))

        _vs = record['name'].split(":", maxsplit=2)
        if len(_vs) == 2:
            self.name = _vs[0]
            self.weight = int(_vs[1])
        else:
            self.name = _vs[0]
        self.recalls = _split(record['recalls'])
        self.filters = _split(record['filters'])
        self.ranks = _split(record['ranks'])
        self.reranks = _split(record['reranks'])
