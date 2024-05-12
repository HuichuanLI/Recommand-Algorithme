from typing import List


class SceneMeta(object):
    def __init__(self):
        # 场景名称
        self.name = "name"
        # 召回策略名称字符串
        self.recalls: List[str] = ["hots", "new", "loc_hots", "loc_news"]
        # self.recalls: List[str] = ["new"]
        # 过滤策略名称字符串
        self.filters: List[str] = ["views", "blacklist"]
        # 排序策略名称字符串
        self.ranks: List[str] = ["random"]
        # 重排序策略名称字符串
        self.reranks: List[str] = []

