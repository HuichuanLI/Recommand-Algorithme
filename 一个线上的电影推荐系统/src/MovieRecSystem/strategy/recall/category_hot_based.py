import random
from typing import List, Optional

from cachetools import cached, TTLCache

from ...entity.rec_item import RecItem
from ...utils.redis_util import RedisUtil


class CategoryHotBasedRecall(object):
    def __init__(self):
        self.REDIS_KEY = "rec:cate:hot_spu:"

    def get_candidates(self, category_id: Optional[int] = None, number_push: Optional[int] = None,
                       explain: str = "cate_hots") -> List[RecItem]:
        """
        提取热销商品作为推荐结果
        :param category_id: 品类id
        :param number_push: 待提取的数量
        :param explain: 名称字符串
        :return: 结果列表
        """
        if category_id is None:
            return []
        _items = self._get_hot_items(category_id, explain)
        random.shuffle(_items)  # 打乱顺序
        if number_push is not None and number_push > 0:
            _items = _items[:number_push]
        return _items

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def _get_hot_items(self, category_id: Optional[int] = None, explain: str = "cate_hots") -> List[RecItem]:
        """
        提取热销商品列表
        :param category_id: 品类id
        :param explain： 名称
        :return:
        """
        _key = f"{self.REDIS_KEY}{category_id}"
        spu_ids = RedisUtil.get_slist_int(_key)
        if spu_ids is None:
            return []
        result: List[RecItem] = []
        for _id in spu_ids:
            result.append(RecItem.get_recall_rec_item(_id, 1.0, explain, explain))
        return result
