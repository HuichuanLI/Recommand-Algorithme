import random
from typing import List, Optional

from cachetools import cached, TTLCache

from ...entity.rec_item import RecItem
from ...utils.redis_util import RedisUtil


class LocationHotBasedRecall(object):
    def __init__(self):
        self.REDIS_KEY = "rec:loc:hot_spu:"

    def get_candidates(self, location_id: Optional[int] = None, number_push: Optional[int] = None) -> List[RecItem]:
        """
        提取热销商品作为推荐结果
        :param location_id: 地域id
        :param number_push: 待提取的数量
        :return: 结果列表
        """
        if location_id is None:
            return []
        _items = self._get_hot_items(location_id)
        random.shuffle(_items)  # 打乱顺序
        if number_push is not None and number_push > 0:
            _items = _items[:number_push]
        return _items

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def _get_hot_items(self, location_id: Optional[int] = None) -> List[RecItem]:
        """
        提取热销商品列表
        :param location_id: 地域id
        :return:
        """
        _key = f"{self.REDIS_KEY}{location_id}"
        spu_ids = RedisUtil.get_slist_int(_key)
        if spu_ids is None:
            return []
        result: List[RecItem] = []
        for _id in spu_ids:
            result.append(RecItem.get_recall_rec_item(_id, 1.0, "loc_hots", "loc_hots"))
        return result
