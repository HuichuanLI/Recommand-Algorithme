import random
from typing import List, Optional

from cachetools import cached, TTLCache

from ...entity.rec_item import RecItem
from ...utils.redis_util import RedisUtil


class HotBasedRecall(object):
    def __init__(self):
        self.REDIS_KEY = "rec:hot_spu"

    def get_candidates(self, number_push: Optional[int] = None) -> List[RecItem]:
        """
        提取热销商品作为推荐结果
        :param number_push: 待提取的数量
        :return: 结果列表
        """
        _items = self._get_hot_items()
        random.shuffle(_items)  # 打乱顺序
        if number_push is not None and number_push > 0:
            _items = _items[:number_push]
        return _items

    @cached(cache=TTLCache(maxsize=1, ttl=600))
    def _get_hot_items(self) -> List[RecItem]:
        """
        提取热销商品列表
        :return:
        """
        spu_ids = RedisUtil.get_slist_int(self.REDIS_KEY)
        if spu_ids is None:
            return []
        result: List[RecItem] = []
        for _id in spu_ids:
            result.append(RecItem.get_recall_rec_item(_id, 1.0, "hots", "hots"))
        return result
