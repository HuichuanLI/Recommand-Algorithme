import random
from typing import List

from MovieRecSystem.entity.rec_item import RecItem


class RandomRank(object):
    def __init__(self):
        pass

    def rank_items(self, items: List[RecItem]) -> List[RecItem]:
        if items is None:
            return []
        random.shuffle(items)
        return items
