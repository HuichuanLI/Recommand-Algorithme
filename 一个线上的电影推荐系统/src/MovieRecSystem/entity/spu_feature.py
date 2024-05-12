from typing import List


class SpuFeatureEntity(object):
    def __init__(self):
        self.id: int = 0
        self.category_id: int = 0
        self.sim_category_ids: List[int] = []
        pass
