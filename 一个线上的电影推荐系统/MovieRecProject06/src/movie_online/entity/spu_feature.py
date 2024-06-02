import json
from typing import Optional, List


class SpuFeatureEntity(object):
    def __init__(self, record: dict):
        self.record: dict = record
        self.id: Optional[int] = record.get('id')
        self.title: Optional[str] = record.get('title')
        self.release_date: Optional[str] = record.get('release_date')
        _actors = record.get('actors')
        self.actors = _actors.split(",") if _actors else []
        # 类别
        self.categorys: List[str] = []
        self.sim_categorys: List[str] = []
        self._add_category(record)
        # 是否可见
        self.viewable: bool = record.get('viewable', 1) > 0

    def _add_category(self, record):
        ks = [
            'unknown', 'action', 'adventure', 'animation', 'children', 'comedy',
            'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
            'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war',
            'western'
        ]
        # TODO: 正常情况下，应该从数据库获取品类的相似品类列表
        sim_ks = {
            'action': ['adventure', 'animation'],
            'romance': ['horror', 'sci_fi']
        }
        _sim_categorys = set()
        for k in ks:
            v = record.get(k)
            if v == 1:
                # 添加这个类别
                self.categorys.append(k)
                # 相似类别(从数据库或者列表中查询出来)
                if k in sim_ks:
                    for _k in sim_ks[k]:
                        _sim_categorys.add(_k)
        self.sim_categorys.extend(list(_sim_categorys))

    def __iter__(self):
        return self.record

    def __str__(self):
        return json.dump(dict(self), ensure_ascii=False)
