# -*- coding: utf-8 -*-
from typing import List, Optional

from .base import BaseRecall
from ...entity import InnerRecItem
from ...entity.spu_feature import SpuFeatureEntity
from ...utils.graph_util import GraphUtil


class ActorRecall(BaseRecall):
    def __init__(self):
        super(ActorRecall, self).__init__('actor')

    def get_candidates(self,
                       spu: Optional[SpuFeatureEntity] = None, k: Optional[int] = None, recall_type: int = 0,
                       **kwargs
                       ) -> List[InnerRecItem]:
        if spu is None:
            return []
        actors = spu.actors  # 获取当前商品对应的演员名称列表
        if actors is None or len(actors) <= 0:
            return []

        # 1. 获取候选的商品id列表
        spu_ids = []
        for actor_name in actors:
            spu_ids.extend(self._fetch_rec_spu_ids_by_actor_name(actor_name, recall_type))

        # 2. 返回预测结果
        return self.fetch_random_candidates(
            list(set(spu_ids)), k,
            name=f"{self.name}_{recall_type}",
            filterd_spu_ids={spu.id}
        )

    @staticmethod
    def _fetch_rec_spu_ids_by_actor_name(actor_name, recall_type: int = 0):
        """
        recall_type == 0 --> 形式一/方向一：获取和actor_name演员同时参演的其他演员参演的电影列表
            match (a1:Actor {name:'actor_name'}) -[:参演]-> (m1:Movie)
            with m1
            match (a2:Actor) -[:参演]-> (m1)
            with a2
            match (a2) -[:参演]-> (m2:Movie)
            return distinct(m2.id) as movie_id
        recall_type == 1 --> 形式二/方式二：获取actor_name参演的其他电影
            match (a1:Actor {name:'actor_name'}) -[:参演]-> (m1:Movie)
            return distinct(m1.id) as movie_id
        :param actor_name:
        :return:
        """
        cypher = "match (a1:Actor {name:'" + actor_name + "'}) -[:参演]-> (m1:Movie)"
        if recall_type == 0:
            cypher = f"""
                {cypher}
                with m1
                match (a2:Actor) -[:参演]-> (m1)
                with a2
                match (a2) -[:参演]-> (m2:Movie)
                return distinct(m2.id) as movie_id
            """
        else:
            cypher = f"""
                {cypher}
                return distinct(m1.id) as movie_id
            """
        rs = GraphUtil.query(cypher)
        return [r.get('movie_id') for r in rs]
