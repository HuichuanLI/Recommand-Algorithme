import json
from typing import List


class RecExplain(object):
    def __init__(self, stage: str, explain: str, source: str):
        self.stage = stage
        self.explain = explain
        self.source = source

    def __iter__(self):
        yield from {
            'stage': self.stage,
            'explain': self.explain,
            'source': self.source
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)


class RecItem(object):
    def __init__(self, spu_id: int, score: float, explains: List[RecExplain]):
        self.spu_id: int = spu_id
        self.score: float = score
        self.explains: List[RecExplain] = explains

    def merge_explain(self, other_explains: List[RecExplain]):
        """
        合并策略
        :param other_explains: 待合并的策略
        :return:
        """
        if self.explains is None:
            self.explains = []
        if other_explains is not None:
            self.explains.extend(other_explains)

    def __iter__(self):
        yield from {
            'spu_id': self.spu_id,
            'score': self.score,
            'explains': [dict(v) for v in self.explains]
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    @staticmethod
    def get_recall_rec_item(spu_id: int, score: float, explain: str, source: str):
        explains = [RecExplain("recall", explain, source)]
        return RecItem(spu_id, score, explains)
