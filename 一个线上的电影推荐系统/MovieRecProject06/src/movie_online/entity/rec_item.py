import json
from typing import List, Optional


class RecExplain(object):
    """
    定义推荐过程中的临时执行explain结果值
    """

    def __init__(self, stage: str, explain: str, source: str):
        super(RecExplain, self).__init__()
        self.stage = stage  # 阶段, eg: recall、filter、rank
        self.explain = explain  # 策略名称: eg: news、fm、deepfm
        self.source = source  # 对应策略的评分: news、fm:0.3、deepfm:0.9

    def to_dict(self):
        return {
            'stage': self.stage,
            'explain': self.explain,
            'source': self.source
        }

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


class InnerRecItem(object):
    """
    推荐系统内部使用的待推荐商品对象
    """

    def __init__(self, spu_id, score: float, explains: List[RecExplain]):
        """
        对象构建
        :param spu_id: 商品id
        :param score: 当前商品的评分
        """
        super(InnerRecItem, self).__init__()
        self.spu_id = spu_id
        self.score = score
        self.explains: List[RecExplain] = explains
        self.is_effect = True  # 当前商品是否有效

    def to_dict(self):
        return {
            'spu_id': self.spu_id,
            'score': self.score,
            'is_effect': self.is_effect,
            'explains': [exp.to_dict() for exp in self.explains]
        }

    def recall_explains(self) -> int:
        return len([t for t in self.explains if t.stage == 'recall'])

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def merge_explain(self, explains: List[RecExplain], score: Optional[float] = None):
        """
        策略的合并
        :param explains:
        :param score:
        :return:
        """
        if score is not None:
            self.score = max(self.score, score)
        if self.explains is None:
            self.explains = []
        if explains is not None:
            self.explains.extend(explains)

    def add_filter(self, explain: str):
        self.explains.append(RecExplain("filter", explain, explain))
        self.is_effect = False

    def add_rank(self, explain: str, source: str):
        self.explains.append(RecExplain("rank", explain, source))

    @staticmethod
    def get_recall_rec_item(spu_id, explain: str, score: float = 1.0, source: Optional[str] = None):
        explains = [RecExplain("recall", explain, source or f"{explain}:{score}")]
        return InnerRecItem(spu_id, score, explains)


class RecItem(object):
    """
    最终推荐返回的商品对象
    """

    def __init__(self, spu_id, score):
        self.spu_id = spu_id
        self.score = score

    @staticmethod
    def build_with_inner_rec_item(item: InnerRecItem) -> 'RecItem':
        return RecItem(spu_id=item.spu_id, score=item.score)

    def to_dict(self):
        return {
            'spu_id': self.spu_id,
            'score': self.score
        }
