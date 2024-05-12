from typing import Optional


class UserFeatureEntity(object):
    def __init__(self, user_id: Optional[int], location_id: int):
        self.id: Optional[int] = user_id
        self.location_id: int = location_id
        pass
