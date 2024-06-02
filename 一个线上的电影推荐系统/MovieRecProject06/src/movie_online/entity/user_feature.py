from typing import Optional


class UserFeatureEntity(object):
    def __init__(self, record: dict):
        self.record: dict = record
        self.id: Optional[int] = record.get('user_id')
        self.location_id: str = record.get('zip_code', '')[:2]
        self.gender: str = record.get('gender', 'M')
