
from typing import TypedDict, List, Dict, Optional


class BaseState(TypedDict):
    user_query: str
    messages: List[Dict]
    response: Optional[str]