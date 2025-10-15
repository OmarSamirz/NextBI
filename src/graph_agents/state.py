from typing import TypedDict, Optional, List, Dict

class MultiAgentState(TypedDict):
    done: bool
    user_query: str
    is_plot: bool
    messages: List[Dict]
    explanation: Optional[str] = None
    response: Optional[str] = None
    manager_decision: Optional[str] = None
    td_agent_response: Optional[str] = None