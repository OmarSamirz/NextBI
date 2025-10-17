from typing import TypedDict, Optional, List, Dict


class MultiAgentState(TypedDict):
    is_plot: bool
    user_query: str
    messages: List[Dict]
    response: Optional[str] = None
    explanation: Optional[str] = None
    manager_decision: Optional[str] = None
    td_agent_response: Optional[str] = None
    plot_agent_response: Optional[str] = None