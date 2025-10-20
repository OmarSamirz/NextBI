from typing import Optional

from states import BaseState


class MultiAgentState(BaseState):
    is_plot: bool
    explanation: Optional[str]
    sql_queries: Optional[str]
    manager_decision: Optional[str]
    td_agent_response: Optional[str]
    plot_agent_response: Optional[str]