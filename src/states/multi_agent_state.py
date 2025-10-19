from typing import Optional

from states.base import BaseState


class MultiAgentState(BaseState):
    is_plot: bool
    explanation: Optional[str]
    manager_decision: Optional[str]
    td_agent_response: Optional[str]
    plot_agent_response: Optional[str]