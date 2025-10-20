"""Extended state used specifically by the MultiAgent orchestrator.

Contains fields used to coordinate between Manager, Teradata and Plot
agents such as whether a plot was produced, the manager explanation,
SQL snippets produced by the Teradata agent, and agent-specific
responses.
"""

from typing import Optional

from states import BaseState


class MultiAgentState(BaseState):
    is_plot: bool
    explanation: Optional[str]
    sql_queries: Optional[str]
    manager_decision: Optional[str]
    td_agent_response: Optional[str]
    plot_agent_response: Optional[str]