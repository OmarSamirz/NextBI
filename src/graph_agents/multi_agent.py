from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph_agents.teradata_agent import TeradataAgent
from graph_agents.plot_agent import PlotAgent
from graph_agents.manager_agent import ManagerAgent
from graph_agents.state import MultiAgentState


class MultiAgent:

    def __init__(
        self,
        manager_agent: ManagerAgent,
        plot_agent: PlotAgent,
        teradata_agent: TeradataAgent,
    ) -> None:
        self.manager_agent = manager_agent
        self.plot_agent = plot_agent
        self.teradata_agent = teradata_agent

        self.memory = MemorySaver()
        self.graph = StateGraph(MultiAgentState)
        self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)

    def route_decision(self, state: MultiAgentState) -> str:
        if state.get("done"):
            return END
        
        decision = state.get("manager_decision", "")
        if "teradata" in decision:
            return "teradata"
        elif "plot" in decision:
            return "plot"
        else:
            return END

    def _build_graph(self) -> None:
        self.graph.add_node("manager", self.manager_agent)
        self.graph.add_node("teradata", self.teradata_agent)
        self.graph.add_node("plot", self.plot_agent)
        self.graph.add_node("route_decision", self.route_decision)

        self.graph.add_conditional_edges("manager", self.route_decision)
        
        self.graph.add_edge("teradata", "manager")
        self.graph.add_edge("plot", "manager")
        self.graph.set_entry_point("manager")

    async def run(self, user_query: str, thread_id: str = "default-thread"):
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await self.memory.aget(config)

        state = None
        if checkpoint and "messages" in checkpoint:
            state = checkpoint
            state["user_query"] = user_query
            state["done"] = False
        else:
            state = MultiAgentState(
                done=False,
                user_query=user_query,
                is_plot=False,
                messages=[]
            )

        final_state = await self.app.ainvoke(state, config=config)

        return final_state