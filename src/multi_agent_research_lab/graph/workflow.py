"""Bounded LangGraph workflow for the research agents."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.workers: dict[str, BaseAgent] = {
            "researcher": researcher or ResearcherAgent(),
            "analyst": analyst or AnalystAgent(),
            "writer": writer or WriterAgent(),
            "critic": critic or CriticAgent(),
        }

    @staticmethod
    def _as_update(state: ResearchState) -> dict[str, Any]:
        return state.model_dump()

    def _node(self, agent: BaseAgent) -> Any:
        def execute(state: ResearchState) -> dict[str, Any]:
            with trace_span(agent.name, {"iteration": state.iteration}) as span:
                try:
                    result = agent.run(state)
                except Exception as exc:
                    state.errors.append(f"{agent.name} failed: {exc}")
                    state.add_trace_event(
                        "agent_error", {"agent": agent.name, "error": str(exc)}
                    )
                    result = state
                finally:
                    state.add_trace_event("span", span)
            return self._as_update(result)

        return execute

    def build(self) -> object:
        """Create a graph with conditional supervisor routing."""

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise AgentExecutionError("Install the 'llm' extra to use LangGraph") from exc

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self._node(self.supervisor))
        for name, agent in self.workers.items():
            graph.add_node(name, self._node(agent))
            graph.add_edge(name, "supervisor")
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state.route_history[-1],
            {**{name: name for name in self.workers}, "done": END},
        )
        return graph

    def run(self, state: ResearchState) -> ResearchState:
        """Compile and invoke the graph within the configured wall-clock timeout."""

        compiled = self.build().compile()  # type: ignore[attr-defined]
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-workflow")
        future = executor.submit(
            compiled.invoke,
            state,
            {"recursion_limit": self.settings.max_iterations * 2 + 2},
        )
        try:
            result = future.result(timeout=self.settings.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise AgentExecutionError(
                f"Workflow exceeded {self.settings.timeout_seconds}s timeout"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return result if isinstance(result, ResearchState) else ResearchState.model_validate(result)
