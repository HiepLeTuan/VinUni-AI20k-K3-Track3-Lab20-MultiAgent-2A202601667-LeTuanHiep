"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Search, deduplicate sources, and create citation-ready notes."""

        results = self.search_client.search(state.request.query, state.request.max_sources)
        unique = []
        seen: set[str] = set()
        for source in results:
            key = source.url or source.title.casefold()
            if key not in seen and source.snippet.strip():
                unique.append(source)
                seen.add(key)
        state.sources = unique
        state.research_notes = "\n".join(
            f"[{index}] {source.title}: {source.snippet.strip()}"
            for index, source in enumerate(unique, start=1)
        )
        if not state.research_notes:
            state.research_notes = "No reliable sources were returned."
            state.errors.append("Research completed without usable sources")
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"source_count": len(unique)},
            )
        )
        state.add_trace_event("research", {"source_count": len(unique)})
        return state
