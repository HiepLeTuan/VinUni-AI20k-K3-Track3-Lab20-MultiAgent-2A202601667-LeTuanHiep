"""Search and research synthesis agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

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
        raw_notes = "\n".join(
            f"[{index}] {source.title}: {source.snippet.strip()}"
            for index, source in enumerate(unique, start=1)
        )
        metadata: dict[str, object] = {"source_count": len(unique), "llm_used": False}
        if raw_notes and self.llm_client.configured:
            response = self.llm_client.complete(
                "You are a research librarian. Summarize only the supplied source excerpts. "
                "Preserve [n] citations for every factual statement and flag evidence gaps.",
                f"Research question: {state.request.query}\n\nSources:\n{raw_notes}",
            )
            state.research_notes = response.content
            metadata.update(
                {
                    "llm_used": True,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                }
            )
        else:
            state.research_notes = raw_notes
        if not state.research_notes:
            state.research_notes = "No reliable sources were returned."
            state.errors.append("Research completed without usable sources")
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata=metadata,
            )
        )
        state.add_trace_event("research", {"source_count": len(unique)})
        return state
