"""Citation-aware research writer."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize the available evidence and include numbered references."""

        findings = state.analysis_notes or state.research_notes or "No evidence was available."
        references = [
            f"[{index}] {source.title}" + (f" - {source.url}" if source.url else "")
            for index, source in enumerate(state.sources, start=1)
        ]
        fallback_answer = (
            f"# {state.request.query}\n\n"
            f"## Findings\n\n{findings}\n\n"
            f"## Sources\n\n" + ("\n".join(references) or "No external sources available.")
        )
        metadata: dict[str, object] = {"citation_count": len(references), "llm_used": False}
        if self.llm_client.configured:
            response = self.llm_client.complete(
                "You are a technical research writer. Answer the question directly for the given "
                "audience. Use only supplied evidence, cite factual claims with [n], include a "
                "limitations section, and finish with the supplied source list.",
                f"Question: {state.request.query}\nAudience: {state.request.audience}\n\n"
                f"Analysis:\n{findings}\n\nSources:\n" + "\n".join(references),
            )
            state.final_answer = response.content
            metadata.update(
                {
                    "llm_used": True,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                }
            )
        else:
            state.final_answer = fallback_answer
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata=metadata,
            )
        )
        state.add_trace_event("write", metadata)
        return state
