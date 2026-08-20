"""Evidence analysis agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Extract source-backed claims and state evidence limitations."""

        fallback_claims = [
            f"- Claim {index}: {source.snippet.strip()} [{index}]"
            for index, source in enumerate(state.sources, start=1)
        ]
        limitations = (
            "Evidence limitation: no usable external sources were available."
            if not state.sources
            else "Evidence limitation: snippets should be verified against the full linked sources."
        )
        metadata: dict[str, object] = {
            "claim_count": len(fallback_claims),
            "llm_used": False,
        }
        if self.llm_client.configured:
            response = self.llm_client.complete(
                "You are an evidence analyst. Extract the main claims, compare sources, identify "
                "agreement or conflict, and flag weak evidence. Every claim must retain [n] "
                "citations.",
                f"Question: {state.request.query}\n\nResearch notes:\n{state.research_notes}",
            )
            state.analysis_notes = response.content
            metadata.update(
                {
                    "llm_used": True,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                }
            )
        else:
            state.analysis_notes = (
                "Key findings:\n" + "\n".join(fallback_claims) + f"\n\n{limitations}"
            )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes,
                metadata=metadata,
            )
        )
        state.add_trace_event("analysis", metadata)
        return state
