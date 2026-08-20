"""Evidence-aware critic agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, parse_json_object


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Check that the answer exists and references every collected source."""

        answer = state.final_answer or ""
        missing = [
            index for index in range(1, len(state.sources) + 1) if f"[{index}]" not in answer
        ]
        findings = "Citation check passed." if not missing else f"Missing citations: {missing}"
        unsupported_claims: list[str] = []
        metadata: dict[str, object] = {"missing_citations": missing, "llm_used": False}
        if not answer:
            findings = "Final answer is empty."
            state.errors.append(findings)
        elif self.llm_client.configured:
            sources = "\n".join(
                f"[{index}] {source.title}: {source.snippet}"
                for index, source in enumerate(state.sources, start=1)
            )
            response = self.llm_client.complete(
                "You are a strict fact checker. Compare each factual claim with the supplied "
                "source excerpts. Return JSON only with keys verdict (pass/fail), "
                "unsupported_claims (array of strings), and summary (string).",
                f"Answer:\n{answer}\n\nSources:\n{sources}",
            )
            try:
                review = parse_json_object(response.content)
                unsupported_claims = [str(item) for item in review.get("unsupported_claims", [])]
                findings = str(review.get("summary") or findings)
                if review.get("verdict") != "pass":
                    state.errors.append(f"Critic rejected answer: {findings}")
            except (ValueError, AttributeError):
                state.errors.append("Critic returned invalid JSON")
                findings = "Critic review could not be parsed."
            metadata.update(
                {
                    "llm_used": True,
                    "unsupported_claims": unsupported_claims,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                }
            )
        elif missing:
            state.errors.append(findings)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=findings,
                metadata=metadata,
            )
        )
        state.add_trace_event("critic", metadata)
        return state
