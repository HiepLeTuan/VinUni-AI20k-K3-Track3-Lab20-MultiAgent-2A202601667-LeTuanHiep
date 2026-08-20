"""LLM-as-judge quality evaluation with a fixed research-answer rubric."""

from dataclasses import dataclass

from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, parse_json_object


@dataclass(frozen=True)
class JudgeResult:
    score: float
    reason: str
    cost_usd: float


class LLMJudge:
    """Score relevance, evidence, synthesis, clarity, and uncertainty from 0 to 10."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def evaluate(self, query: str, state: ResearchState) -> JudgeResult:
        sources = "\n".join(
            f"[{index}] {source.title}: {source.snippet}"
            for index, source in enumerate(state.sources, start=1)
        )
        response = self.llm_client.complete(
            "You are an impartial evaluator. Score the answer from 0 to 10 using: relevance "
            "0-2, evidence correctness 0-3, synthesis 0-2, clarity 0-2, uncertainty 0-1. "
            "Return JSON only with numeric total_score and concise reason. Do not reward citation "
            "markers unless the supplied excerpts support the associated claims.",
            f"Question:\n{query}\n\nAnswer:\n{state.final_answer}\n\nSources:\n{sources}",
        )
        try:
            payload = parse_json_object(response.content)
            score = max(0.0, min(10.0, float(payload["total_score"])))
            reason = str(payload["reason"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("LLM judge returned invalid JSON") from exc
        return JudgeResult(score=score, reason=reason, cost_usd=response.cost_usd or 0.0)
