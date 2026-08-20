"""Benchmark runner for single-agent versus multi-agent research."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.judge import LLMJudge

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    judge: LLMJudge | None = None,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure reproducible workflow, output, citation, and failure metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    answer = state.final_answer or ""
    cited = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    expected = set(range(1, len(state.sources) + 1))
    citation_coverage = len(cited & expected) / len(expected) if expected else 0.0
    word_count = len(answer.split())
    quality = min(
        10.0,
        (4.0 if answer else 0.0) + min(3.0, word_count / 100) + 3 * citation_coverage,
    )
    total_cost = sum(
        float(result.metadata.get("cost_usd", 0) or 0) for result in state.agent_results
    )
    quality_note = "structural score"
    if judge is not None:
        try:
            judgment = judge.evaluate(query, state)
            quality = judgment.score
            quality_note = f"judge: {judgment.reason}"
            total_cost += judgment.cost_usd
        except Exception as exc:
            quality_note = f"judge unavailable ({type(exc).__name__}); structural score"
    trace_url = next(
        (
            str(event["payload"]["trace_url"])
            for event in reversed(state.trace)
            if event["name"] == "workflow_span" and event["payload"].get("trace_url")
        ),
        None,
    )
    metrics = BenchmarkMetrics(
        run_name=run_name,
        query=query,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality,
        citation_coverage=citation_coverage,
        failure_rate=1.0 if state.errors or not answer else 0.0,
        trace_url=trace_url,
        notes=f"{len(state.sources)} sources; {word_count} words; {quality_note}",
    )
    return state, metrics
