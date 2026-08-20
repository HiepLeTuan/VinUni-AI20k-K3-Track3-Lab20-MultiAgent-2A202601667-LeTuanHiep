"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
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
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality,
        citation_coverage=citation_coverage,
        failure_rate=1.0 if state.errors or not answer else 0.0,
        notes=f"{len(state.sources)} sources; {word_count} answer words",
    )
    return state, metrics
