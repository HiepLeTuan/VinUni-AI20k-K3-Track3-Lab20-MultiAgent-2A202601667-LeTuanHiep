"""Benchmark report rendering."""

from statistics import mean

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render metrics and a concise interpretation section to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Query | Latency (s) | Cost (USD) | Quality | Citation cov. "
        "| Failure rate | Trace | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        trace_link = f"[open]({item.trace_url})" if item.trace_url else "n/a"
        query = item.query.replace("|", "\\|").replace("\n", " ")
        notes = item.notes.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item.run_name} | {query} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {trace_link} | {notes} |"
        )
    baseline = [item for item in metrics if item.run_name.startswith("single-agent/")]
    multi = [item for item in metrics if item.run_name.startswith("multi-agent/")]
    lines.extend(["", "## Summary", ""])
    if baseline and multi:
        baseline_quality = mean(item.quality_score or 0 for item in baseline)
        multi_quality = mean(item.quality_score or 0 for item in multi)
        lines.extend(
            [
                f"Across {len(multi)} queries, single-agent averaged "
                f"{mean(item.latency_seconds for item in baseline):.2f}s and "
                f"{baseline_quality:.1f}/10 quality; multi-agent averaged "
                f"{mean(item.latency_seconds for item in multi):.2f}s and "
                f"{multi_quality:.1f}/10 quality.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Quality is a deterministic 0-10 structural score based on answer presence, "
            "length, and citation coverage unless the row notes an LLM judge. Judged runs use "
            "a fixed relevance, evidence, synthesis, clarity, and uncertainty rubric. Cost "
            "includes generation and judge usage. Failure rate is 100% when a run records an "
            "error or produces no answer.",
            "",
            "Trace events are available in each returned `ResearchState.trace` for per-agent "
            "routing and latency inspection.",
            "",
            "## Failure Mode",
            "",
            "The `failure-injection` run deliberately makes the search provider unavailable. "
            "The supervisor records the worker error and routes to Writer so the workflow "
            "terminates with an explicit evidence limitation instead of hanging. The production "
            "fix is provider retry plus a curated offline fallback; `max_iterations` and provider "
            "timeouts bound the remaining failure path.",
        ]
    )
    return "\n".join(lines) + "\n"
