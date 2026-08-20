"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render metrics and a concise interpretation section to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Quality is a deterministic 0-10 structural score based on answer presence, "
            "length, and citation coverage. Cost is aggregated from provider metadata; "
            "offline runs therefore report zero. Failure rate is 100% when a run records "
            "an error or produces no answer.",
            "",
            "Trace events are available in each returned `ResearchState.trace` for per-agent "
            "routing and latency inspection.",
        ]
    )
    return "\n".join(lines) + "\n"
