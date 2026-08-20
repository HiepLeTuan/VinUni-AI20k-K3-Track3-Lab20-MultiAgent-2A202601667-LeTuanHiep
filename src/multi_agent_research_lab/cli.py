"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_baseline(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    response = LLMClient().complete(
        "You are a careful research assistant. State uncertainty and do not invent sources.",
        query,
    )
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "mode": "single-agent",
            },
        )
    )
    return state


def _run_multi(query: str) -> ResearchState:
    return MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=query)))


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run one real LLM call as the single-agent baseline."""

    _init()
    _parse_query(query)
    try:
        state = _run_baseline(query)
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="Baseline Error", style="red"))
        raise typer.Exit(code=2) from exc
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the bounded multi-agent workflow."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="Workflow Error", style="red"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    include_baseline: Annotated[
        bool, typer.Option(help="Run the API-backed single-agent baseline")
    ] = False,
) -> None:
    """Benchmark workflows and write reports/benchmark_report.md."""

    _init()
    _parse_query(query)
    metrics = []
    if include_baseline:
        _, baseline_metrics = run_benchmark("single-agent", query, _run_baseline)
        metrics.append(baseline_metrics)
    _, multi_metrics = run_benchmark("multi-agent", query, _run_multi)
    metrics.append(multi_metrics)
    report = render_markdown_report(metrics)
    path = LocalArtifactStore().write_text("benchmark_report.md", report)
    console.print(Panel.fit(report, title=str(path)))


if __name__ == "__main__":
    app()
