"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.agents import ResearcherAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, LabError
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.judge import LLMJudge
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
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


class _FailingSearchClient(SearchClient):
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        raise AgentExecutionError("Injected search provider outage")


def _run_failure_mode(query: str) -> ResearchState:
    settings = get_settings()
    llm_client = LLMClient(settings)
    workflow = MultiAgentWorkflow(
        settings=settings,
        researcher=ResearcherAgent(_FailingSearchClient(settings), llm_client),
    )
    return workflow.run(ResearchState(request=ResearchQuery(query=query)))


def _benchmark_queries(query: str | None, config_path: Path) -> list[str]:
    if query:
        return [query]
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    queries = data.get("benchmark", {}).get("queries", [])
    return [ResearchQuery(query=str(item)).query for item in queries]


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
    query: Annotated[str | None, typer.Option("--query", "-q", help="One research query")] = None,
    include_baseline: Annotated[
        bool, typer.Option(help="Run the API-backed single-agent baseline")
    ] = True,
    config: Annotated[Path, typer.Option(help="YAML file used when --query is omitted")] = Path(
        "configs/lab_default.yaml"
    ),
    failure_experiment: Annotated[
        bool, typer.Option(help="Include an injected search-provider failure")
    ] = True,
) -> None:
    """Benchmark workflows and write reports/benchmark_report.md."""

    _init()
    queries = _benchmark_queries(query, config)
    if not queries:
        raise typer.BadParameter("Benchmark config contains no queries")
    metrics: list[BenchmarkMetrics] = []
    judge = LLMJudge()
    store = LocalArtifactStore()

    def checkpoint() -> None:
        store.write_text("benchmark_report.md", render_markdown_report(metrics))

    for index, benchmark_query in enumerate(queries, start=1):
        if include_baseline:
            baseline_state, baseline_metrics = run_benchmark(
                f"single-agent/q{index}", benchmark_query, _run_baseline, judge
            )
            metrics.append(baseline_metrics)
            checkpoint()
            store.write_text(
                f"traces/single-agent-q{index}.json",
                baseline_state.model_dump_json(indent=2),
            )
        multi_state, multi_metrics = run_benchmark(
            f"multi-agent/q{index}", benchmark_query, _run_multi, judge
        )
        metrics.append(multi_metrics)
        checkpoint()
        store.write_text(f"traces/multi-agent-q{index}.json", multi_state.model_dump_json(indent=2))
    if failure_experiment:
        failure_query = queries[0]
        failure_state, failure_metrics = run_benchmark(
            "failure-injection", failure_query, _run_failure_mode, judge
        )
        metrics.append(failure_metrics)
        checkpoint()
        store.write_text("traces/failure-injection.json", failure_state.model_dump_json(indent=2))
    report = render_markdown_report(metrics)
    path = store.write_text("benchmark_report.md", report)
    console.print(Panel.fit(report, title=str(path)))


if __name__ == "__main__":
    app()
