"""Provider-neutral spans stored in workflow state as JSON-serializable events."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings


@lru_cache(maxsize=2)
def _langsmith_client(api_key: str | None) -> Any | None:
    if not api_key:
        return None
    try:
        from langsmith import Client
    except ImportError:
        return None
    return Client(api_key=api_key)


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> Iterator[dict[str, Any]]:
    """Capture timing, attributes, status, and errors for one operation."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    active_settings = settings or get_settings()
    client = _langsmith_client(active_settings.langsmith_api_key)
    run = None
    try:
        if client is None:
            yield span
        else:
            from langsmith import trace
            from langsmith.run_helpers import tracing_context

            with (
                tracing_context(
                    project_name=active_settings.langsmith_project,
                    client=client,
                    enabled=True,
                ),
                trace(
                    name,
                    inputs=attributes or {},
                    project_name=active_settings.langsmith_project,
                    client=client,
                    tags=["multi-agent-research-lab"],
                ) as run,
            ):
                span["provider"] = "langsmith"
                span["run_id"] = str(run.id)
                yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span.setdefault("status", "ok")
        span["duration_seconds"] = perf_counter() - started
        if client is not None and run is not None:
            try:
                span["trace_url"] = client.get_run_url(
                    run=run, project_name=active_settings.langsmith_project
                )
            except Exception as exc:
                span["trace_url_error"] = str(exc)


def flush_traces(settings: Settings | None = None) -> None:
    """Flush buffered LangSmith runs before a CLI process exits."""

    active_settings = settings or get_settings()
    client = _langsmith_client(active_settings.langsmith_api_key)
    if client is not None:
        client.flush(timeout=10)
