# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| multi-agent | 0.86 | 0.0000 | 8.0 | 100% | 0% | 3 sources; 95 answer words |

The single-agent row is intentionally absent because this run had no authorized OpenAI
provider call. Re-run with `--include-baseline` after configuring `OPENAI_API_KEY` to produce
the direct comparison; the command records provider latency and token metadata.

## Interpretation

Quality is a deterministic 0-10 structural score based on answer presence, length, and citation coverage. Cost is aggregated from provider metadata; offline runs therefore report zero. Failure rate is 100% when a run records an error or produces no answer.

Trace events are available in each returned `ResearchState.trace` for per-agent routing and latency inspection.
