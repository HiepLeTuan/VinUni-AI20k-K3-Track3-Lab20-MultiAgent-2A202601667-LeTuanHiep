# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single-agent | 8.93 | 0.0003 | 7.0 | 0% | 0% | 0 sources; 364 answer words |
| multi-agent | 1.12 | 0.0000 | 8.0 | 100% | 0% | 3 sources; 95 answer words |

## Interpretation

Quality is a deterministic 0-10 structural score based on answer presence, length, and citation coverage. Cost is aggregated from provider metadata; offline runs therefore report zero. Failure rate is 100% when a run records an error or produces no answer.

Trace events are available in each returned `ResearchState.trace` for per-agent routing and latency inspection.
