# Design Template

## Problem

Hệ thống nhận một câu hỏi nghiên cứu, thu thập tối đa số nguồn được yêu cầu, tổng hợp
các luận điểm có dẫn nguồn, phân tích giới hạn bằng chứng và trả về báo cáo có thể trace.

## Why multi-agent?

Single-agent là baseline phù hợp cho câu hỏi ngắn nhưng trộn tìm kiếm, đánh giá bằng chứng
và viết trong một call nên khó kiểm tra lỗi. Multi-agent tách các trách nhiệm đó thành các
bước có state và trace riêng, phù hợp với câu hỏi cần nhiều nguồn hoặc cần audit.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Chọn bước kế tiếp và dừng workflow | Toàn bộ state | Route | Lặp vô hạn; chặn bằng `max_iterations` |
| Researcher | Tìm, lọc trùng và đánh số nguồn | Query, `max_sources` | Sources, research notes | Provider lỗi; ghi lỗi và fallback |
| Analyst | Rút claims và nêu giới hạn bằng chứng | Sources, research notes | Analysis notes | Thiếu nguồn; ghi rõ evidence gap |
| Writer | Tổng hợp báo cáo có citation | Query, analysis, sources | Final answer | Thiếu đầu vào; viết fallback rõ ràng |
| Critic | Kiểm tra answer và citation | Final answer, sources | Validation finding | Citation thiếu; ghi vào errors |

## Shared state

`request` giữ input đã validation; `sources`, `research_notes`, `analysis_notes` và
`final_answer` là các artifact handoff. `iteration` cùng `route_history` kiểm soát graph.
`agent_results` lưu output/usage có cấu trúc, `trace` phục vụ quan sát và `errors` điều khiển
fallback cũng như failure-rate benchmark.

## Routing policy

```text
START -> Supervisor -> Researcher -> Supervisor -> Analyst -> Supervisor
                    -> Writer -> Supervisor -> Critic -> Supervisor -> END
```

Supervisor chọn field còn thiếu theo thứ tự. Khi worker lỗi, route chuyển tới Writer để tạo
kết quả fallback; khi đủ answer và critic đã chạy, route là `done`.

## Guardrails

- Max iterations: `MAX_ITERATIONS`, mặc định 6.
- Timeout: `TIMEOUT_SECONDS`, mặc định 60 giây cho provider và toàn workflow.
- Retry: LLM retry tối đa 3 lần với exponential backoff.
- Fallback: Search dùng bộ nguồn offline khi không có Tavily key; lỗi worker được ghi vào state.
- Validation: Pydantic kiểm tra query/state; Critic kiểm tra answer và citation coverage.

## Benchmark plan

Chạy ba query trong `configs/lab_default.yaml`. Đo wall-clock latency, provider cost từ usage,
quality cấu trúc 0-10, citation coverage và failure rate. Kỳ vọng multi-agent chậm hơn baseline
nhưng có citation coverage và khả năng truy vết tốt hơn. Baseline cần `OPENAI_API_KEY`; workflow
multi-agent có thể benchmark offline.
