# Three backends with one client, 2026-11-24

One full run of `nim-three-backends.ipynb` (commit `d116d63`) in the Kaggle notebook of the same name, on 2x Tesla T4 and the hosted NVIDIA API.
The run chain stopped once before experiment 1 with no error, so it was resumed from that cell in the same kernel.

A first run (commit `fce11f3`) found two faults:

- The streaming check asked for one word, so the reply came in one chunk and the check failed on both backends.
- The mini Blueprint stopped on all 10 questions. The hosted topic guard returned a server error (a CUDA error on the service), and the pipeline failed closed. The notebook now tries a second candidate for each guard.

Real NIM containers need Docker and an NGC login, which Kaggle does not have. vLLM stands in for a self-hosted NIM.

## Experiment 1: traditional serving (A) against NIM-style serving (B)

Same model (`Qwen/Qwen2.5-1.5B-Instruct`, fp16), same GPU type, 96 output tokens for each request.
A is a small Python server around Transformers `generate`, one request at a time. B is vLLM.

| Concurrency | A tokens/s | B tokens/s | A p95 latency s | B p95 latency s | B TTFT p50 s |
|---:|---:|---:|---:|---:|---:|
| 1 | 30.7 | 66.6 | 3.2 | 1.4 | 0.03 |
| 8 | 31.2 | 475.6 | 24.7 | 1.6 | 0.09 |
| 16 | 31.3 | 858.5 | 49.2 | 1.8 | 0.11 |

| Built in | A | B |
|---|---|---|
| Health endpoint | yes | yes |
| Prometheus metrics | no | yes |
| Model list, OpenAI chat route, token streaming | no | yes |

## Experiment 2: the same client on B and on hosted NIM (C)

C is `nvidia/nemotron-3.5-bolt-e4b`. Only the base URL, the model name and the key change. No code line changes.

| Check | B | C |
|---|---|---|
| List models, chat, streaming, max_tokens, stop sequence, JSON reply, tool call | pass | pass |
| Error format for a wrong model name | pass (JSON with `error`) | fail (plain text 404) |

## Experiment 3: hybrid routing for data control

A local pattern detector sends prompts with personal data to B, and the rest to C. 20 prompts, 10 with personal data (all fake).

| Measure | Result |
|---|---|
| Detector precision and recall | 0.89 and 0.80 |
| Personal-data prompts that reached the cloud | 2 of 10, both a name with a street address, which the patterns cannot see |
| Forced send of a detected prompt to the cloud | blocked by the egress check |
| Cloud unreachable | 3 of 3 general prompts answered by B |
| B stopped | 9 of 9 detected prompts failed closed, 0 reached the cloud |

## Experiment 4: a mini Blueprint from NIM parts

| Part | First choice | Used |
|---|---|---|
| Topic guard | `llama-3.1-nemoguard-8b-topic-control`: server error | `nemotron-3.5-bolt-e4b` with the same policy prompt |
| Safety guard | `nemotron-3.5-content-safety`: 404 for this account | `llama-3.1-nemotron-safety-guard-8b-v3` |
| Embedding | `nemotron-3.5-embed-1b` | the same |

| Outcome | Questions |
|---|---|
| In-domain: answered with a correct citation | 6 of 6 (right document in the top 2: 6 of 6) |
| Off-topic: stopped by the topic guard | 3 of 3 |
| Unsafe: stopped by the topic guard | 1 of 1 |

Median seconds: topic guard 0.26, embedding 0.15, LLM 0.43, safety guard 1.12, total 2.03.
A swap to `nemotron-3.5-embed-8b` needed only a new name. The vector size went from 2,048 to 4,096, and all 6 answers kept their citation.

## Experiment 5: compute-aware selection

A small function picked the right profile in the 3 known cases (7B on a T4).
Its KV cache sizes were off by -0.1%, +1.0% and +9.1% from the sizes that vLLM reported.

## What the numbers show

- **Batching is the gain.** At 16 requests at once, vLLM gave 27 times the tokens/s and 27 times less p95 latency. The simple server only made a queue.
- **One API makes backends swappable.** The client moved from self-hosted to hosted with no code change. Error bodies still differ, so parse them with care.
- **Data control needs more than one layer.** The router, the egress check and fail-closed behaviour held. The detector is the weak part: recall was 0.80.
- **Guards can be the slowest stage.** The safety guard took more than half of the total time.
- **Plan for a model that is listed but not available.** Two hosted guards failed for this account. The pipeline failed closed, which is the safe way to fail.

## Limits

A is a plain server on purpose. A better traditional server could batch too.
The unsafe question stopped at the topic guard, so the safety guard was not tested inside the pipeline.
10 questions and 20 prompts are small samples.
B and C run different models on different hardware, so their latency is not compared.
