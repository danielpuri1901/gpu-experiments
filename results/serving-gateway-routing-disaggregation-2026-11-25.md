# Serving: modes, routing, failover, disaggregation and scaling, 2026-11-25

One full run of `serving-gateway-routing-disaggregation.ipynb` (commit `8259c12`) in the Kaggle notebook of the same name (Version 1, T4 x2, 26 minutes). All 8 parts passed.
Parts A to E are measured with vLLM 0.34.0 and `Qwen/Qwen2.5-1.5B-Instruct` in fp16, one replica per T4. Part F is a simulation fed with measured times. Part G is a model of the two Triton batchers in Python, not a Triton server.

## Batch, real-time, streaming

200 requests. TTFT is the time to the first token. E2E is the time to the full reply.

| Mode | Tokens/s | TTFT p50 | E2E p50 |
|---|---:|---:|---:|
| Offline batch, 200 at once | 1,870 | | |
| Real-time, full reply, 8 at once | 483 | | 2,121 ms |
| Real-time, full reply, 64 at once | 1,757 | | 4,133 ms |
| Streaming, 8 at once | 466 | 85 ms | 2,196 ms |

Streaming does not make the reply faster. The user sees the first word after 85 ms and not after 2,121 ms.
The client was not the bottleneck: its TTFT was 6.4 ms above the server's own.

## Routing over two replicas

96 requests per policy: multi-turn chats in two repeats. The prefix cache holds earlier turns, so the replica that served a chat before can skip its prefill.

| Policy | TTFT p50 | TTFT p95 | Cache hit rate |
|---|---:|---:|---|
| Round robin | 349 ms | 2,557 ms | 0.804 [0.803, 0.806] |
| Least outstanding requests | 181 ms | 1,517 ms | 0.856 [0.854, 0.857] |
| Prefix-aware | 176 ms | 1,395 ms | 0.897 [0.896, 0.898] |

On turns 2 to 4 (72 pairs), prefix-aware was 545 ms faster than round robin [363, 744], and least outstanding was 441 ms faster [217, 667].
Prefix-aware was not clearly better than least outstanding on TTFT. It did not overload one replica: each got 50% of the requests.

## Failover

Replica 1 is killed in the middle of 120 requests.

| Setting | Failed | Retries |
|---|---|---:|
| No retry, no health check | 44 of 120, 36.7% [28.6, 45.6] | 0 |
| Retry on the other replica | 0 of 120 [0, 3.1] | 72 |
| Retry and health check | 0 of 120 [0, 3.1] | 3 |

Retries remove the errors (exact McNemar p = 1e-13). The health check removes 69 of the 72 wasted first tries.
With replica 1 dead, a hosted NVIDIA model answered 3 of 3 requests through the same client.

## Prefill and decode on the same GPU, and apart

8 chat streams of 160 tokens run for 26 s. 16 long prompts of 3,000 tokens arrive 1.5 s apart. ITL is the gap between two tokens of a chat.
Chats finished comes from the number of gaps (about 157 per chat), because the notebook did not record the TTFT of the chats.

| Setup | ITL p50 | ITL p99 | Chats finished | Long-prompt TTFT p50 |
|---|---:|---:|---:|---:|
| Chats only (floor) | 16.6 ms | 33 ms | 80 | |
| Mixed, budget 8,192 | 16.8 ms | 4,860 ms | 8 | 6,427 ms |
| Mixed, budget 512 (chunked prefill) | 17.1 ms | 501 ms | 16 | 4,618 ms |
| Roles split, no KV transfer | 16.9 ms | 34 ms | 80 | 6,195 ms |
| Disaggregated, NIXL KV transfer | 16.6 ms | 35 ms | 32 | 6,262 ms |

- One long prefill blocks every chat on the GPU for seconds. The median ITL hides it. The p99 is 145 times the floor.
- Chunked prefill cuts the p99 by 10 times, and it raises the share of gaps over 100 ms from 7% to 26%.
- Real disaggregated prefill and decode ran on two T4s with no NVLink. GPU 0 prefills, NIXL moves the KV cache, GPU 1 decodes. The reply matched the producer's own reply, and `vllm:nixl_xfer_time_seconds` counted the transfers.
- Disaggregation keeps ITL at the floor, but it moves the wait to the first token. Only 32 chats finished against 80, because each chat's own prefill waits behind the long prompts on GPU 0. The "roles split" row avoids this only because the chats never touch the prefill GPU.
- The long prompts arrive faster than one T4 can prefill them, so their TTFT is mostly queue time in every setup (an inference from the 1.5 s gap, not measured directly).
- The KV connector rows pass through a small proxy, which adds one hop.

## Cold start

| State | First answer | Download | Weight load |
|---|---:|---:|---:|
| No local weights | 133 s | 25 s | 31 s |
| Cached weights, page cache dropped | 58 s | | 7 s |
| Cached weights, warm page cache | 57 s | | 6 s |

- 51 s of the 58 s is process start and imports, so a warm page cache saves 1 s.
- With one request per second, 133, 58 and 57 requests were lost before the first answer. A readiness probe that says ready too early lets this traffic in. The first answer comes at the same time.
- A shared weight cache saves about 75 s per pod, and 16 pods download 2.9 GiB once and not 46.2 GiB (arithmetic, not measured).
- KV cache: 2 x layers x KV heads x head size x 2 bytes = 28,672 bytes per token. The vLLM log gave 29,433, a 3% gap.

| Context | KV per sequence | Sequences that fit in fp16 |
|---:|---:|---:|
| 1,024 | 0.027 GiB | 324 |
| 4,096 | 0.109 GiB | 81 |
| 32,768 | 0.875 GiB | 10 |

## Queue and scaling (simulation)

2,745 requests in 400 s: 60% of the capacity of 2 replicas, then 180% for 100 s, then 60%. The SLO is a latency under 6.4 s. The simulator uses the measured service time and the 133 s and 58 s cold starts.

| Setup | Shed | p95 | SLO violations | Replica-seconds |
|---|---:|---:|---|---:|
| Fixed 2, unbounded queue | 0% | 76.4 s | 79.4% [77.9, 80.9] | 799 |
| Fixed 2, queue limit 32 | 20.8% | 6.3 s | 22.5% [21.0, 24.1] | 799 |
| Autoscale, 133 s cold start | 0% | 61.1 s | 56.1% [54.2, 57.9] | 1,977 |
| Autoscale, 58 s cold start | 0% | 28.5 s | 42.4% [40.6, 44.3] | 1,977 |
| Autoscale, warm pool of 2 | 0% | 3.1 s | 0.0% [0.0, 0.1] | 2,183 |

A shorter cold start is worth more than a later scale-up. A warm pool costs 2.7 times the replica-seconds of 2 fixed replicas and removes every violation.

## Triton batchers (model in Python)

A stateful model keeps state per sequence. The dynamic batcher mixes the requests of different sequences in one batch, so the state goes to the wrong slot.

| Batcher | Model | Wrong per 1,000 windows, 3 seeds | Mean batch |
|---|---|---|---:|
| Dynamic | Stateful | 930, 931, 922 | 8.0 |
| Sequence | Stateful | 0, 0, 0 | 7.9 |
| Dynamic | Stateless | 0, 0, 0 | 8.0 |

The sequence batcher keeps each sequence in one slot and gives the same batch size. Triton uses it for stateful models and the dynamic batcher for stateless ones.
