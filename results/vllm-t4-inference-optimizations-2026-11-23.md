# vLLM inference optimizations on a T4, 2026-11-23

One run of `vllm-t4-inference-optimizations.ipynb` in the Kaggle notebook of the same name.
It took 25 minutes of GPU time, and all five server configs started.

## Setup

| Item | Value |
|---|---|
| GPU | 1x Tesla T4 (the second T4 was idle) |
| Software | vLLM 0.34.0 (V1 engine) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, fp16, `--max-model-len 4096 --gpu-memory-utilization 0.9` |
| Client | A streaming load generator in the notebook (Python standard library). It reads TTFT from the first text chunk and the token count from the usage chunk. |

| Config | Flag |
|---|---|
| S1 baseline | vLLM defaults (prefix caching on) |
| S2 | `--no-enable-prefix-caching` |
| S3 | `--speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}'` |
| S4 | `--max-num-batched-tokens 512` |
| S5 | `--max-num-seqs 4` |

## Results

| Optimization | Workload | Metric | Off | On | Change |
|---|---|---|---:|---:|---:|
| Prefix caching | 64 requests, shared 2k-token prefix, concurrency 8 | TTFT median ms | 2843.3 | 278.9 | -90.2% |
| Prefix caching | same | TTFT p99 ms | 6938.9 | 299.2 | -95.7% |
| Speculative decoding (n-gram) | copy a 500-token passage, concurrency 1 | TPOT median ms | 16.5 | 3.6 | -78% |
| Speculative decoding (n-gram) | same, concurrency 1 | output tok/s | 55.8 | 224.3 | +302% |
| Speculative decoding (n-gram) | same, concurrency 16 | TPOT median ms | 26.1 | 10.7 | -59% |
| Speculative decoding (n-gram) | same, concurrency 16 | output tok/s | 584.3 | 1227.3 | +110% |
| Token budget 512 (chunked prefill) | 8 short streams + 8 long prompts, concurrency 16 | short-request TPOT p99 ms | 77.6 | 76.7 | -1.2% |
| Token budget 512 (chunked prefill) | same | long-request TTFT median ms | 7107.2 | 7044.2 | -0.9% |
| Sequence limit 4 (request queuing) | 64 requests, concurrency 32 | TTFT median ms | 1047.6 | 16678.3 | +1492% |
| Sequence limit 4 (request queuing) | same | TTFT p99 ms | 1510.6 | 17687.1 | +1071% |
| Sequence limit 4 (request queuing) | same | TPOT median ms | 33.0 | 17.3 | -48% |
| Streaming | 8 requests, concurrency 1, 256 tokens | wait before first text, ms (off: full reply, on: first token) | 5447.2 | 38.7 | -99.3% |

For the sequence limit, "off" is the vLLM default and "on" is a limit of 4 sequences.

From the vLLM `/metrics` endpoint:

- Prefix caching on: during the prefix workload, 126,976 of 129,334 prompt tokens were cache hits (98%).
- Speculative decoding: 6,472 of 7,052 draft tokens were accepted (92%).

## What the numbers show

- **Prefix caching removes repeated prefill.** With a shared 2,000-token prefix, the median TTFT fell 10 times. Prefill is the compute-bound part of a request, and the cache skips it for the shared tokens.
- **Speculative decoding helps most at low batch size.** On a copy task, n-gram drafts were right 92% of the time, so each decode step produced several tokens. The speed-up was 4 times at concurrency 1 and 2.1 times at concurrency 16. At low batch the GPU has spare compute to check drafts. At high batch that spare compute is smaller. A copy task is the best case for n-gram drafts, so normal chat will gain less.
- **Request queuing trades TTFT for TPOT.** With only 4 sequences running, each one decoded about twice as fast, but new requests waited in the queue for about 17 seconds. This is the latency against throughput trade-off at the scheduler level.
- **Streaming changes perceived latency, not total time.** The user saw text after 39 ms instead of 5.4 seconds.
- **The token budget made no difference in this run.** Short-request TPOT p99 and long-request TTFT changed by about 1%. A probable reason is that the T4 is compute bound during the long prefills, so smaller chunks reorder the work without reducing it. This run does not prove that reason.

## Limits

One run, one seed, one small model.
The speculative decoding task is chosen to suit n-gram drafts.
The token budget workload may not stress the scheduler enough to show the effect.
