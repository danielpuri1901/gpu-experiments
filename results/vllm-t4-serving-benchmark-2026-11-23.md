# vLLM serving benchmark on a Kaggle T4, 2026-11-23

One run of `vllm-t4-serving-benchmark.ipynb` on one Tesla T4.

## Setup

| Item | Value |
|---|---|
| GPU | 1x Tesla T4, 15360 MiB, compute capability 7.5 (the second T4 was idle) |
| Software | vLLM 0.34.0 (V1 engine), torch 2.14.0+cu130, driver 580.178.04 |
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, 1.54 B parameters, fp16 (T4 has no bf16) |
| Server | `vllm serve --dtype half --max-model-len 4096 --gpu-memory-utilization 0.90` |
| Load | `vllm bench serve`, random prompts, 512 input tokens, 128 output tokens, `--ignore-eos` |

## Startup facts from the vLLM log

- Attention backend: `TRITON_ATTN`. FlashAttention needs a newer GPU, so vLLM picked Triton on Turing.
- Weight download took 21 s. Model loading took 2.98 GiB of GPU memory.
- vLLM saw 14.56 GiB total memory, not 15.0 GiB. It used 0.9 of that, which is 13.11 GiB.
- Transient activations took 0.45 GiB.
- Final KV cache: 5248 blocks of 64 tokens, which is 335,872 tokens in 8.97 GiB.
- "Maximum concurrency for 4,096 tokens per request: 82.00x".
- Server was ready 146 s after start, including torch.compile and CUDA graph capture.

## Sizing math against the log

KV cache per token = 2 (K and V) x 28 layers x 2 KV heads x 128 head_dim x 2 bytes = 28,672 bytes (28 KiB).
335,872 tokens x 28,672 bytes = 8.97 GiB, which is the same as the log.
So the per-token formula is exact.

The notebook predicted 360,433 tokens and 88 full-length requests.
vLLM reported 335,872 tokens and 82 requests, 7% less.
The difference comes from total memory: the driver reserves some of the 15 GiB, so vLLM sees only 14.56 GiB.
The 1 GiB overhead guess was too high (real: 0.45 GiB transient), which partly cancels that error.

The model uses grouped-query attention (GQA): 12 query heads share 2 KV heads.
Without GQA the KV cache would be 6 times bigger, and only about 14 full-length requests would fit.

## Results

| Concurrency | Requests | TTFT median ms | TTFT p99 ms | TPOT median ms | Output tok/s | Requests/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 115 | 979 | 15.7 | 59 | 0.46 |
| 4 | 16 | 104 | 199 | 17.3 | 220 | 1.72 |
| 16 | 64 | 754 | 1734 | 28.5 | 463 | 3.62 |
| 64 | 256 | 1461 | 6878 | 96.1 | 650 | 5.08 |

GPU memory in use during the run: 13,225 MiB of 15,360 MiB.
vLLM reserves the memory at startup, whatever the load.

## What the numbers show

- Decode is memory-bandwidth bound at low load. Each output token reads all 3.1 GB of weights. At the T4's 320 GB/s, the floor is about 9.7 ms per token. The measured 15.7 ms is about 60% of that roofline.
- Continuous batching shares each weight read across sequences. From 1 to 4 concurrent requests, throughput grew 3.7 times while TPOT grew only 10%.
- From 16 to 64, the GPU probably became compute bound (not measured directly). Throughput grew only 1.4 times, and TPOT grew 3.4 times.
- KV cache was not the limit. At 64 requests, the cache held about 41,000 tokens, which is 12% of its 335,872-token capacity.
- TTFT grows with load because new prompts wait for prefill behind other work. The p99 at 64 is 6.9 s.
- The p99 TTFT of 979 ms at concurrency 1 is probably a first-request warm-up outlier, because the median is 115 ms. A warm-up request before the sweep would remove it.

## Limits

One run, one seed, one model, one GPU.
Random prompts, so the prefix cache does nothing.
Requests went from the same VM, so there is no network latency.
