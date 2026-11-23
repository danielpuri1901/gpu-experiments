# 7B on Kaggle T4s: tensor parallelism vs INT4 AWQ, 2026-11-23

One run of `vllm-t4-7b-tensor-parallel-vs-awq.ipynb` in the Kaggle notebook of the same name.

## Setup

| Item | Value |
|---|---|
| GPUs | 2x Tesla T4, 15360 MiB each, compute capability 7.5 |
| GPU link | `nvidia-smi topo -m` says PHB: the two T4s talk over PCIe through the CPU's host bridge. No NVLink. |
| Software | vLLM 0.34.0 (V1 engine), torch 2.14.0+cu130, NCCL 2.30.7 |
| Config A | `Qwen/Qwen2.5-7B-Instruct`, fp16, `--tensor-parallel-size 2` on both T4s |
| Config B | `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit, group size 128), fp16 activations, one T4 |
| Load | `vllm bench serve`, random prompts, 512 input tokens, 128 output tokens, `--ignore-eos`, one warm-up request first |
| Quality | First 100 GSM8K test questions, greedy decoding, max 512 tokens |

fp16 7B does not fit on one T4: 14.18 GiB of weights against a 13.10 GiB budget (0.9 of the 14.56 GiB that vLLM sees).

## Startup facts from the vLLM log

| | TP=2 fp16 | AWQ on one T4 |
|---|---|---|
| Server ready after | 291 s (includes the 15 GB download) | 165 s |
| Weights per GPU ("Model loading took") | 7.16 GiB on each T4 | 5.29 GiB |
| Kernel or backend notes | All-reduce backends CUSTOM then PYNCCL. FlashInfer all-reduce off for world size 2. Triton attention. | `MarlinLinearKernel` for AWQ. Triton attention. |
| KV cache predicted | 208,269 tokens | 139,793 tokens |
| KV cache from vLLM | 159,744 tokens (39 full 4,096-token requests) | 112,640 tokens (27.5 requests) |
| GPU memory in use | 12,739 MiB on each T4 | 12,529 MiB on one T4 |

## Results

| Config | Concurrency | TTFT median ms | TTFT p99 ms | TPOT median ms | Output tok/s | Requests/s |
|---|---:|---:|---:|---:|---:|---:|
| TP=2 fp16 | 1 | 263 | 339 | 32.2 | 28 | 0.22 |
| TP=2 fp16 | 4 | 228 | 310 | 36.2 | 106 | 0.83 |
| TP=2 fp16 | 16 | 1509 | 3750 | 55.7 | 231 | 1.80 |
| TP=2 fp16 | 64 | 2744 | 13780 | 170.4 | 371 | 2.90 |
| AWQ 1 GPU | 1 | 423 | 493 | 25.0 | 35 | 0.28 |
| AWQ 1 GPU | 4 | 285 | 397 | 27.3 | 136 | 1.06 |
| AWQ 1 GPU | 16 | 3016 | 6719 | 58.9 | 192 | 1.50 |
| AWQ 1 GPU | 64 | 5043 | 27121 | 264.1 | 241 | 1.88 |

| | TP=2 fp16 | AWQ on one T4 |
|---|---:|---:|
| GSM8K, 100 questions | 91 | 90 |
| GSM8K wall time, 32 parallel requests | 55 s | 51 s |

## What the numbers show

- **AWQ wins at low load.** TPOT at concurrency 1 is 25.0 ms for AWQ against 32.2 ms for TP=2. Decode is bandwidth bound. AWQ reads about 5.7 GB per token on one GPU (floor about 18 ms at 320 GB/s). TP=2 reads about 7.7 GB per GPU (floor about 24 ms), then adds two all-reduces per layer, 56 per token, over PCIe.
- **TP=2 wins on prefill and at high load.** TTFT at concurrency 1 is 263 ms against 423 ms, because prefill is compute bound and TP=2 has two GPUs of compute. At concurrency 64, TP=2 gives 371 tok/s against 241 tok/s, with a p99 TTFT of 13.8 s against 27.1 s.
- **Per GPU, AWQ is more efficient.** At concurrency 64, TP=2 gives 186 tok/s per GPU and AWQ gives 241. Two AWQ replicas, one per T4, would probably give about 480 tok/s (an estimate, not measured). So when the model fits on one GPU, more replicas (horizontal scaling) beat tensor parallelism for throughput. Tensor parallelism is for a model that does not fit, or for a lower TTFT.
- **No measurable quality loss.** 91 against 90 correct out of 100 is one question. The standard error at 100 questions is about 3 points, so this run cannot show a difference. A real quantization check needs more questions and a task close to the use case.
- **Memory overhead grows with model size.** The 0.45 GiB activation guess from the 1.5B run was too low for 7B. The real non-KV overhead was about 1.7 GiB per GPU for TP=2 (CUDA graphs, NCCL buffers, larger activations and the 152k-token vocabulary logits) and about 1.8 GiB for AWQ (plus the dequantization workspace). So the KV cache came out 19% to 23% below the prediction.
- **AWQ compresses about 2.7x, not 4x.** 14.18 GiB in fp16 became 5.29 GiB, because the embeddings and the output layer stay in fp16 and every group of 128 weights carries a scale.

## Limits

One run, one seed, one model family, 100 GSM8K questions.
Random prompts, so the prefix cache does nothing.
No two-replica AWQ run, so the horizontal-scaling number is an estimate.
