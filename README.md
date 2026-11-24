# gpu-experiments

Small LLM inference experiments on free Kaggle Tesla T4 GPUs, measured with vLLM.
I did them while I prepared for the NVIDIA NCA-AMAD certification.
Each experiment follows one method: predict the result with simple math, measure it, then explain the gap.

## Key findings

- The KV cache size matched the per-token formula exactly. One token takes 28,672 bytes for Qwen2.5-1.5B, and 335,872 tokens fill 8.97 GiB, as vLLM reported. (KV cache: the stored attention keys and values of past tokens.)
- At low load, decode is limited by memory bandwidth. The 1.5B model read 3.1 GB of weights per token. The T4 floor is about 9.7 ms. The measured TPOT was 15.7 ms. (TPOT: time per output token.)
- Batching is cheap. From 1 to 4 concurrent requests, throughput grew 3.7 times and TPOT grew only 10%.
- The 7B model in 4-bit AWQ on one T4 had a lower TPOT than 7B fp16 on two T4s: 25.0 ms against 32.2 ms at concurrency 1. Two T4s gave more throughput at concurrency 64: 371 tokens/s against 241. (AWQ: 4-bit weight quantization. Tensor parallelism, or TP: one model split over several GPUs.)
- GSM8K scored 91 of 100 for fp16 and 90 of 100 for AWQ. The standard error is about 3 points, so this run shows no quality difference.
- Prefix caching cut the median TTFT 10 times on prompts that share a 2,000-token prefix. N-gram speculative decoding made a copy task 4 times faster at concurrency 1 and 2.1 times faster at concurrency 16.
- Under a 2 s TTFT p99 SLA, 500 users need 3 T4s on the 1.5B model, 8 on 7B AWQ and 14 on 7B with tensor parallelism. The latency target, not raw throughput, sets the count.
- In RAG over 87 NVIDIA articles, a cross-encoder reranker lifted recall@1 from 0.62 to 0.81 for 130 ms per query. BM25 beat dense retrieval, but the synthetic questions reuse passage words, which favours keyword search.
- QLoRA on Qwen2.5-0.5B took 5.8 minutes on one T4 and lifted news-label accuracy from 0.62 (4-shot prompt) to 0.89, with every reply valid JSON. A 50-question GSM8K probe showed no forgetting.
- In vLLM, the LoRA adapter and the merged model scored the same (0.87 on 100 items). The adapter request took 169 ms, the merged model 53 ms and the plain base 55 ms. The adapter is 34 MiB per task and the merged model is 953 MiB.
- RAPIDS cuDF ran six pandas steps on 10 million rows 13.9 times faster than the CPU, with no code change. At 100 thousand rows three of the six steps were slower on the GPU.
- A data flywheel with 250 human labels per turn raised the app-specific score from 0.60 to 0.82 in three turns. When the model labelled its own failures, the score fell to 0.29.
- A 7B ReAct agent with tools passed 54% of 24 support tasks. One call with all data in the prompt passed 79%. The agent failed all 4 tasks that it must refuse.
- Against 12 prompt-injection attacks, only a code check before the tool runs stopped all of them. Prompt hardening stopped none of the 5 that worked.
- On Kubernetes, a readiness probe kept 800 requests free of errors while new pods loaded a model. Without it, 134 of 800 failed. Mesh outlier detection cut failures from a poisoned pod from 145 to 3 of 300.
- A cart-pole controller trained on one set of physics scored 1.00 in simulation and 0.14 on a hidden machine. Wide domain randomization scored 0.82 with no real data. An extra action delay of 50 ms cut success from 0.94 to 0.13, so control must run at the edge.
- NVIDIA Warp on a T4 ran 16,384 simulated machines 256 times faster than NumPy on the CPU. With one machine the same Warp kernel was 12 times faster on the CPU.
- On hosted Nemotron models, reasoning took arithmetic from 0.1 to 1.0 on a 30B MoE model and gave nothing on extraction and routing, at 9 to 105 times the tokens. A router kept the accuracy of the largest model with 94% less compute proxy (15 tasks).
- With the same model and GPU, vLLM gave 858 tokens/s at 16 requests against 31 for a simple one-request server. The client moved to a hosted NIM with no code change.

## Charts

![Latency against throughput](charts/latency-throughput.png)

Each curve bends up when the GPU saturates. The 1.5B model gives the most throughput at the lowest latency.

![TTFT p99 against concurrency](charts/ttft-p99.png)

The slowest 1% of requests wait much longer for a first token as load grows. (TTFT: time to first token.)
The high 1.5B point at concurrency 1 is a cold start: that run had no warm-up request, and the median there was 115 ms.

![Memory per GPU](charts/memory-per-gpu.png)

Weights take less space in AWQ, so one T4 keeps more room for KV cache.

Run `python scripts/make_charts.py` to rebuild the charts from `results/sweeps-2026-11-23.csv`.

## Experiments

| Notebook | Question | Result |
|---|---|---|
| `vllm-t4-serving-benchmark.ipynb` | How do TTFT, TPOT and throughput change with concurrency, and does the vLLM KV cache match the sizing math? | [Results](results/vllm-t4-serving-benchmark-2026-11-23.md) |
| `vllm-t4-7b-tensor-parallel-vs-awq.ipynb` | A 7B fp16 model does not fit on one T4. Is tensor parallelism on two T4s or AWQ on one T4 better for latency, throughput and accuracy? | [Results](results/vllm-t4-7b-tensor-parallel-vs-awq-2026-11-23.md) |
| `sizing-capacity-plan.ipynb` | How many T4 GPUs do 500 users need under a latency SLA? | [Results](results/sizing-capacity-plan-2026-11-23.md) |
| `vllm-t4-inference-optimizations.ipynb` | What do prefix caching, speculative decoding, chunked prefill and request queuing each change? | [Results](results/vllm-t4-inference-optimizations-2026-11-23.md) |
| `rag-chunking-hybrid-rerank.ipynb` | On a corpus of public NVIDIA articles, how do chunk size, BM25 vs dense vs hybrid search and a reranker change retrieval and groundedness? | [Results](results/rag-chunking-hybrid-rerank-2026-11-23.md) |
| `lora-qlora-finetune-t4.ipynb` | When does QLoRA fine-tuning beat prompting, how do overfitting and forgetting look, and how does adapter serving compare with a merged model? | [Results](results/lora-qlora-finetune-t4-2026-11-24.md) |
| `data-flywheel-cudf.ipynb` | How much does cuDF speed up data preparation, what does bad data cost, and does a data flywheel improve the application? | [Results](results/data-flywheel-cudf-2026-11-24.md) |
| `agent-react-guardrails.ipynb` | What does a multi-step agent buy over one LLM call, what does it cost, and which guardrail stops a prompt injection? | [Results](results/agent-react-guardrails-2026-11-24.md) |
| [`k8s-model-service/`](k8s-model-service/README.md) (GitHub Actions, no GPU) | How does Kubernetes keep a model service available while it crashes, updates and grows, and what does a service mesh add? | [Results](results/k8s-model-service-2026-11-24.md) |
| `sim-to-real-domain-randomization.ipynb` | How much do domain randomization and a sim-to-real flywheel close the gap, and why must the controller run at the edge? | [Results](results/sim-to-real-domain-randomization-2026-11-24.md) |
| `nim-model-selection.ipynb` | Which hosted Nemotron model should serve which workload, and when is reasoning worth its tokens? | [Results](results/nim-model-selection-2026-11-24.md) |
| `nim-three-backends.ipynb` | Can one client move between self-hosted and cloud, and how does private data stay off the cloud? | [Results](results/nim-three-backends-2026-11-24.md) |
| `cosmos-physical-check.ipynb` | Can a physical-AI model tell possible motion from impossible motion? | Not run yet |

## How to run

1. Open a notebook in Kaggle: File, Import Notebook.
2. Open Settings, then Accelerator, and choose GPU T4 x2. `sizing-capacity-plan.ipynb` needs no GPU.
3. Turn Internet on.
4. Choose Run All.

Kaggle gives about 30 GPU hours each week for free.

## Setup

This table is for the two serving benchmarks.
The results file of each other experiment gives its own setup.

| Item | Value |
|---|---|
| GPU | Tesla T4, 15 GB, compute capability 7.5 (no bf16, so fp16) |
| Serving | vLLM 0.34.0 (V1 engine) |
| Framework | torch 2.14 with CUDA 13.0 (cu130) |
| Models | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| Load | `vllm bench serve`, random prompts, 512 input tokens, 128 output tokens |

## Limits

Each number comes from one run with one seed.
The samples are small: 16 to 256 requests per point and 100 GSM8K questions in the serving benchmarks, 101 synthetic questions in RAG, and 300 test items in QLoRA.
The two serving benchmarks use random prompts, so prefix caching does nothing there.
The load generator runs on the same machine, so there is no network latency.
The two-replica AWQ throughput in the results file is an estimate and not a measurement.

## License

MIT. See [LICENSE](LICENSE).
