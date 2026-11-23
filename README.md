# gpu-experiments

Small GPU experiments on free compute, to learn inference, sizing and fine-tuning by measuring them.

| Notebook | Question | Where it runs |
|---|---|---|
| `vllm-t4-serving-benchmark.ipynb` | How do TTFT, TPOT and throughput change with concurrency, and does vLLM's KV cache match the sizing math? | Kaggle, 1x Tesla T4 |
| `vllm-t4-7b-tensor-parallel-vs-awq.ipynb` | A 7B fp16 model does not fit one T4. Is tensor parallelism over two T4s or INT4 AWQ on one T4 better for latency, throughput and GSM8K accuracy? | Kaggle, 2x Tesla T4 |

To run one: open it in a Kaggle notebook, set Accelerator to GPU T4 x2 and Internet on, then Run All.
Results land in `results/`.
