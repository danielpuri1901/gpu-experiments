# QLoRA fine-tuning and adapter serving on a T4, 2026-11-24

One full run of `lora-qlora-finetune-t4.ipynb` (commit `0ca561d`) in the Kaggle notebook of the same name.
All cells ran in order in one kernel, in 16.2 minutes after the library install.

A first run on 2026-11-23 finished the training steps but not the serving step.
The laptop went to sleep about two minutes after the vLLM server cell started, and Kaggle ended the idle session.
The vLLM server with LoRA needs 172 seconds to start, so it had no time to answer.
An earlier version of this file said that the server stalled. That was wrong.

The 2026-11-24 run gave the same accuracy numbers as the first run for every training and probe row.

## Setup

| Item | Value |
|---|---|
| GPU | Tesla T4 (GPU 0 for training, GPU 1 for serving) |
| Libraries | torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0 |
| Serving | vLLM 0.26.0 (V1 engine), fp16, `--max-model-len 2048 --gpu-memory-utilization 0.8` |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Task | `fancyzhx/ag_news` topic label as strict JSON, `{"label": "<topic>"}`. Topics: World, Sports, Business, Sci/Tech. |
| Data | 2,000 train, 300 validation, 300 test, balanced, fixed seed |
| QLoRA | 4-bit NF4 base, fp16 compute, LoRA r=16, alpha=32, dropout 0.05 on attention and MLP projections, 2 epochs, loss on answer tokens only |

pip installed vLLM 0.26.0 in this notebook and 0.34.0 in the serving benchmarks.
So do not compare latency numbers between them.

## Training and prompting

| Method | Setting | Test accuracy | Valid JSON |
|---|---|---:|---:|
| Prompting | base fp16, zero-shot | 0.523 | 0.987 |
| Prompting | base fp16, 4-shot | 0.623 | 0.787 |
| QLoRA SFT | 2,000 examples, 2 epochs | 0.893 | 1.000 |
| QLoRA SFT | 64 examples, 8 epochs (overfit run) | 0.883 | 0.993 |

| Run | Training time | Peak GPU memory allocated | Notes |
|---|---:|---:|---|
| Main, 2,000 examples | 5.8 min | 5.72 GiB | Adapter file 33.6 MiB. 8.8 million trainable parameters (1.75%). |
| Overfit, 64 examples | 2.3 min | 6.43 GiB | Validation loss lowest 0.070 at step 16, last 0.083 at step 64. Train loss 0.150 to 0.001. |

Forgetting probe, 50 GSM8K test questions, greedy, same 4-bit base:

| Model | GSM8K accuracy | Replies that start with `{` |
|---|---:|---:|
| Base (adapter off) | 0.14 | 0 |
| Base + adapter | 0.16 | 0 |

## Serving: adapter against merged model

100 balanced test items, one request at a time, at most 16 output tokens.

| Server | Accuracy | Valid JSON | Median latency | Server start |
|---|---:|---:|---:|---:|
| vLLM fp16 base, no adapter | 0.510 | 0.990 | 54.7 ms | 172 s |
| vLLM fp16 base + adapter `news` | 0.870 | 1.000 | 169.2 ms | same server |
| vLLM merged fp16 model | 0.870 | 1.000 | 53.2 ms | 88 s |
| Transformers, 4-bit base + adapter | 0.880 | 1.000 | not measured | |

The adapter is 33.6 MiB on disk. The merged model is 953 MiB on disk.

In the LoRA server log, `torch.compile` took 28 seconds and CUDA graph capture took 53 seconds.
The LoRA server ran first, with cold caches, so the difference in start time is not only the cost of LoRA.

## What the numbers show

- **Fine-tuning beat prompting by a wide margin.** QLoRA reached 0.893 against 0.623 for the best prompt, and every reply was valid JSON. The 4-shot prompt improved accuracy but broke the output format in a fifth of replies.
- **QLoRA is cheap.** Training took under 6 minutes on one T4 and peaked at 5.72 GiB.
- **Overfitting shows in the loss before the accuracy.** With 64 examples, train loss fell to 0.001 while validation loss turned up after step 16. Test accuracy was still 0.883, because the task is easy and the label set is small.
- **No forgetting shown on this probe.** GSM8K was 0.14 without the adapter and 0.16 with it, and no math reply leaked the news JSON format. One question is 2 points, so the probe can only catch large damage.
- **Merging kept the accuracy and removed the adapter cost.** The adapter and the merged model scored the same on the 100 items. The merged model was as fast as the base model. The adapter request was 3.1 times slower.
- **The prediction was wrong.** The notebook predicted "a little extra latency" for the adapter. This run does not show the cause. vLLM captured CUDA graphs with LoRA on, so a fallback to eager mode does not look like the cause. The model is very small and the replies are short, so a fixed cost per layer weighs more here than on a large model. That part is a guess.
- **So the choice is a trade.** One base model plus one 33.6 MiB adapter per task saves memory when there are many tasks with low traffic. One merged model per task is faster per request and costs about 1 GB for each task.

## Limits

One seed and one small model.
Latency is for single requests, not for a batch.
100 items give a standard error of about 3 points on accuracy.
The 4-bit and fp16 rows differ in base precision, so a one-item gap between them is noise.
