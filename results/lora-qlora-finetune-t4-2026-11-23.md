# QLoRA fine-tuning on a T4, 2026-11-23 (partial)

One run of `lora-qlora-finetune-t4.ipynb` in the Kaggle notebook of the same name.
Training, evaluation and the forgetting probe finished.
The adapter serving step did not.
The laptop went to sleep about 90 seconds after the vLLM server cell started, and Kaggle later ended the idle session.
No evidence shows that the server itself stalled.
An earlier version of this file said that it did.
So this file has no serving numbers yet.

Two fixes went in during the run:

- The run chain stopped twice at a cell boundary with no error, so it was resumed from the next cell in the same session.
- The merge cell failed because Kaggle ships torchao 0.10 and peft rejects versions below 0.16 when it loads an adapter onto an fp16 base. The fix tells peft that torchao is not available (commit `6d93597`). The 4-bit training path never reaches that check.

## Setup

| Item | Value |
|---|---|
| GPU | Tesla T4 (GPU 0 for training) |
| Libraries | torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0 |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Task | `fancyzhx/ag_news` topic label as strict JSON, `{"label": "<World|Sports|Business|Sci/Tech>"}` |
| Data | 2,000 train, 300 validation, 300 test, balanced, fixed seed |
| QLoRA | 4-bit NF4 base, fp16 compute, LoRA r=16, alpha=32, dropout 0.05 on attention and MLP projections, 2 epochs, loss on answer tokens only |

## Results

| Method | Setting | Test accuracy | Valid JSON |
|---|---|---:|---:|
| Prompting | base fp16, zero-shot | 0.523 | 0.987 |
| Prompting | base fp16, 4-shot | 0.623 | 0.787 |
| QLoRA SFT | 2,000 examples, 2 epochs | 0.893 | 1.000 |
| QLoRA SFT | 64 examples, 8 epochs (overfit run) | 0.883 | 0.993 |

| Run | Training time | Peak GPU memory allocated | Notes |
|---|---:|---:|---|
| Main, 2,000 examples | 5.6 min | 5.72 GiB | Adapter file 33.6 MiB |
| Overfit, 64 examples | 2.2 min | 6.43 GiB | Validation loss lowest 0.070 at step 16, last 0.083 at step 64. Train loss 0.150 to 0.001. |

Forgetting probe, 50 GSM8K test questions, greedy, same 4-bit base:

| Model | GSM8K accuracy | Replies that start with `{` |
|---|---:|---:|
| Base (adapter off) | 0.14 | 0 |
| Base + adapter | 0.16 | 0 |

## What the numbers show

- **Fine-tuning beat prompting by a wide margin.** QLoRA reached 0.893 against 0.623 for the best prompt, and every reply was valid JSON. The 4-shot prompt improved accuracy but broke the output format in a fifth of replies.
- **QLoRA is cheap.** Training took under 6 minutes on one T4 and peaked at 5.72 GiB. The adapter is 33.6 MiB against about 1 GB for the fp16 model.
- **Overfitting shows in the loss before the accuracy.** With 64 examples, train loss fell to 0.001 while validation loss turned up after step 16. Test accuracy was still 0.883, because the task is easy and the label set is small. Loss curves catch overfitting earlier than a coarse accuracy number.
- **No forgetting shown on this probe.** GSM8K was 0.14 without the adapter and 0.16 with it, and no math reply leaked the news JSON format. One question is 2 points, so the probe can only catch large damage. A 0.5B model in 4-bit is also weak at math, so the floor is low.

## Still to run

Adapter serving in vLLM (`--enable-lora --lora-modules news=<dir> --max-lora-rank 16`) against the merged fp16 model, on 100 test items.
The next run is server-side (Save Version, Save and Run All), so a sleeping laptop cannot stop it.
The notebook now sends 5 smoke requests first, and the health check has a timeout (commit `0ca561d`).
