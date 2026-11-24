# NIM model selection, 2026-11-24

One full run of `nim-model-selection.ipynb` (commit `3041704`) in the Kaggle notebook of the same name, on a CPU session.
The calls went to the hosted API at `integrate.api.nvidia.com`. The run took 8.5 minutes.

## Setup

30 tasks with exact answers: 10 arithmetic word problems, 10 JSON field extractions, 10 message routings.
Two requests at a time, 120 seconds for each model and mode, at most 2,048 output tokens.
The compute proxy is completion tokens times active parameters in billions. It is not a price.

| Model | Total and active parameters | Reasoning switch that worked |
|---|---|---|
| `nemotron-3.5-bolt-e4b` | treated as 4B | none: 4 tokens in every mode, so it does not think on this prompt |
| `nemotron-3.5-lightning-30b-a3b` | 30B, 3B active | `enable_thinking=false` and `reasoning_effort=none` |
| `nemotron-3.5-super-vl-120b-a12b` | 120B, 12B active | `enable_thinking=false` and `reasoning_effort=none` |
| `nemotron-3-ultra-550b-a55b` | 550B, 55B active | a switch was found and used |
| `google/gemma-3-4b-it` | 4B | not tested: every request gave HTTP 404, although the model is in the list |

The `/no_think` system prompt shortened the thinking of two models but did not switch it off.

## Results (accuracy of 10 tasks, median seconds)

| Model | Mode | Arithmetic | Extraction | Routing |
|---|---|---|---|---|
| bolt-e4b | default | 0.1, 0.18 s | 1.0, 0.76 s | 1.0, 0.17 s |
| lightning-30b | reasoning | 1.0, 2.37 s | 1.0, 2.67 s | 1.0, 1.55 s |
| lightning-30b | off | 0.1, 0.24 s | 1.0, 0.46 s | 1.0, 0.23 s |
| ultra-550b | reasoning | 1.0, 5.69 s | 1.0, 6.30 s | 1.0, 2.42 s |
| ultra-550b | off | 0.7, 0.85 s | 1.0, 3.00 s | 1.0, 1.11 s |
| super-vl-120b | reasoning | 1.0 of 4 tasks, 11.2 s | 1.0 of 3 tasks, 20.5 s | 1.0 of 5 tasks, 20.2 s |
| super-vl-120b | off | 0.4 of 5 tasks, 13.8 s | 1.0 of 6 tasks, 16.0 s | 1.0 of 5 tasks, 15.4 s |

The super-vl-120b runs were cut off by the time budget. Its wait was 15 seconds or more even for 4 tokens, so this looks like a queue on the service and not model speed.

## Reasoning on against off

| Model | Workload | Accuracy gain | Tokens with and without | Token ratio |
|---|---|---:|---|---:|
| lightning-30b | arithmetic | +0.9 | 369 and 3.4 | 108x |
| lightning-30b | extraction | 0 | 467 and 50 | 9x |
| lightning-30b | routing | 0 | 274 and 2.6 | 105x |
| ultra-550b | arithmetic | +0.3 | 158 and 3.3 | 48x |
| ultra-550b | extraction | 0 | 194 and 50 | 4x |
| ultra-550b | routing | 0 | 58 and 2.6 | 22x |

## Router against the largest model

The router picks on 5 tasks per workload and is scored on the other 5.
It chose lightning-30b for all three: reasoning on for arithmetic, off for extraction and routing.

| Strategy | Tasks | Accuracy | Mean tokens | Mean compute proxy | Mean latency |
|---|---:|---:|---:|---:|---:|
| Router | 15 | 1.0 | 138 | 415 | 1.54 s |
| Always ultra-550b with reasoning | 15 | 1.0 | 136 | 7,473 | 8.26 s |

## Tokens per word

| Text | Tokens per word (lightning-30b) |
|---|---:|
| English prose | 1.06 |
| Spanish | 1.33 |
| Python code | 2.22 |
| JSON | 3.89 |
| Long numbers | 10.75 |

## What the numbers show

- **Reasoning pays only where the task needs steps.** It took arithmetic from 0.1 to 1.0 on the small MoE model. On extraction and routing it gave nothing and cost 9 to 105 times the tokens.
- **A small model with reasoning beat a huge model without it.** lightning-30b with reasoning scored 1.0 on arithmetic. ultra-550b without reasoning scored 0.7.
- **A router kept the accuracy and cut the compute proxy by 94% and the latency by 81%.** The gain comes from the workload, so measure your own.
- **Active parameters set the compute per token.** 30B total with 3B active has a lower proxy than a dense 4B model.
- **A model in the list may not be callable.** Check with a real request.
- **Numbers and JSON cost more tokens than prose.** Long digits took 10.75 tokens for each number.

## Limits

10 tasks per workload, one run, and a shared service with changing load.
Many scores are at the ceiling of 1.0, so the tasks cannot rank the good models.
The router comparison has 15 tasks.
