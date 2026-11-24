# Possible or impossible motion, hosted models, 2026-11-24

One full run of `cosmos-physical-check.ipynb` (commit `d116d63`) in the Kaggle notebook of the same name, on a CPU session with the hosted NVIDIA API.
The run chain stopped once before the media probe with no error, so it was resumed from that cell in the same kernel.

**The main question has no answer here.** `nvidia/cosmos-reason2-8b` is in the model list, but every request returned HTTP 404 for this account.
`google/gemma-3-12b-it` and `google/gemma-3-4b-it` did the same.
The notebook `cosmos-reason2-vs-base-t4.ipynb` tests the open 2B Cosmos model on a T4.

A first run (commit `f00f2e5`) gave the two reasoning models too few tokens and too little time. Many replies were cut. This run uses 4,000 tokens and adds a mode with reasoning off.

## Setup

24 clips of 3 seconds from a 2D physics simulation: 12 plausible, 12 impossible, 7 kinds of violation.
The simulator gives the exact answer, and a check proves each clip.
One fixed question, temperature 0, video input.

## Results

| Model | Mode | Answered | Accuracy (95% interval) | Accuracy over all 24 | Recall, impossible | False alarm | Says plausible | Median s | Tokens |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| nemotron-3.5-super-vl-120b | reasoning | 17 | 0.71 (0.47 to 0.87) | 0.50 | 0.75 | 0.33 | 0.47 | 27.0 | 2,235 |
| nemotron-3.5-super-vl-120b | off | 24 | 0.42 (0.24 to 0.61) | 0.42 | 0.25 | 0.42 | 0.67 | 13.5 | 49 |
| nemotron-3-nano-omni-30b | reasoning | 24 | 0.62 (0.43 to 0.79) | 0.62 | 0.50 | 0.25 | 0.62 | 11.0 | 1,227 |
| nemotron-3-nano-omni-30b | off | 24 | 0.58 (0.39 to 0.76) | 0.58 | 0.92 | 0.75 | 0.17 | 1.1 | 26 |
| Always plausible | baseline | 24 | 0.50 (0.31 to 0.69) | 0.50 | 0.00 | 0.00 | 1.00 | | |

Critic test with the best model (nano-omni with reasoning) on 40 clips with 30% bad: purity rose from 0.70 to 0.77, with a yield of 0.55.
It threw away 11 of 28 good clips and kept 5 of 12 bad ones.

Stated evidence that names the right violation, among correct "impossible" answers: 3 of 6 (super-vl, reasoning), 1 of 3 (super-vl, off), 1 of 6 (nano-omni, reasoning), 3 of 11 (nano-omni, off).

## What the numbers show

- **No model clearly beat a guess.** Every interval includes 0.5. With 24 clips, only a large difference would show.
- **A high recall can be a bias.** With reasoning off, nano-omni found 11 of 12 impossible clips, but it called 9 of 12 plausible clips impossible too.
- **Right answer, wrong reason.** Most correct "impossible" answers gave the wrong evidence. Several answers described a ball that splits in two, which happens in no clip.
- **A weak critic costs good data.** Purity rose by 7 points and 39% of the good clips were lost.
- **Reasoning needs a token budget.** Even with 4,000 tokens, 7 of 24 replies of the 120B model had no verdict.

## Limits

24 simple 2D clips, one run, one question.
The hosted service shares its load, so latency is not a model property.
