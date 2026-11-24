# Possible or impossible motion, hosted models, 2026-11-24

Two full runs of `cosmos-physical-check.ipynb` in the Kaggle notebook of the same name, on CPU with the hosted NVIDIA API.
Run A used commit `d116d63` in an interactive session. It stopped once before the media probe with no error, so it was resumed from that cell.
Run B used commit `303d5cf` as a server-side batch run (Version 2, 18 minutes).
Run B has a fixed answer parser that also accepts a short "yes" or "no".

**The main question has no answer here.** `nvidia/cosmos-reason2-8b` is in the model list, but every request returned HTTP 404 for this account in both runs.
`google/gemma-3-12b-it` and `google/gemma-3-4b-it` did the same.
The notebook `cosmos-reason2-vs-base-t4.ipynb` tests the open 2B Cosmos model on a T4.

An earlier run (commit `f00f2e5`) gave the two reasoning models too few tokens and too little time, so many replies were cut. Runs A and B use 4,000 tokens and add a mode with reasoning off.

## Setup

24 clips of 3 seconds from a 2D physics simulation: 12 plausible, 12 impossible, 7 kinds of violation.
The simulator gives the exact answer, and a check proves each clip.
One fixed question, temperature 0, a fixed seed, video input.
Both runs use the same clips and the same question.

## Results of run B

| Model | Mode | Answered | Accuracy (95% interval) | Accuracy over all 24 | Recall, impossible | False alarm | Says plausible | Median s | Tokens |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| nemotron-3.5-super-vl-120b | reasoning | 19 | 0.53 (0.32 to 0.73) | 0.42 | 0.80 | 0.78 | 0.21 | 39.1 | 2,178 |
| nemotron-3.5-super-vl-120b | off | 24 | 0.54 (0.35 to 0.72) | 0.54 | 0.33 | 0.25 | 0.71 | 24.4 | 49 |
| nemotron-3-nano-omni-30b | reasoning | 20 | 0.55 (0.34 to 0.74) | 0.46 | 0.22 | 0.18 | 0.80 | 14.0 | 1,655 |
| nemotron-3-nano-omni-30b | off | 24 | 0.58 (0.39 to 0.76) | 0.58 | 0.92 | 0.75 | 0.17 | 1.4 | 31 |
| Always plausible | baseline | 24 | 0.50 (0.31 to 0.69) | 0.50 | 0.00 | 0.00 | 1.00 | | |

Stated evidence that names the right violation, among correct "impossible" answers: 2 of 8 (super-vl, reasoning), 1 of 4 (super-vl, off), 2 of 2 (nano-omni, reasoning), 5 of 11 (nano-omni, off).
This check looks for words of the violation, so it is generous.

## The same test two times

| Model | Mode | Correct of answered, run A | Correct of answered, run B |
|---|---|---:|---:|
| nemotron-3.5-super-vl-120b | reasoning | 12 of 17 (0.71) | 10 of 19 (0.53) |
| nemotron-3.5-super-vl-120b | off | 10 of 24 (0.42) | 13 of 24 (0.54) |
| nemotron-3-nano-omni-30b | reasoning | 15 of 24 (0.62) | 11 of 20 (0.55) |
| nemotron-3-nano-omni-30b | off | 14 of 24 (0.58) | 14 of 24 (0.58) |

| Critic test, 40 clips with 30% bad | Run A | Run B |
|---|---|---|
| Config that scored best on the 24 clips | nano-omni with reasoning | nano-omni without reasoning |
| Purity before and after | 0.70 and 0.77 | 0.70 and 0.50 |
| Yield of good clips | 0.55 | 0.05 |
| Good clips rejected, of 28 | 11 | 27 |
| Bad clips kept, of 12 | 5 | 1 |

In run B the critic kept 2 of 40 clips, 1 good and 1 bad.

## What the numbers show

- **No model clearly beat a guess.** Every interval includes 0.5 in both runs. With 24 clips, only a large difference would show.
- **Temperature 0 does not make a hosted model repeatable.** Three of four scores moved by 7 to 18 points with the same clips, question and seed. With reasoning off, every clip has an answer in both runs, so the parser change cannot explain the move there. One run is not a measurement.
- **The best config was a lucky pick.** Four noisy scores on the same 24 clips gave a different winner each time, and the critic result changed sign. Choose on one set and measure on another.
- **A high recall can be a bias.** With reasoning off, nano-omni found 11 of 12 impossible clips, but it called 9 of 12 plausible clips impossible too. It did this in both runs.
- **Right answer, wrong reason.** Most correct "impossible" answers of the 120B model gave the wrong evidence. Several answers described a ball that splits in two, which happens in no clip.
- **Reasoning needs a token budget.** The 120B model gave no verdict on 5 of 24 clips. All 5 replies stopped at exactly 4,000 tokens.
- **Latency is mostly the service.** With reasoning off, 23 of 24 replies of the 120B model took 20 to 29 seconds, from 8 to 209 tokens. One reply of 18 tokens took 1.4 seconds. This looks like a queue and not model speed.

## Limits

24 simple 2D clips, two runs, one question.
The hosted service shares its load, so latency is not a model property.
The two runs differ in the parser, in the session type and in the load on the service.
