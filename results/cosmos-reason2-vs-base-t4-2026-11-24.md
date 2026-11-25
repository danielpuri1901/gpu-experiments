# Cosmos-Reason2-2B against its base model, 2026-11-24

One full run of `cosmos-reason2-vs-base-t4.ipynb` (commit `303d5cf`) in the Kaggle notebook of the same name, on 2x Tesla T4.
It ran as a server-side batch run (Version 1) and took 38 minutes, with no errors.

**The post-training shows no measurable accuracy gain on this task.** On the clips where both models answered, Cosmos was not significantly better than its base in any mode.
**The post-training does change behavior.** In the plain mode Cosmos finishes every reply, and in the think mode it flags impossible clips. The base model flags none.

## Setup

NVIDIA post-trained Cosmos-Reason2-2B from Qwen3-VL-2B-Instruct with the same architecture. So the two models differ only by the post-training.
vLLM 0.34.0 served each model on its own T4 in fp16 (the T4 has no bf16), at the same time, with 8 requests in flight.
120 clips of 3 seconds come from a 2D physics simulation: 60 plausible, 60 impossible, 7 kinds of violation. The simulator gives the exact answer, and a check proves each clip.
Both models got the same video, the same prompt and temperature 0. A text question and a still frame showed no obvious breakage in fp16.

Three modes:

- **plain**: is the motion plausible or impossible? At most 800 tokens.
- **opposite**: does the motion break physics, yes or no? It tests answer bias.
- **think**: the plain question with a `<think>` block. At most 3,072 tokens.

A reply that hits the token limit counts as "cut".

## Results

| Model | Mode | Correct | Wrong | Cut | Accuracy on answered (95% interval) | Accuracy, cut = wrong | Recall, impossible | False alarm | Says plausible | Median s | Mean tokens |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Cosmos-Reason2-2B | plain | 60 | 60 | 0 | 0.50 (0.41 to 0.59) | 0.50 | 0.00 | 0.00 | 1.00 | 2.8 | 45 |
| Qwen3-VL-2B (base) | plain | 53 | 51 | 16 | 0.51 (0.41 to 0.60) | 0.44 | 0.00 | 0.00 | 1.00 | 4.4 | 187 |
| Cosmos-Reason2-2B | opposite | 68 | 52 | 0 | 0.57 (0.48 to 0.65) | 0.57 | 0.63 | 0.50 | 0.43 | 1.8 | 3 |
| Qwen3-VL-2B (base) | opposite | 63 | 57 | 0 | 0.52 (0.44 to 0.61) | 0.52 | 0.20 | 0.15 | 0.82 | 1.8 | 2 |
| Cosmos-Reason2-2B | think | 43 | 34 | 43 | 0.56 (0.45 to 0.66) | 0.36 | 0.50 | 0.38 | 0.56 | 53.8 | 1,536 |
| Qwen3-VL-2B (base) | think | 48 | 50 | 22 | 0.49 (0.39 to 0.59) | 0.40 | 0.00 | 0.00 | 1.00 | 7.4 | 665 |
| Always plausible | baseline | 60 | 60 | 0 | 0.50 (0.41 to 0.59) | 0.50 | 0.00 | 0.00 | 1.00 | | |

### Paired tests (exact McNemar)

| Pair | Clips | Cosmos right alone | Base right alone | p |
|---|---:|---:|---:|---:|
| plain, both answered | 104 | 0 | 0 | 1.00 |
| plain, cut counts as wrong | 120 | 7 | 0 | 0.016 |
| opposite | 120 | 26 | 21 | 0.56 |
| think, both answered | 65 | 18 | 13 | 0.47 |
| think, cut counts as wrong | 120 | 23 | 28 | 0.58 |

The row with p = 0.016 is not a skill gap.
Both models say "plausible" for every clip in the plain mode. The 7 clips are plausible clips where the base reply was cut.
The notebook prints "Cosmos is ahead" for that row. Read the row above it.

### Does the answer follow the wording?

| Model | Clips with both answers | Verdict flips | Flip rate | Right in both wordings |
|---|---:|---:|---:|---:|
| Cosmos-Reason2-2B | 120 | 68 | 0.57 | 0.25 |
| Qwen3-VL-2B (base) | 104 | 20 | 0.19 | 0.42 |

A flip means the plain question and the opposite question gave different verdicts for the same clip.
Right in both wordings can only be plausible clips, because the plain verdict was always "plausible".

### Impossible clips found in the think mode (Cosmos)

| Violation | Found of answered |
|---|---:|
| energy_gain | 4 of 4 |
| spontaneous_motion | 4 of 5 |
| gravity_reversal | 3 of 4 |
| teleport | 4 of 7 |
| pass_through | 2 of 4 |
| vanish_return | 2 of 8 |
| amplitude_growth | 0 of 6 |

The base model found 0 of 50 answered impossible clips in the same mode.

## What the numbers show

- **Post-training changed the behavior and not the accuracy.** In the think mode Cosmos called 34 of 77 clips impossible. The base called 0 of 98. But 19 of those 34 calls were right and 15 were false alarms, so the accuracy interval still holds 0.5.
- **Accuracy alone hides a one-answer model.** A model that always says "plausible" scores 0.50. Look at the share of one answer, at recall and at false alarms.
- **Ask the question in opposite words.** Cosmos changed its verdict on 57% of the clips when only the wording changed. A judge that follows the wording is not a judge. Its 0.57 in the opposite mode comes with a 0.50 false alarm rate.
- **A cut reply can fake a win.** Count cut replies as their own outcome, and compare models only on clips where both answered.
- **Thinking was not worth its cost here.** It took 34 times the tokens of the plain mode. The accuracy on answered clips rose from 0.50 to 0.56, inside the noise, and 43 of 120 replies were cut. With cut replies counted as wrong, thinking was worse than a guess.
- **A critic that keeps everything has a perfect yield.** The notebook chose the plain mode for both critics, because it counts cut replies as wrong. Cosmos then kept every clip: purity 0.70 before and after, yield 1.00. From the counts of this run, a think-mode Cosmos critic would keep about 40% of the good clips and 32% of the bad ones, which lifts purity from 0.70 to about 0.75. That number was not measured on held-out clips.

## Repeat check

A first interactive run had two faults.
The parser rejected short answers such as "yes" and "No" (24 of 78 replies), and the base model had a limit of 300 tokens, which made a false win (p = 0.008).
Two of the runs that did not depend on these faults matched exactly: Cosmos opposite gave 68 and 52 both times, and base think gave 48, 50 and 22 both times.
Base opposite moved from 60 to 63 correct of 120.
The local server at temperature 0 is far more repeatable than the hosted API, where the same clips moved scores by 7 to 18 points (see `cosmos-physical-check-2026-11-24.md`).

## Limits

The clips are simple 2D shapes, not real or generated video. One fixed question, one run at temperature 0.
NVIDIA recommends a temperature of 0.6 to 0.7 for these models, and fp16 on a T4 is not a setting that NVIDIA tested.
Cosmos 3 models need bf16 and a newer GPU, so they do not run here.
The grounding check (does the evidence name the right violation?) gave 1 of 19 for the Cosmos think mode. Many final answers are only "No", and the check does not read the thinking text, so it is not a fair score.
The sample per violation type is 4 to 8 clips.
