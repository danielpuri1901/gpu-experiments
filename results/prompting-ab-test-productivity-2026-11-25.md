# Prompting, A/B tests and productivity tasks, 2026-11-25

One full run of `prompting-ab-test-productivity.ipynb` (commit `ebe77d0`). It ran on a laptop from the same notebook cells, because the hosted NVIDIA API throttles Kaggle's shared network address.
1,354 of 1,361 calls returned an answer. Seven calls to the 30B model were throttled. It took 13 minutes and all 6 parts passed.
A Kaggle run (commit `a975a46`) gave the same picture in parts A and B, and its prompt-optimization part is not used, because calls were refused in the last rounds.

Models: `nemotron-3.5-bolt-e4b` (4B) and `nemotron-3.5-lightning-30b` (30B), both with reasoning off. Ultra wrote the prompts of the optimizer.
90 fictional office tasks with exact graders: 15 each for email triage, action items, document questions, summary, SQL and log diagnosis. 48 dev items and 42 test items.

## Zero-shot by task

| Task | 4B | 30B |
|---|---:|---:|
| Email triage | 1.00 | 1.00 |
| Meeting notes to action items | 0.53 | 0.20 |
| Document questions | 0.60 | 0.60 |
| Summary with given facts | 1.00 | 0.47 |
| Question to SQL | 0.67 | 0.80 |
| Log diagnosis | 0.93 | 0.87 |
| All | 0.79 (0.69 to 0.86) | 0.66 (0.55 to 0.75) |

## Prompt techniques (4B model)

| Technique | Dev | Test | All | Output tokens | p against zero-shot |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 0.75 | 0.83 | 0.79 | 29 | |
| Few-shot, 1 | 0.79 | 0.81 | 0.80 | 27 | 1.00 |
| Few-shot, 3 | 0.77 | 0.76 | 0.77 | 25 | 0.80 |
| Few-shot, 5 | 0.79 | 0.88 | 0.83 | 26 | 0.34 |
| Chain of thought | 0.83 | 0.93 | 0.88 | 145 | 0.12 |
| System prompt with rules | 0.77 | 0.83 | 0.80 | 31 | 1.00 |
| Structured output (JSON schema) | 0.54 | 0.60 | **0.57** | 42 | below 0.001 |

Zero-shot twice: 0.79 and 0.77. Chain of thought twice: 0.88 and 0.84. The API is not fully repeatable at temperature 0.

## Automatic prompt optimization

| Round | Dev | Test |
|---|---:|---:|
| 0, no system prompt | 0.75 | 0.83 |
| 1 | **0.79** (picked) | 0.79 |
| 2 | 0.71 | 0.81 |
| 3 | 0.69 | 0.83 |
| 4 | 0.75 | 0.83 |

The best dev round was not better on the test items (0.79 against 0.83, p = 0.75).

## Offline experiments

- Zero-shot to chain of thought: gain 0.09, paired interval 0.00 to 0.19. The gate (lower end above 0) fails, so the offline data does not prove a gain. The unpaired interval is wider (0.21 against 0.19).
- Golden set size for power 0.80: a gain of 0.15 needs 90 items, 0.10 needs 250, 0.05 needs 1,000, and 0.02 needs more than 1,000.
- A change can win overall and lose a slice: chain of thought gave +0.09 overall, and summary fell from 1.00 to 0.67.

## Online A/B test (simulated from the measured rates)

| Question | Result |
|---|---|
| Users per arm to see a gain of 0.05 with power 0.80 | 948 |
| False positive rate with 1 look, 10 looks, 20 looks (A/A test) | 0.048, 0.195, 0.253 |
| Same with a Bonferroni correction | 0.052, 0.026, 0.019 |
| Sample ratio mismatch alarm: clean split, logging bug | 0.0015, 0.998. The bug makes a fake lift of +0.017 |
| Novelty effect: 3-day test against the steady lift | +0.063 against +0.020 |
| Guardrails on chain of thought | success passes, p95 latency 4.5x and tokens 4.9x fail: do not ship |
| Simpson's paradox | With a biased split the new prompt wins overall and loses in both segments |
| A bad version, users exposed: shadow, canary 5%, A/B | 0, 304, 10,000 |

## Business value (minutes saved per right answer and weekly volumes are assumptions)

Chain of thought saves 62 hours a week with free repairs and 36 hours if a failure costs 30 minutes. Summary then costs time (0.67 success is below its break-even of 0.83).

## What the numbers show

- **A small model can beat a bigger one on narrow tasks.** The 4B model with reasoning off scored 0.79 against 0.66.
- **Forcing a JSON schema is not free.** Success fell from 0.79 to 0.57 on this model. Check format validity and the answer together.
- **Chain of thought looked best, at 5 times the tokens, and the gain is not proven** with 90 items.
- **An automatic optimizer overfits a small dev set.** Keep a test set that it never sees.
- **Peeking inflates false positives.** Ten looks at an A/A test give 20%, not 5%.
- **Judge a change on slices and guardrails,** not on the average.

## Limits

Fictional items, 15 per task, two models, one language. The online part is a simulation from measured rates. Business numbers are assumptions.
