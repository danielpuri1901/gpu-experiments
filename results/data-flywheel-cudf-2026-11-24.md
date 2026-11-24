# RAPIDS cuDF, bad data and a data flywheel, 2026-11-24

One full run of `data-flywheel-cudf.ipynb` (commit `f51d3cb`) in the Kaggle notebook of the same name, on 2x Tesla T4.
All cells ran in order in one kernel, in 2.6 minutes.

A first run stopped after part 1 with "Kernel died unexpectedly".
A second run in a fresh session finished with no change to the code.
The cause of the first stop is not known.

## Part 1: cuDF pandas accelerator

The same pandas script ran as `python bench.py` (CPU) and as `python -m cudf.pandas bench.py` (GPU).
Both runs gave the same checksum for every step.

| Step | 100k rows CPU s | 100k rows GPU s | Speed-up | 10M rows CPU s | 10M rows GPU s | Speed-up |
|---|---:|---:|---:|---:|---:|---:|
| Group-by aggregation | 0.008 | 0.034 | 0.2x | 0.687 | 0.069 | 9.9x |
| Join | 0.018 | 0.012 | 1.6x | 3.637 | 0.127 | 28.7x |
| Sort | 0.025 | 0.018 | 1.4x | 9.173 | 0.860 | 10.7x |
| Filter and derived columns | 0.010 | 0.016 | 0.6x | 1.228 | 0.061 | 20.0x |
| String contains | 0.092 | 0.006 | 15.4x | 8.681 | 0.059 | 146.9x |
| Rolling mean and quantile | 0.020 | 0.031 | 0.7x | 2.434 | 0.681 | 3.6x |
| Total | 0.17 | 0.12 | 1.5x | 25.84 | 1.86 | 13.9x |

The first run gave 16.6x at 10M rows, so the total moves by a few points between runs.
The profiler listed `Timestamp`, `to_timedelta` and the timestamp addition as calls that fell back to the CPU.

## Part 2: data quality checks (AG News, 12,000 rows, injected faults)

| Fault | Injected | Flagged | Precision | Recall |
|---|---:|---:|---:|---:|
| Missing text | 360 | 360 | 1.00 | 1.00 |
| Exact duplicate | 600 | 600 | 1.00 | 1.00 |
| Near duplicate | 600 | 703 | 0.85 | 1.00 |
| Stale row | 600 | 600 | 1.00 | 1.00 |
| PII | 480 | 480 | 1.00 | 1.00 |
| Wrong label | 1,800 | 2,014 | 0.71 | 0.79 |

The readiness gate failed on the faulty data and passed on the cleaned data.
After redaction, no PII row matched the patterns.

## Part 3: effect on accuracy

| Training data | Rows | Test accuracy |
|---|---:|---:|
| A, clean sample | 12,000 | 0.893 |
| B, faulty data | 13,200 | 0.822 |
| C, faulty data after cleaning | 9,007 | 0.882 |

| Test set | Accuracy |
|---|---:|
| Honest, 2,000 rows | 0.897 |
| Leaky, 1,000 honest and 1,000 copied from training | 0.944 |
| Leaky set after the copies are removed | 0.891 |

## Part 4: flywheel, 250 labels per turn

| Arm | Turn | General benchmark | App-specific eval |
|---|---:|---:|---:|
| Failure mining and human labels | 0 | 0.749 | 0.604 |
| | 1 | 0.805 | 0.777 |
| | 2 | 0.811 | 0.827 |
| | 3 | 0.825 | 0.824 |
| Random rows and human labels | 1 | 0.790 | 0.776 |
| | 2 | 0.805 | 0.807 |
| | 3 | 0.811 | 0.818 |
| Same failures, labelled by the model itself | 1 | 0.709 | 0.443 |
| | 2 | 0.671 | 0.353 |
| | 3 | 0.643 | 0.295 |

## Part 5: monitoring, drift starts at batch 8

| Signal | Batches until alert | False alarm before drift |
|---|---:|---|
| Mean confidence | 3 | no |
| Thumbs-down rate | 3 | no |
| Accuracy from delayed labels | 7 | no |
| Drift measure (PSI of the predicted classes) | 12 | no |

## What the numbers show

- **The GPU pays off on big tables.** At 10 million rows the six steps ran 13.9 times faster. At 100 thousand rows three of six steps were slower on the GPU, because fixed costs dominate.
- **Rules find rule-shaped faults.** Missing text, exact copies, old dates and PII patterns reached precision and recall of 1.00. Wrong labels need a model, and that check found 79% with 71% precision.
- **Bad data cost 7 points of accuracy.** Cleaning gave back 6 of them with 25% fewer rows than the clean set.
- **Leakage flatters the score.** Copies in the test set added 5 points that are not real.
- **The flywheel raised the app score from 0.604 to 0.824.** The general benchmark rose only from 0.749 to 0.825, so it understated the gain. It was not blind, as the notebook asked: it moved in the same direction.
- **Failure mining beat random sampling by less than 2 points.** With one seed, that gap can be noise.
- **A model must not label its own failures.** The pseudo-label arm fell from 0.604 to 0.295, because about 97% of the rows it picked were wrong. This is the feedback loop.
- **Signals without labels alert first.** Confidence and thumbs-down alerted after 3 batches, against 7 for accuracy with delayed labels.

## Limits

One seed and simulated users, labels and drift.
The wording change is an artificial transform.
A benchmark of one script is not a general speed-up figure for cuDF.
