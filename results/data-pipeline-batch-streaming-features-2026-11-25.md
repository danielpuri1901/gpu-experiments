# Data pipeline, streaming, features and labels, 2026-11-25

One full run of `data-pipeline-batch-streaming-features.ipynb` (commit `5442374`) in the Kaggle notebook of the same name (Version 2, CPU, 3.1 minutes). All 14 parts ran.
Kafka 4.2.2 (one real broker in KRaft mode), Feast 0.68.0 and Spark 4.3.0 are real. All data is generated with a known truth.
Version 1 found two faults: Spark did not start on the small Java of `jdk4py` (now Java 17), and the Kafka download from the Apache archive took 9 minutes (now a mirror, 46 seconds for the whole part).

## Three shapes of data

| Shape | Parser | Records fully right | Lines of code |
|---|---|---|---:|
| Structured (SQL table) | SELECT | 1.00 | 1 |
| Semi-structured (nested JSON) | strict | 0.81 (0.77 to 0.84) | 5 |
| Semi-structured (nested JSON) | tolerant | 0.96 (0.94 to 0.97) | 6 |
| Unstructured (email text) | regex v1, v2, v3 | 0.62, 0.80, 1.00 | 4, 8, 2 more |

A revenue query on the raw JSON was 13.6% too low. The raw email text could not answer it.

## ETL

20,000 records, 12% broken on purpose. Every broken record went to the dead-letter queue with the right reason, and no clean record did.

| Run | Without an idempotency key: duplicate rows | With a key |
|---|---:|---:|
| Finished | 600 | 0 |
| Killed half way, then run again | 1,600 | 0 |

No record was lost. ELT keeps the raw data in the target, so a rule change is one SQL statement.

## Exploratory data analysis: five planted faults

AUC 0.619 with all faults and 0.885 with all fixes. The missing values hurt most: fixing only them gives 0.827, and fixing all but them gives 0.770.
Accuracy hides the imbalance: an always-negative model scores 0.969 with recall 0.
The leaking column gave AUC 0.995 on the training rows and 0.619 in production.

## Features (two missed exam questions)

| Target mean per group | Cross-validation R2 | Next month R2 |
|---|---:|---:|
| Built on all rows | 0.272 | 0.091 |
| Built inside each training fold | 0.072 | 0.102 |

| Eight columns with gaps, the pattern predicts the target | Test R2 |
|---|---:|
| Drop rows with a gap | -0.233 |
| Impute only | 0.049 |
| Impute and one any-missing flag | 0.062 |
| Impute and 8 flags (tree model) | 0.532 |
| Impute and one integer mask (tree model) | 0.534 |
| Linear model: 8 flags, then the target mean of the mask | 0.034, then 0.598 |

## Feature store

| Join | Rows that use a future value | Cross-validation AUC | AUC at serving |
|---|---:|---:|---:|
| Naive, latest snapshot | 100% | 0.814 | 0.577 |
| Point-in-time | 0% | 0.673 | 0.607 |

Feast gave the same 4,000 of 4,000 rows as an independent point-in-time join.
Training-serving skew: the online path computed the feature from the last 3 orders. 74% of the rows differed, and AUC fell from 0.607 to 0.497 (a guess).
One online read took 0.2 ms in Feast, and one historical read 173 ms.

## Labels

| Labels | Label accuracy | Classifier test accuracy |
|---|---|---|
| Majority vote of 5 annotators | 0.745 | 0.695 |
| Best single annotator | 0.784 | 0.735 |
| Dawid-Skene | 0.835 | 0.758 |
| Truth | 1.000 | 0.805 |

Annotator 5 is wrong 71% of the time. Majority vote counts that vote, and Dawid-Skene turns it around.
Weak rules labelled 93% of the rows with 0.68 accuracy. Active learning against random choice: +0.027 at 100 labels, interval -0.013 to +0.064, so no clear gain.

## Text preprocessing

The standard scikit-learn stop-word list cut accuracy from 0.800 to 0.639, because it removes "not". The same list without negations gave 0.808 with 16% fewer tokens.
MinHash found planted duplicates with precision 1.00 and recall 0.96.

## Batch against streaming

| Kafka, consumer crash and resume, 2,000 records | Append sink | Upsert sink |
|---|---|---|
| At-least-once | 50 duplicates | exact |
| At-most-once | 50 lost | 50 lost |

| Event-time windows with late events (last 3 hours) | Windows exact | Mean age of a result |
|---|---:|---:|
| Watermark 0 s | 0.03 | 49 s |
| Watermark 300 s | 0.06 | 387 s |
| Watermark 900 s | 0.11 | 1,039 s |
| Watermark 1,800 s | 1.00 | 1,953 s |
| Hourly batch recompute | 1.00 | 3,450 s |

| Rows | pandas s | Spark local, 4 cores, s | Same result |
|---:|---:|---:|---|
| 100,000 | 0.03 | 2.36 | yes |
| 2,000,000 | 0.17 | 3.84 | yes |
| 20,000,000 | 1.13 | 15.29 | yes |

Spark needed 12 seconds to start.

## Rules on data

Retention, sovereignty, bias (no proxy feature) and purpose limits each have a test that passes with the rule on and fails with the rule off. Without the retention rule on a derived table, 4,609 old rows stayed.

## What the numbers show

- **Any feature that uses the target must be built inside the training fold.** Validation looked 3 times better than the next month.
- **The pattern of missing values can be the signal.** Dropping rows deletes it.
- **Point-in-time joins lower the offline score and raise the real one.**
- **Exactly-once results need an idempotent sink.** The broker only promises delivery at least once or at most once.
- **Streaming trades freshness for correctness.** The watermark must cover the latest late event, here 30 minutes.
- **Spark on one machine is slower than pandas** at all three sizes. It pays when the data does not fit on one machine.
- **Preprocessing can remove the signal.** Measure each step.

## Limits

Generated data, so effects are larger and cleaner than in real data. Annotator errors and late events follow fixed distributions. One Kafka broker and one Spark machine.
