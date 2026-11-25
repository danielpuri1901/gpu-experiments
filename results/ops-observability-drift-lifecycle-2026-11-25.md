# Observability, drift, lifecycle and governance, 2026-11-25

One full run of `ops-observability-drift-lifecycle.ipynb` (commit `51b4214`) in the Kaggle notebook of the same name (Version 2, CPU, 81 seconds). All 9 parts passed.
Version 1 failed in the governance part: MLflow returns versions with the newest first, and the check expected them in order. The fix sorts them.
Everything runs on one simulated classifier service with labels that arrive 7 days late. MLflow 3.18.0 and OpenTelemetry SDK 1.42.1 are real.

## Observability

A trace tree showed the slow span (retrieval, 1,159 ms against a median of 42 ms) and the failed span (tool) in 28 of 28 failed requests.
Anomaly detection on latency, with the setting chosen on the first half and measured on the second half (3 incidents):

| Detector | Alarm episodes | Precision | Recall | Delay, requests |
|---|---:|---|---:|---:|
| Fixed threshold | 7 | 57% (25 to 84) | 3 of 3 | 0 |
| EWMA | 3 | 100% (44 to 100) | 3 of 3 | 2 |

## Drift: which detector sees what

Days from the drift start to the alarm, median of 5 seeds (loose threshold). A dash means no alarm.

| Drift | Accuracy change | PSI 0.1 | KS 0.05 | Prediction share | Delayed labels |
|---|---:|---|---|---|---|
| Gradual covariate | -0.147 | 5 | 3 | 5 | 21 |
| Sudden covariate | -0.284 | 1 | 1 | 1 | 8 |
| Concept (labels change, inputs do not) | -0.199 | - | 1 of 5 seeds | 1 of 5 seeds | 8 |
| Irrelevant column | +0.001 | 1 | 1 | 3 of 5 seeds | - |

Alarms on healthy days were 0 to 1.2% for every detector (850 days).

## Retraining triggers

Accuracy over 60 days, 3 seeds. "Bad" counts retrains that lowered the accuracy.

| Trigger | Gradual | Sudden | Concept | Faulty labels | Retrains | Bad |
|---|---:|---:|---:|---:|---:|---:|
| Never | 70.1 | 65.6 | 69.8 | **79.8** | 0 | 0 |
| Weekly | **77.3** | **75.6** | **76.3** | 76.1 | 18 in each | 3 (faulty labels) |
| Drift alert (PSI) | 76.9 | 75.5 | 69.8 | 79.8 | 12, 12, 0, 0 | 0 |
| Performance alert | 75.3 | 74.0 | 75.1 | 71.1 | 5 to 6 | 3 (faulty labels) |

## Release and governance

Version 3 has a planted skew bug: two columns are swapped in training and in the offline test, and not in serving.

| Stage | Version 2 (good) | Version 3 (bug) |
|---|---|---|
| Offline test | 0.88, pass | 0.88, pass |
| Shadow, positive-share gap | 0.128, pass | 0.001, pass |
| Canary, against the champion | 0.896 against 0.647, promote | 0.665 against 0.926, rollback |
| Wrong answers added | 60 fewer | 60 more (a full switch: 730 more) |

- Lineage: from one prediction ID back to the version, run, parameters, data hash and 2,400 source rows. All three hashes match.
- Tamper test: one value changed by 0.001. The data hash differs and the row hashes name the row.
- Deletion request: one customer has 17 rows, and versions 1, 2 and 3 trained on them.
- Audit log: the hash chain finds an edit, an edit with a repaired hash, and a deleted entry. A cut tail passes the chain. Only a head hash that is stored elsewhere finds it.

## Runbooks and GPU sharing

| | Automated runbook | Manual (assumed human delays) |
|---|---:|---:|
| MTTD, min | 2.0 | 8.8 |
| MTTR, min | 7.8 | 38.1 |

| Scheduler, 48 jobs on 8 GPUs | Mean wait, min | Useful GPU share | Quota broken, job-min | Cost per useful GPU-hour |
|---|---:|---:|---:|---:|
| Static partition | 84.3 | 63.1% | 0 | 1.59 |
| Over-quota, no preemption | 52.8 | 74.5% | 27 | 1.34 |
| Fair share: over-quota and preemption | 57.0 | 73.6% | 0 | 1.38 |
| Fair share and GPU fractions | 36.4 | 82.2% | 0 | 1.22 |

## What the numbers show

- **Input drift and accuracy loss are two different things.** PSI and KS alarmed on a column that the model ignores, and stayed silent for concept drift.
- **Only delayed labels see concept drift**, and they need at least the label delay.
- **A retrain trigger is no better than its labels.** With faulty labels the performance trigger lost 9 points and never retraining was best. Check the data before a retrain.
- **An offline test cannot see training-serving skew.** Shadow did not see it either, because the share of positive answers stayed the same. Only the canary with real labels did.
- **Preemption is what lets a team borrow idle GPUs and keep its quota.**

## Limits

A simulated service, a linear model, 3 to 5 seeds. Manual recovery times are stated assumptions, so the 4.9x ratio is an assumption too.
The batch, sequence and precision numbers come from a CPU. The float16 time is slow there because the CPU has no fast float16 path.
