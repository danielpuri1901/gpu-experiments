# Kubernetes model service, 2026-11-24

Run 3 of the `k8s-model-service` workflow (commit `ae37a83`, [run 45405932583](https://github.com/danielpuri1901/gpu-experiments/actions/runs/45405932583)) on a GitHub-hosted runner.
All 14 checks passed.

Runs 1 and 2 passed 13 of 14 checks.
Both failures were faults in the tests:

- The autoscaler was part of the base manifests. Each test applied it again, and it scaled from 3 to 6 pods within 0.2 seconds, on the CPU load of the test before. The controller log shows it. The autoscaler now has its own file, and only test 4 applies it.
- `awk` printed epoch times with 6 digits, so the latency windows of test 4 in run 1 were wrong.

## Results

| Test | Measured |
|---|---|
| Scale 3 to 6 with a 15 s model load, readiness probe on | 0 of 800 requests failed |
| Same, without a readiness probe (control) | 134 of 800 requests failed |
| Delete a pod under load | Back to 3 ready pods in 6.7 s, 0 of 1,600 requests failed |
| Break a pod's health check | Liveness restarted it in 12.4 s |
| Rolling update v1 to v2 under load | 0 of 1,500 requests failed |
| Bad release v3 that never gets ready | Rollout stalled, 3 pods stayed ready, v3 served 0 requests, 0 of 1,200 failed, undo worked |
| Autoscaling on CPU load | First scale-out after 30.5 s, 3 to 6 pods, p50 latency 158 ms before and 78 ms after |
| Scale-in | Back to 3 pods 88 s after the load stopped (60 s stabilisation window) |
| Canary by mesh weight 90/10 | v2 share 0.112 over 500 requests |
| Canary 50/50 and 0/100 | v2 share 0.496 and 1.000 |
| Outlier detection, one poisoned pod | 145 of 300 requests failed without it, 3 of 300 with it |
| mTLS STRICT | Plain-text call from outside the mesh: 200 before, connection reset after. Call from inside: 200. |

## What the numbers show

- **Readiness decides who gets traffic.** Without the probe, 17% of requests failed while new pods loaded the model.
- **A controller repairs the state.** A deleted pod was back in 7 seconds, and no request failed.
- **A safe rollout fails safe.** With `maxUnavailable: 0`, the bad release never replaced a good pod.
- **Autoscaling is slow compared with a request.** The first new pods came after 30 seconds. An autoscaler also acts on old metrics, as the test fault shows.
- **A mesh routes by weight and not by pod count.** It also ejects a failing pod and can refuse plain text.

## Limits

One runner with 4 vCPU, one passing run, and a Python server that stands in for a NIM.
The server burns CPU. A real NIM would scale on a GPU metric.
Latency numbers are noisy.
