# Sizing and capacity plan, 2026-11-23

One run of `sizing-capacity-plan.ipynb` on a Kaggle CPU session.
It uses the measured sweeps in `sweeps-2026-11-23.csv`, so no GPU is needed.

## Inputs

| Knob | Value |
|---|---|
| SLA | TTFT p99 at most 2,000 ms and TPOT median at most 50 ms |
| Users | 500 active users, 60 s think time between requests, 128 output tokens per answer |
| GPU price | 0.50 USD per GPU-hour. This is an assumption, not a quote. A month is 730 hours. |

SLA: service level agreement, the latency limit promised to users.
The SLA concurrency is the highest number of requests in flight that still meets both limits.
It comes from linear interpolation on log2(concurrency) between measured points.

## Results

| Config | SLA concurrency per replica | Requests in flight (L) | Replicas | GPUs | USD per 1M output tokens | USD per month | USD per user per month |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5B fp16, 1 T4 | 17.2 | 37.7 | 3 | 3 | 0.29 | 1,095 | 2.19 |
| 7B fp16, TP=2 on 2 T4 | 7.9 | 50.3 | 7 | 14 | 1.66 | 5,110 | 10.22 |
| 7B AWQ, 1 T4 | 5.7 | 42.0 | 8 | 8 | 0.92 | 2,920 | 5.84 |

The throughput check passes for all three configs.
For example, the AWQ fleet must deliver 977 tokens/s and can deliver 1,202 tokens/s.

## How one row is computed (AWQ)

1. The TTFT p99 limit is hit first, at concurrency 5.68. There the median TTFT is 976 ms and the TPOT is 35.3 ms.
2. Request time W = 0.976 s + 128 x 0.0353 s = 5.50 s.
3. Little's law for a closed system: arrival rate = 500 users / (60 s think + 5.50 s) = 7.63 requests/s. Requests in flight L = 7.63 x 5.50 = 42.0.
4. Replicas = ceil(42.0 / 5.68) = 8, on 8 GPUs.
5. Cost per 1M output tokens = 0.50 USD / (150 tokens/s x 3,600 s) x 1,000,000 = 0.92 USD.

## What the numbers show

- The latency target, not raw throughput, sets the replica count. Each replica is limited to the concurrency where TTFT p99 stays under 2 s.
- For the 7B model, horizontal scaling with AWQ needs 8 GPUs and vertical scaling with TP=2 needs 14. AWQ meets the SLA at a lower concurrency per replica, but each replica uses one GPU instead of two.
- Model size drives cost. The 1.5B model costs about a third as much per token as 7B AWQ. Whether it is good enough is a quality question that this notebook does not answer.
- Monitoring drill (synthetic data): with a rolling 5-minute p99 TPOT alert that needs 3 failed checks in a row, MTTD was 2 minutes. With a 10-minute scale-out, MTTR was 12 minutes. MTTD: mean time to detect. MTTR: mean time to recover.

## Limits

The latency curves come from single runs with 16 to 256 requests per point.
Interpolation between points 4x apart is rough.
The price is an assumption, so compare the configs with each other, not with a real bill.
