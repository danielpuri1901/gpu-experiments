# Sim-to-real, domain randomization and edge inference, 2026-11-24

One full run of `sim-to-real-domain-randomization.ipynb` (commit `d59b71e`) in the Kaggle notebook of the same name, on a Tesla T4 with NVIDIA Warp 1.19.0.
The run chain stopped once after the simulator check with no error, so it was resumed from the next cell in the same kernel.
The "real" machine is a second set of hidden physics in the same simulator. It is not hardware.

## Simulator check

Warp on the GPU and the NumPy reference differ by at most 6.3e-07 over 6 worlds and 50 steps.
Energy drifts 2.04% of the swing energy in 2 seconds.
A replay with the true delay has an error of 0, and a replay with one extra step of delay does not.

## Simulation speed (world-steps per second)

| Worlds | NumPy | Warp on CPU | Warp on GPU | GPU against NumPy |
|---:|---:|---:|---:|---:|
| 1 | 9.0 thousand | 1.93 million | 0.16 million | 18x |
| 64 | 0.50 million | 4.69 million | 9.56 million | 19x |
| 1,024 | 3.0 million | 4.78 million | 113 million | 38x |
| 4,096 | 4.1 million | 4.77 million | 452 million | 109x |
| 16,384 | 4.6 million | 4.79 million | 1,184 million | 256x |

With one world, the GPU is 12 times slower than the same kernel on the CPU.
The Warp times cover the kernel only, and not the copy of arrays to and from the device.

## The gap (5 seeds, same training budget)

| Training worlds | Success in its own simulation | Success on the real machine |
|---|---:|---:|
| Nominal physics only | 1.00 | 0.14 |
| Narrow randomization around a wrong guess | 1.00 | 0.22 |
| Wide domain randomization | 0.56 | 0.82 |

The wide box holds machines that are harder than the real one, so its own score is lower than its real score.

## Sim-to-real flywheel (mean of 5 seeds)

| Turn | Real trajectories | Real success | Control: more wide training, no real data |
|---:|---:|---:|---:|
| 0 | 0 | 0.825 | 0.825 |
| 1 | 8 | 0.924 | 0.896 |
| 2 | 16 | 0.942 | 0.936 |
| 3 | 24 | 0.944 | 0.948 |

| Turn | Pole mass error | Friction error | Motor gain error | Delay error (steps) |
|---:|---:|---:|---:|---:|
| 1 | 15% | 16% | 1.3% | 0.18 |
| 2 | 19% | 14% | 1.3% | 0.06 |
| 3 | 21% | 12% | 2.0% | 0.15 |

## Edge inference

One decision takes 0.007 ms (p50) and 0.020 ms (p99) on one CPU thread. The control period is 20 ms.

| Extra action delay | Success on the real machine |
|---:|---:|
| 0 ms | 0.94 |
| 20 ms | 0.84 |
| 50 ms | 0.13 |
| 100 ms | 0.00 |
| 200 ms | 0.00 |

## Newton

The built-in `basic_pendulum` example of Newton ran headless on the T4 with exit code 0.

## What the numbers show

- **A perfect simulation score says little about the real machine.** Training on one set of physics scored 1.00 in simulation and 0.14 on the real machine.
- **A narrow wrong guess is nearly as bad.** Randomization helps when the real machine is inside the box.
- **Wide domain randomization closed most of the gap without real data.** Success rose from 0.14 to 0.82.
- **The flywheel was faster, but not better in the end.** With 8 real trajectories it reached 0.92, against 0.90 for the control. After three turns the control, with no real data, had 0.948 against 0.944. The gain came mostly from more training. With 5 seeds these gaps are small.
- **System identification does not find every parameter.** Motor gain was within 2%, but the pole mass error grew from 15% to 21%. Cart motion says little about the pole mass, so more data of the same kind did not help.
- **Control must run on the robot.** The decision needs 0.1% of the control period. An extra round trip of 50 ms cut success from 0.94 to 0.13.
- **Parallel worlds are the reason to use a GPU.** It pays off from 64 worlds and up.

## Limits

A cart-pole is a small task.
Task difficulty (push size and the hidden machine) was chosen so that the methods separate.
Isaac Sim does not run on this GPU, so there is no rendered sensor data.
