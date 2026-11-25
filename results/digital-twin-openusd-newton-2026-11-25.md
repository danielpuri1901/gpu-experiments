# Digital twin with OpenUSD and Newton, 2026-11-25

One full run of `digital-twin-openusd-newton.ipynb` (commit `eef07b6`) in the Kaggle notebook of the same name (Version 4, T4, 13 minutes). All 6 parts passed, and 18 of 18 self-checks.
OpenUSD 26.11, Newton 1.8.0 and Warp 1.19.0 are real. The "real" machine in parts C and E is a Python function with hidden numbers, and the "real" images in part D come from a second renderer. None of it is hardware.

Versions 1 and 2 had a fault in part D. The randomized workpiece colours always moved all three channels the same way, and the test colours move some up and some down. So the randomization never covered the test range, and it scored 0%. The sampler now moves each channel on its own, and a self-check fails if this comes back. Version 1 also had too few training epochs, and 5 times more epochs did not help.

## A. The scene as data

The work cell is an OpenUSD stage: a table, an arm with a joint and a workpiece. An `override` layer moved the workpiece and made the arm payload heavy. The source layers stayed unchanged. The checker found 2 rigid bodies, 4 colliders and 1 joint, and the workpiece size matched the scene.

## B. Physics

| Check | Expected | Newton | Error |
|---|---:|---:|---:|
| Start height | 0.9500 m | 0.9500 m | 0.0% |
| Fall in 0.25 s (half g t squared) | 0.3066 m | 0.3078 m | 0.4% |
| Rest height | 0.4500 m | 0.4500 m | 0.0% |

| Copies of the scene | World-steps per second | Against one copy |
|---:|---:|---:|
| 1 | 232 | 1x |
| 64 | 12,946 | 56x |
| 1,024 | 244,145 | 1,052x |
| 4,096 | 887,040 | 3,823x |

## C. The twin follows a machine

The twin learns three friction and motor numbers from sensor values. Median error 0.5%, 0.9% and 2.6% (worst 8.5%).

| Method, 30 machines that wear out | False alarms | Detected | Median delay |
|---|---|---|---:|
| Twin, threshold 2.0 | 0 of 30 | 30 of 30 | 14.5 s |
| Raw sensor, threshold 2.0 | 0 of 30 | 30 of 30 | 23.5 s |
| Raw sensor, threshold 1.1 | 30 of 30 | 30 of 30 | 0.5 s |

- The twin alarm came 8.2 s earlier than the raw one at no false alarms (95% interval 5.4 to 11.0, 30 paired machines). Detection itself was equal.
- What-if: the healthy voltage (3.5 V) met the goal of 0.45 s on a healthy machine and missed it on a machine with doubled friction (0.55 s). The twin predicted 4.6 V, and the real machine then met 0.45 s.

## D. Synthetic data and the sim-to-real gap

A small CNN learns the pose of the workpiece from 6,000 simulated images. Success means under 2 cm and under 10 degrees. 400 images per test, 3 seeds (the success column is the mean over seeds).

| Training images | Simulated look (sim-clean): success, error | "Real" renderer: success, error |
|---|---|---|
| Fixed look | 100%, 0.3 cm | 0.4%, 11.5 cm |
| Domain randomization | 34%, 3.0 cm | 7.5%, 6.3 cm |
| Fit to 40 labelled real images | 74%, 1.6 cm | **79%, 1.4 cm** |
| Always the mean pose | 0%, 18.5 cm | 0%, 18.9 cm |

- The fixed-look network is perfect on its own simulation and useless on the real renderer (0.4% of 400).
- Domain randomization cut the position error on the real renderer by 5.2 cm [4.9, 5.6] against the fixed look. The gain in success is small (p = 0.016), because 2 cm is strict for this network. It lost accuracy at home (34% instead of 100%), because the same network now has a harder job.
- 40 labelled real images closed most of the gap (6.3 to 1.4 cm, p below 0.001). A label budget this small beat wide randomization.
- This does not say randomization is weak. The network is small, with 6,000 images. Notebook `sim-to-real-domain-randomization` shows the same idea with physics, where it raised success from 0.14 to 0.82.

## E. Three computers: train, simulate, run

| Where the policy runs | Extra delay | Success (300 machines, 2 runs) |
|---|---:|---|
| Edge, float | 0 ms | 100% |
| Edge, int8 | 0.2 ms | 100% |
| Cloud, +20 ms | 20 ms | 100% |
| Cloud, +50 ms | 50 ms | 59% and 54% |
| Cloud, +100 ms | 100 ms | 0% |

- The cliff is between 40 ms (100%) and 50 ms (57%). At 70 ms and more, success is 0%.
- int8 changes a command by at most 0.256 V of a 5 V limit. Success stayed 100%. Float inference took 0.016 ms and int8 0.1 ms on this CPU.
- The policy works on hidden machines (100%), while the controller with nominal gains scored 69% and 74%, and 100% with the gains that the twin estimated.

## F. Physical AI

Two tables from the saved NVIDIA sources (physical against digital AI, model families) and a matching quiz. A random answer scores 0.12.

## What the numbers show

- **Check that a randomization range covers the test range.** A sampler that moves all channels together scored 0% and looked like undertraining.
- **A twin gives earlier alarms when it models the physics,** and it answers what-if questions without touching the machine.
- **A few real labels beat a lot of randomness,** when you can get them.
- **Run the policy where the delay is small.** One network trip of 50 ms halved success and 100 ms removed it.
- **OpenUSD layers keep one scene and many variants.** The source did not change.

## Limits

One seed for most parts, three for D. The "real" machine and images are simulated. Isaac Sim and Omniverse did not run on this GPU.
