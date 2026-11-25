# Alignment: SFT, DPO, KTO, reward models, GRPO and distillation, 2026-11-25

One full run of `alignment-sft-dpo-kto-grpo.ipynb` (commit `87eb7ce`) in the Kaggle notebook of the same name (Version 2, T4 x2, 42 minutes). All 12 parts passed.
Version 1 is not used. Its SFT, QLoRA, DPO, KTO and distillation runs could not emit their end token, so replies ran to the length limit. The end token of the base model is now `<|endoftext|>`, and every trained model stops on it.

Student `Qwen/Qwen2.5-0.5B` (LoRA, fp32), teacher `Qwen/Qwen2.5-1.5B-Instruct`, TRL 1.18.2, one seed. Six instruction families with graders that code runs. Held-out: 120 items, no prompt in two sets. All intervals are 95% Wilson. Paired tests are exact McNemar.

## What does each method change?

| Model | Held-out right | Reply length, characters | Train time | Peak memory |
|---|---|---:|---:|---:|
| Base | 0.01 [0.00, 0.05] | 282 | | |
| Instruct (reference) | 0.53 [0.44, 0.62] | 43 | | |
| SFT, LoRA | 1.00 [0.97, 1.00] | 28 | 378 s | 2.18 GiB |
| SFT, QLoRA | 0.99 [0.95, 1.00] | 27 | 435 s | 0.99 GiB |
| DPO after SFT | 0.90 [0.83, 0.94] | 18 | 135 s | 2.11 GiB |
| KTO after SFT | 0.97 [0.93, 0.99] | 18 | 194 s | 2.11 GiB |

- The base model follows 0 to 1% of the instructions, and 84 of 120 of its replies never stop. SFT takes it to 100% on its own rules.
- The 0.5B instruct model fails the list, word-limit and lower-case families (0, 8 and 2 of 20). SFT on 1,500 items beats it, so a narrow task needs little data.
- QLoRA used 55% less memory and 15% more time. Quality did not differ (one item, p = 1.0).
- DPO made replies shorter and **lowered** the pass rate from 1.00 to 0.90 (p = 0.0005). KTO stayed at 0.97 (p = 0.25 against SFT). KTO was better than DPO (p = 0.004).
- The preference rule says a right reply beats a wrong one, and the shorter of two right replies wins. Against SFT, DPO won 44, tied 58 and lost 18 (win rate 0.61), and KTO 51, 62 and 7 (0.68). They won by getting shorter (28 to 18 characters), and DPO paid with right answers. The training pairs always have the shorter reply as the chosen one, so length and preference cannot be told apart.

## Reward hacking

Two reward models, same recipe, 1,000 pairs each. K1 is right-short over right-long, K2 right-short over wrong-long, K3 right-long over wrong-short. The first model never sees K3.

| | Confounded (K1, K2) | Balanced (K1, K2, K3) |
|---|---|---|
| K1 and K2 accuracy | 1.00 and 1.00 | 0.99 and 0.99 |
| K3 accuracy | **0.00** [0.00, 0.03] | 0.83 [0.76, 0.89] |
| Spearman(reward, length) | -0.77 | -0.24 |
| Empty reply beats the right reply | 32% | 1% |
| "yes yes yes..." beats the right reply | 23% | 17% |

The confounded model scored 100% on the data it resembles and learned "short is good". An empty reply got a higher score than the right one in 32% of the prompts.
Best-of-n with n = 8 or 16 raised the reward from 12 to 22 and the true score did not change (0.96 to 0.97), because SFT already scores 0.96 on one sample.

## GRPO on arithmetic

| | Base | GRPO |
|---|---|---|
| Arithmetic right, 100 items | 0.00 [0.00, 0.04] | 1.00 [0.96, 1.00] |
| Replies with no answer line | 79 | 0 |
| Held-out instructions, 120 items | 0.01 | 0.38 [0.29, 0.46] |
| GSM8K probe, 50 items | 7 | 14 |

Mean reward went from 0.25 to 0.96 of 1.0 in 89 steps and 15 minutes.
The base model's raw replies contain the right number in the samples that were printed (`8 * 25 + 15 = 200 + 15 = 215`) and then never stop. So part of the gain is learning the answer format and to stop. The share of items whose reasoning was already right was not measured.
The instruct model was the opposite case. In the answer-only prompt it writes a well-formed line for 100 of 100 items and gets 4 right. In the think-first prompt it reasons step by step but writes no answer line, so it scores 0 of 100. A prompt change alone moves this score, so read it as a format effect.

## Distillation (0.5B student, 1.5B teacher)

| | Held-out right | GSM8K, 50 items |
|---|---|---|
| Teacher, 1.5B instruct | 0.55 [0.46, 0.64] | 0.52 |
| Student from teacher replies | 0.56 [0.47, 0.64] | 0.00 |
| Student from replies, then logits | 0.61 [0.52, 0.69] | 0.00 |
| SFT on code-made replies | 1.00 [0.97, 1.00] | 0.12 |

- A student stays at the teacher's level. The students scored 0.56 and 0.61 against the teacher's 0.55, and neither is reliably different from the teacher (p = 0.14 for the logit student). SFT on correct labels beats both (p below 0.001).
- Logits added 5 points to the held-out score (p = 0.18, no reliable difference).
- Distillation used 8.8 GiB for the logit stage, because it holds the teacher and the student.
- The student lost all of its GSM8K ability (0 of 50).

## Forgetting (GSM8K probe, 50 items, base 7 right)

| Method | Right |
|---|---:|
| SFT | 6 |
| DPO | 2 |
| KTO | 1 |
| GRPO | 14 |
| Distilled students | 0 |
| Instruct (reference) | 18 |

The probe is small, and one item is 2 points. Only the distilled students (0 of 50) and the instruct model (18) differ clearly from the base (unpaired Fisher p = 0.012 and 0.020). The drops to 2 (DPO) and 1 (KTO) and the rise to 14 (GRPO) are not certain (p = 0.16, 0.06 and 0.14).

## What the numbers show

- **SFT sets what a model does.** Preference methods then move how it answers, and a confounded preference moves it the wrong way.
- **A reward is a proxy.** Check it on cases it never saw (K3), and on junk replies.
- **Read the stop reason.** A model that never stops scores zero, and an untrained end token looks like a training failure.
- **Distillation copies a teacher.** The student stays near the teacher's score, and here a narrow SFT on code-made labels beat both.
- **Check a second ability after every fine-tune.** The distilled students went from 7 to 0 of 50 on GSM8K.

## Limits

One 0.5B student, one seed, LoRA only, 120 held-out items and 50 GSM8K items. PPO did not run: best-of-n stands in for the RL step of RLHF. The logit student is short-trained (100 steps). The teacher runs in fp32.
