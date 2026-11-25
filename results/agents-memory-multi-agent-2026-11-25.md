# Agents, memory, state and multi-agent design, 2026-11-25

The notebook is `agents-memory-multi-agent.ipynb`. Its parts ran in three places, because the hosted NVIDIA API throttles Kaggle's shared network address:

| Part | Where | Commit |
|---|---|---|
| A (rules, call, agent), B (state), C (memory) | Kaggle batch run, Version 2 | `b9b7f86` |
| F (NeMo Agent Toolkit) | Kaggle batch run, Version 4 | `9f10e92` |
| A0 (model choice), D (multi-agent), E (improvement loop) | Laptop, same notebook cells | `93bf985` |

On the laptop, 1,743 of 1,744 calls returned an answer and none was throttled.
The agent model is `nemotron-3.5-bolt-e4b` (4B, reasoning off). Every key measurement ran twice.
A first version used `nemotron-3.5-lightning-30b` and gave the agent 27%, because 29 of 96 calls were throttled and the model answered without calling its tool.

## A0. Which model should call the tools? (18 tool tasks, two runs)

| Config | Success | Tool called | Tokens | Seconds |
|---|---|---:|---:|---:|
| bolt 4B, reasoning off | 94% (82 to 98) | 100% | 516 | 5.0 |
| nano-omni 30B, reasoning off | 72% (56 to 84) | 83% | 918 | 20.6 |
| nano-omni 30B, reasoning on | 89% (75 to 96) | 100% | 1,396 | 24.7 |

Exact McNemar: reasoning on against off for the 30B model, p = 0.031. The 4B model against the 30B model without reasoning, p = 0.008.
Reasoning off made the 30B model answer without a tool in one of six tasks ("I need the order ID").

## A. Rules, one call, agent (24 tasks, two runs)

| | Success | LLM calls | Tokens |
|---|---|---:|---:|
| Fixed rules | 12% (6 to 25) | 0 | 0 |
| One LLM call | 25% (15 to 39) | 1 | 93 |
| Agent with tools | 88% (75 to 94) | 1.8 | 429 |

Rules are perfect on the wording they expect, and wrong on 6 of 6 tasks whose live value changed. One call is perfect on facts in the prompt and scores 0 on the tool tasks.

## B. State and context (10 conversations, facts from early turns)

| Strategy | Clean chat, all facts | Noisy chat, early facts only | Prompt tokens, noisy |
|---|---:|---:|---:|
| Stateless | 0% | 0% | 75 |
| Full history | 100% | 100% | 1,671 |
| Sliding window | 33% | 0% | 714 |
| Summary | 100% | 80% | 127 |
| Structured state object | 100% | 90% | 121 |

## C. Memory across sessions (questions that need an earlier session)

| Store | Right | Another user's value (leak) |
|---|---:|---:|
| None | 0% | 0% |
| Transcript dump | 100% | 0% |
| BM25 retrieval, scoped per user | 100% | 0% |
| Key-value profile, scoped per user | 100% | 0% |
| BM25, **not** scoped per user | 0% | 30% |
| Key-value profile, **not** scoped per user | 20% | **80%** |

## D. One agent against several (12 tasks per family, two runs)

| Family | One agent | Supervisor and parallel workers | Pipeline | Generator and critic |
|---|---:|---:|---:|---:|
| Separable (three independent lookups) | 100% | 100% | 100% | 100% |
| Coupled (each step needs the one before) | 100% | **0%** | 92% | 54% |
| Simple (one lookup) | 100% | 58% | 100% | 100% |
| Tokens, all families | 825 | 1,506 | 867 | 1,198 |

The supervisor was also 2.0 to 2.3 times slower in wall time, although its workers run in parallel.

## E. Improvement loop for a ticket router (10 dev, 20 held-out, two runs)

| Change | Dev | Held-out |
|---|---:|---:|
| Baseline prompt | 20% | 25% |
| Add a check step (workflow) | 40% | 52% |
| Rewrite the prompt from the errors (kept) | 85% | 80% |
| Swap in the 30B model (reverted, no gain) | 85% | 72% |

Always the most common answer: 30%.

## F. NVIDIA tools

The NeMo Agent Toolkit (`nvidia-nat` 1.9.0) installed and ran a ReAct workflow from a YAML file, and answered `2026` with a tool call. Its evaluation scored 4 of 4 with exact match, and its profiler wrote its traces, a Gantt chart and an optimization report (2,913 events).
In the first run the profiler failed, because Kaggle sets an inline matplotlib backend.
NeMo, the Agent Toolkit and NemoClaw are explained from the saved NVIDIA sources in the notebook.

## What the numbers show

- **Use an agent when facts change or must be fetched.** The agent passed 88% where one call passed 25%.
- **Model size is not the same as tool skill.** A 4B model that calls its tool beat a 30B model that does not.
- **State must keep facts, not text.** A small structured state object kept 90% of the early facts for 121 prompt tokens, against 1,671 for the full history.
- **Memory must be scoped per user.** Without a scope, a key-value store gave another user's value in 80% of the questions.
- **More agents is not better by default.** One agent was best or equal in all three families, with the fewest tokens. The supervisor scored 0% on coupled tasks, because it splits a chain into parallel parts.
- **The prompt change gave the gain and the bigger model did not.** The dev and held-out scores stayed close (5 points).

## Limits

One small model in most parts, 5 to 24 tasks per cell, and tasks that code can check. The designs of part D use simple prompts, and a better supervisor prompt may score higher. The tickets and the shop world are fictional.
