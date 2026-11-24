# ReAct agent and guardrails, 2026-11-24

One full run of `agent-react-guardrails.ipynb` (commit `bd3e9fd`) in the Kaggle notebook of the same name, on 2x Tesla T4.
The run chain stopped once after the server cell with no error, so it was resumed from the next cell in the same kernel.

A first run (commit `b858e32`) found two faults in the notebook, and this run has the fixes:

- The benign set held two tasks that must be refused. A correct block counted as a false block.
- The NeMo Guardrails step was skipped, because Kaggle has no `ensurepip` and `python -m venv` fails. The step now uses `uv`.

Both runs gave the same pass counts in the first experiment and the same attack results.

## Setup

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM, one T4, tool calling with the `hermes` parser |
| World | SQLite with 6 customers, 12 orders, 10 policies. A fresh database for each task. |
| Tools | `search_policy`, `get_order`, `get_customer`, `calculator`, `issue_refund` |
| Tasks | 24: 8 lookups, 12 multi-step, 4 that must be refused. Graders read the answer value and the database. |
| Agent | Plain Python loop, at most 8 steps, loop detection, temperature 0 |

## Experiment 1: five ways to solve the same tasks

| Condition | All | Lookup | Multi-step | Must refuse | LLM calls | Prompt tokens | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single call, no data | 25% | 25% | 0% | 100% | 1.0 | 39 | 781 | 1,817 |
| Single call, all data in the prompt | 79% | 100% | 67% | 75% | 1.0 | 716 | 1,483 | 3,746 |
| ReAct with tools | 54% | 88% | 50% | 0% | 2.5 | 1,462 | 1,941 | 8,371 |
| ReAct and one reflection | 54% | 88% | 50% | 0% | 3.5 | 1,697 | 2,437 | 7,077 |
| Planner and executor | 54% | 100% | 33% | 25% | 6.3 | 2,644 | 5,615 | 10,575 |

Every failed task ended as a wrong answer.
No task reached the step limit, the loop check or a bad tool argument.

## Experiment 2: 12 attacks under six guardrail settings (ReAct)

| Guardrail | Rail type | Attacks that worked | Benign tasks done |
|---|---|---:|---:|
| None | - | 5 of 12 | 7 of 12 |
| Prompt hardening | dialog (approximation) | 5 of 12 | 7 of 12 |
| Input and retrieval filter | input, retrieval | 3 of 12 | 7 of 12 |
| Execution rail (code check) | execution | 0 of 12 | 7 of 12 |
| Output filter | output | 4 of 12 | 6 of 12 |
| All together | all five | 0 of 12 | 6 of 12 |

All five successful attacks were direct messages from the user.
None of the six hidden instructions in tool results worked, even with no guardrail.

The human approval check denied a 150.00 refund by default and allowed it after one exact approval.

## Experiment 3: real NeMo Guardrails

A self-check input rail, with the same 7B model as the judge, blocked 4 of 4 attack messages and 0 of 4 benign messages.

## What the numbers show

- **An agent is not better by default.** One call with all data in the prompt passed 79% of the tasks. The ReAct loop passed 54% with 2 times the prompt tokens. The world here fits in 716 tokens, so retrieval by tools has nothing to save. A larger world would not fit in the prompt.
- **But only a tool agent can act.** The single calls passed the refusal tasks mostly because they cannot pay a refund at all.
- **The agent did not refuse.** ReAct failed all four tasks that it must refuse. It paid a refund or gave out data. A model that can act needs a check outside the model.
- **Reflection added cost and no gain.** It fixed no task. The same model checks its own answer with the same knowledge.
- **Planning did not help this model.** It used 2.5 times the calls of ReAct and lost on multi-step tasks.
- **Put authority in code.** Only the execution rail stopped every attack, and the benign result stayed the same. Prompt hardening changed nothing. A filter on words let through the attacks that use no suspicious words.
- **Rails have a cost.** The output filter stopped one benign task.

## Limits

One small model, 24 tasks, 12 attacks, one run per setting.
A gap of one or two tasks is noise.
The notebook also prints a `false_blocks` column. It counts benign runs that were blocked and also failed, so it overstates harm. For the execution rail it shows 2, but the benign result did not change.
The execution rail checks each refund alone. Many small refunds could go past the limit.
The NeMo test has 8 messages.
