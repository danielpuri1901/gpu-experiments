# RAG chunking, hybrid search and reranking, 2026-11-23

One run of `rag-chunking-hybrid-rerank.ipynb` in the Kaggle notebook of the same name, on 2x Tesla T4.
The run stopped once at a cell boundary with no error, and once on a bug in the reranker warm-up line (fixed in commit `c3e1072`).
Both times it was resumed from that cell in the same session, so all results come from one kernel.

## Setup

| Item | Value |
|---|---|
| Corpus | 97 public NVIDIA article URLs, fetched at run time. 89 loaded, 87 kept (200 words or more), 170,352 words. |
| Eval set | 120 passages of 150 words, seed 0. `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM wrote one question per passage. 101 questions passed the JSON and wording checks. |
| Hit rule | A retrieved chunk is a hit if it comes from the gold page and covers at least 50% of the gold passage words. |
| Retrievers | BM25 (`rank_bm25`), dense (`BAAI/bge-small-en-v1.5`, cosine), hybrid with reciprocal rank fusion (k = 60) |
| Reranker | `BAAI/bge-reranker-base` cross-encoder on the hybrid top 20 |
| Answers | 30 questions, answered by the same vLLM model, judged for groundedness by the same model |

## Retrieval

| Chunk words | Method | Chunks | Recall@1 | Recall@5 | MRR@10 |
|---:|---|---:|---:|---:|---:|
| 128 | BM25 | 1607 | 0.713 | 0.941 | 0.803 |
| 128 | Dense | 1607 | 0.525 | 0.822 | 0.649 |
| 128 | Hybrid (RRF) | 1607 | 0.614 | 0.891 | 0.729 |
| 256 | BM25 | 817 | 0.743 | 0.990 | 0.838 |
| 256 | Dense | 817 | 0.564 | 0.812 | 0.658 |
| 256 | Hybrid (RRF) | 817 | 0.624 | 0.921 | 0.748 |
| 512 | BM25 | 415 | 0.743 | 0.970 | 0.841 |
| 512 | Dense | 415 | 0.436 | 0.733 | 0.558 |
| 512 | Hybrid (RRF) | 415 | 0.554 | 0.891 | 0.696 |
| 256 | Hybrid + rerank top 20 | 817 | 0.812 | 0.960 | 0.879 |

The reranker added 130 ms per query on the T4.

## Answer groundedness (30 questions, LLM judge)

| Condition | Grounded rate |
|---|---:|
| (a) No context | 0.67 |
| (b) Top 5 hybrid chunks | 0.97 |
| (c) Top 5 reranked chunks | 0.93 |

## What the numbers show

- **BM25 won on this eval set, and the eval set explains why.** The questions were written from the passages, so they reuse the passage words. That favours keyword search. This is a known bias of synthetic QA. Real user questions use their own words, where dense retrieval usually does better. So this result says "measure on your own questions", not "BM25 beats embeddings".
- **The reranker gave the best top-1.** It lifted hybrid recall@1 from 0.624 to 0.812, above BM25 alone (0.743). A cross-encoder reads the question and the chunk together, so it judges relevance better than a vector or a keyword match. The cost was 130 ms per query.
- **Chunk size matters most for dense retrieval.** Dense recall@5 fell from 0.822 at 128 words to 0.733 at 512 words. The embedding model reads at most 512 tokens, so it cuts long chunks, and one vector for a long chunk mixes many topics.
- **Hybrid fusion fell between its two parts.** Reciprocal rank fusion helped dense a lot but pulled BM25 down here, because the dense ranking was much weaker on this set. Fusion helps most when the two rankers are about equally good.
- **Retrieved context made answers grounded.** Without context, the judge found unsupported claims in a third of the answers. With 5 retrieved chunks, almost none. The gap between hybrid and reranked context (0.97 against 0.93) is one answer out of 30, so it is noise.

## Limits

One run, one seed, 101 synthetic questions, 30 judged answers.
The same model wrote the questions, the answers and the judgments.
The hit rule is a proxy for relevance.
