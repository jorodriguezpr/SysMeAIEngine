# Grounding & Knowledge Graph

How SysMeAIEngine answers questions **from your documents** rather than from
model weights, and how the knowledge-graph addon adds traversal-based
reasoning on top of plain retrieval.

## Why grounding matters more than fine-tuning

Fine-tuning on a small corpus teaches answer *format*, not book *content* —
a few hundred QA rows cannot encode a book's knowledge, and small models
reliably fail to call tools on their own. Grounding inverts the topology:
**the documents are the memory, retrieval is the senses, the model is only
the voice.** With scarce training data this is the only topology that works.

## P0 — retrieve-before-generate

`/api/chat` now has a `chat_mode` (Settings page → *Chat grounding*):

- **`plain`** — the original weights-only generation (default, unchanged).
- **`grounded`** — every prompt first retrieves passages from the local RAG
  indexes (and knowledge graph, if built), then generates an answer
  **strictly from those passages**, with an explicit refusal when retrieval
  finds nothing (`grounded_refuse_when_empty`). Cloud provider writes the
  final answer when enabled (`grounded_use_cloud`), degrading to the local
  model on failure — including network errors and an empty answer — with
  the reason in `grounding.fallback`. If no RAG index exists at all, the
  request falls back to the plain path instead of erroring.
- A per-request `use_cloud: false` wins over the setting and keeps the turn
  **fully local**: no multi-query rewrite, no cloud answer.
- The local fallback is only as good as the local model: with
  `ENGINE_DEFAULT_MODEL=minilm` (TinyGPT toy) it produces gibberish — pick a
  real model (e.g. qwen2.5) in the chat or as the default.

Request overrides: `{"chat_mode": "grounded", "top_k": 6}` on any
`/api/chat` call. The response `model` field shows what actually answered:
`grounded:cloud:<provider>:<model>`, `grounded:local:<model>`, or
`grounded:none` (refusal). Grounded responses also carry `sources`
(corpus/source/id/score per passage) and `grounding` (variants used,
parent expansions, graph hits, corpus, use_cloud, fallback reason).

`/api/rag/chat` (the Chat page's "Ground in ingested books" checkbox) now
runs the same P1 upgrades by default; `grounded: false` restores the exact
old single-query behavior.

## P1 — retrieval upgrades (rag.py)

- **Multi-query expansion** (`expand_query`): rewrites the question into
  2–3 differently-phrased variants with the cloud provider and searches all
  them, fusing results by RRF. Bridges the phrasing gap between user
  questions and book prose ("quien bautizo a jesus" vs "el bautismo del
  Señor fue administrado por…"). Best-effort: any provider failure
  (including network errors) returns the original question unchanged.
  Uses the default 10000-token budget — reasoning models like
  glm-5.3-flash returned **empty** content at the old 200-token cap.
- **Parent-child expansion** (`expand_context`): each hit is widened to its
  surrounding paragraphs from the same source document (±1 neighbor
  minimum, up to `parent_chars`), in document order, so the generator sees
  full passages, not fragments. A chunk already in the context (via another
  hit's window) is never repeated. Pure widening of already-indexed content.
- **MMR diversity** (`_mmr_diversify`): char-3gram MMR on 0..1-scaled
  rerank scores, plus a hard skip for near-identical passages (≥ 0.9
  similarity) — the same passage indexed in several corpora.
- **`retrieve_grounded`**: multi-query → cheap semantic+TF-IDF candidates
  per variant and corpus, fused by one RRF, exact copies dropped by content
  id, cut to a 30-passage pool (`_grounded_candidates`) → ONE cross-encoder
  rerank → optional kgraph merge → MMR → parent-child. Returns
  `(hits, debug)`.

## Knowledge graph (`src/core/kgraph.py`)

The lightweight KR layer: entities + relations extracted from the same
chunks the RAG index uses, stored in SQLite at `data/kgraph/<corpus>.db`,
each triple linked back to its source chunk id for citations.

- **Build** (UI: *Knowledge Graph* page; CLI: `python -m src.cli kgraph
  build --corpus <name> [--max-chunks N]`; API: `POST /api/kgraph/build`):
  one cloud API call per indexed chunk (same cost-bounding pattern as cloud
  QA generation), writes triples as `(subject | relation | object)` lines
  in the source language. Resumable — every successfully processed chunk
  (including ones that yielded zero triples) is recorded in
  `processed_chunks` and skipped on rebuild; failed chunks (network error,
  empty output) are counted in `chunks_failed` and retried next build.
  One failure never aborts the build. Entity names are normalized for
  case, accents, whitespace and underscores (`rey_del_Norte` = `rey del
  Norte`). Corpus names are validated (no path components).
- **Query**: entities are detected in the question (accent/case-normalized,
  light plural tolerance), and the 1-hop neighborhood is returned with
  provenance: `graph_facts()` / `POST /api/kgraph/query`.
- **In chat**: when `chat_kgraph` is enabled, `retrieve_graph_context`
  merges fact rows into the grounded context shaped exactly like RAG hits,
  so entity/relation knowledge rides through the same prompt, MMR, and
  citation pipeline.

Cost note: extraction is one API call per chunk, one-time per corpus.
`--max-chunks` bounds it while validating on a new corpus.

## Settings reference

| Setting (Settings page → Chat grounding) | Default | Effect |
|---|---|---|
| `chat_mode` | `plain` | `grounded` retrieves before generating on `/api/chat` |
| `chat_top_k` | `6` | passages injected per grounded turn (1–20) |
| `chat_multi_query` | `true` | multi-query expansion (needs cloud provider) |
| `chat_kgraph` | `true` | merge knowledge-graph facts into context |
| `grounded_use_cloud` | `true` | cloud writes the final answer; local model fallback (a request's `use_cloud` overrides) |
| `grounded_refuse_when_empty` | `true` | refuse when retrieval finds nothing |

API: `GET/POST /api/settings/grounding`.

## Recommended setup for a small corpus (e.g. aprocrifos)

1. Build the RAG index (`rag build --method hybrid`) — grounded chat needs it.
2. Enable the cloud provider on Settings (it is the reasoner, not a
   dependency — local fallback is automatic).
3. Set *Chat mode* = **Grounded** on the Settings page.
4. Build the knowledge graph (`kgraph build --corpus <name>`), inspect the
   top entities on the UI page, and spot-check `kgraph query --corpus <name>
   --query "..."` results.
5. Re-run `rag eval` + `kgraph stats` after re-ingestion; graph builds are
   incremental so only new chunks cost API calls.
## Verification (2026-09-27)

Live against the real aprocrifos indexes and the configured `ollama_cloud`
`glm-5.3-flash` provider; knowledge-graph builds ran in a temp copy of the
index so `data/` was never written. Problems found in the first version, all
fixed and covered by `tests/test_grounding_fixes.py` /
`tests/test_grounding_api.py`:

| Problem (reproduced live) | Fix |
|---|---|
| Network error → HTTP 500 on `/api/chat` and `/api/rag/chat` | `cloud_chat._post` turns transport errors into `RuntimeError`; grounded chat falls back to local |
| `use_cloud: false` still made 2 cloud calls (rewrite + answer) with the passages | per-request flag honored; `allow_cloud=False` also skips the rewrite |
| Multi-query produced 0 variants (200-token cap, reasoning model) | default 10000-token budget |
| Grounded answer came back **empty** (reasoning used the whole budget) | 10000 default; empty → one retry, then local fallback |
| ~70s per grounded turn on corpus "all" | single-rerank candidate pool: 2.4s vs 34.6s retrieval, same gold recall |
| MMR never changed a selection; same passage 3–4× in top 6 on "all" | scaled relevance + near-duplicate skip |
| Up to 10 of 53 context paragraphs duplicated; neighbors out of order | chunk-level dedupe + document order in `expand_context` |
| kgraph rebuild re-paid for zero-triple chunks; one timeout aborted the build | `processed_chunks` table; per-chunk failure handling |
| `DELETE /api/kgraph/..%5C…` deleted a file outside `data/kgraph/` | corpus-name validation → 400 |
| `chat_top_k: -5` accepted | validated 1–20 |

End-to-end after fixes (grounded `/api/chat`, corpus "all", cloud answerer):
Pilatos question 34s (cold) with a quoted, cited answer; María Magdalena
20s, 1635-char grounded answer (was empty); an out-of-corpus question
refused in 8s.

Known limits: the refusal sentence is always English (fixed string in
`build_prompt`), even for Spanish questions; char-trigram similarity is a
proxy, not semantic dedupe.

## Measured feature value (P5, 2026-09-28)

`scripts/eval_grounded.py` (module `src/core/grounded_eval.py`) measures the
whole grounded path: a saved question set per corpus
(`data/eval/<corpus>-grounded.json` — natural single-passage questions,
two-passage questions paired through a shared knowledge-graph entity, and
out-of-corpus questions), each configuration = baseline + ONE feature, the
cloud model answering, the cloud model judging against a reference answer.
Apparatus failures (empty answer after retries, unparseable judge, dropped
connection) are excluded and counted, never scored as wrong.

```
python scripts/eval_grounded.py --corpus aprocrifos-v2 --build-set --configs none   # once
python scripts/eval_grounded.py --corpus aprocrifos-v2 --configs all                # ablation
python scripts/eval_grounded.py --corpus aprocrifos-v2 --configs baseline --top-k 3 --answerer local:<model>
```

aprocrifos-v2, 20 single + 5 multi + 5 out-of-corpus questions, top-6, two
runs combined:

| config | recall single / multi | accuracy | multi correct | retrieval | context |
|---|---|---|---|---|---|
| baseline | 0.90 / 0.80 | 0.918 | 0.89 | 3.2s | 7.1K chars |
| +multi_query | 0.90 / 0.80 | 0.920 | 0.90 | 6.5s | 7.1K |
| **+parent_child** | **0.95 / 0.90** | **0.958** | **1.00** | 2.2s | 16.1K |
| +kgraph | 0.90 / 0.80 | 0.920 | 0.90 | 2.2s | 6.3K |
| +mmr | 0.90 / 0.80 | 0.957 | 0.94 | 2.2s | 7.1K |
| all | 0.95 / 0.90 | 0.957 | 0.88 | 6.4s | 14.4K |

Noise floor: repeating a config changes 0–2 of ~24 graded answers, so
accuracy gaps of ~0.04 are noise; recall (deterministic) is the firm signal.
Every config refused all out-of-corpus questions.

Decisions (applied as defaults): **parent-child on** (only feature that
retrieves passages baseline misses), **MMR on** (neutral in one corpus,
removes the 3–4× duplicates of corpus="all"), **multi-query off** (no recall
gain, 2× retrieval time, sends the question to the cloud), **knowledge graph
off** (no gain; its rows displace a passage — the graph stays available).
**P3 agentic loop: not built** — one retrieval pass with parent-child
answered every two-passage question correctly in both runs.

Offline (local qwen2.5-0.5b answering, cloud judging, top-3): accuracy
**0.08**, out-of-corpus refusal **0.2**; parent-child made it worse
(correctness 0.33 → 0.20, refusals 0.2 → 0.0), so local answers use the
compact (non-widened) context. Caveat throughout: the judge is the same
cloud model; compare configs, don't over-read absolute scores; 25
answerable questions can only reveal large effects.

### Offline answers and P4 (grounded fine-tune)

**Engine bug found by this eval (fixed):** `hf.py` applied
`repetition_penalty` / `no_repeat_ngram_size` to the PROMPT too, so a local
model answering from retrieved passages was penalized for copying their
words — "comadrona Zelomi" came out as "comadrona Zalomi" with a Cyrillic
letter, "judíacos", "comadRONA ZelOMI descUBRIÓ". Penalties now apply to
generated tokens only (`generated_only_penalties`); loops are still blocked.

P4 = `scripts/build_grounded_dataset.py` (140 chat-format rows from
non-eval chunks: cloud-written concise answers + 25 refusals) → LoRA on
Qwen2.5-0.5B-Instruct, 2 epochs → `aprocrifos-grounded-0.5b`. Training uses
the `{"prompt","completion"}` row shape, tokenized exactly as inference
sees it (chat template, prompt masked). The older QA format trained on a
prompt shape the chat path never uses.

Offline eval (compact top-3 context, cloud judge, penalty fix applied):

| local answerer | accuracy | single correct | multi correct | OOC refused | answer time |
|---|---|---|---|---|---|
| qwen2.5-0.5b (base) | 0.04 | 0.28 | 0.20 | 0.0 | 11s |
| aprocrifos-grounded-0.5b (P4) | 0.04 | 0.33 | 0.30 | 0.2 | 16s |
| qwen2.5-1.5b (base, untrained) | **0.28** | **0.48** | **0.40** | **0.6** | 43s |
| cloud (reference) | ~0.95 | | | 1.0 | ~10–14s |

Conclusions: P4 on the 0.5B model gave no meaningful gain (it learned the
answer *style*, not refusing — not even on its own training rows); the
0.5B model's limit is comprehension, not training. Model size is what moves
offline quality: the untrained 1.5B is ~7× more accurate, still far below
the cloud and ~4× slower. Recommended: cloud answers; for offline use,
qwen2.5-1.5b-instruct (needs ~6 GB RAM loaded).

## Offline answering (no cloud at chat time) — 2026-09-29

Goal: the system must answer correctly from the trained documents **without a
cloud provider**, even for a single book. Retrieval was already fully local
(multilingual embedder + cross-encoder + TF-IDF); the weak link was the local
model that writes the answer. Measured on the same 30-question set (cloud only
as the *judge*):

| local answerer (compact top-3 context) | accuracy | correct or partial | **wrong when it answers** | out-of-corpus refused | time |
|---|---|---|---|---|---|
| qwen2.5-0.5b (hf) | 0.04 | — | high | 0.0 | 11s |
| qwen2.5-1.5b (hf) | 0.28 | — | — | 0.6 | 43s |
| qwen2.5-3b (Ollama), short prompt | 0.32 | 0.68 | 0.23 | 1.0 | 14s |
| qwen2.5-7b (Ollama), short prompt | 0.44 | 0.76 | 0.10 | 1.0 | 29s |
| **qwen2.5-7b (Ollama), "complete" prompt** | **0.67** | **0.83** | **0.05** | **1.0** | 28s |
| qwen2.5-7b, "complete", top-5 | 0.56 | 0.76 | 0.10 | 1.0 | 29s |

What made the difference:
- **A 7B local reader** (Ollama `qwen2.5:7b-instruct`, 4.7 GB, registered as
  `qwen7b-local`; `qwen3b-local` is the faster, less precise option).
- **Instructions in the question's language** with a localized refusal
  (`build_prompt(..., style="native"|"complete")`), and the **"complete"**
  style: failure analysis showed most misses were *correct but incomplete*
  answers caused by a "be brief" instruction.
- **Compact context** (top-3 passages, no parent-child widening) for local
  readers — widening/top-5 lowered quality for local models.
- **Script-drift guard** (`rag.local_answer`): Qwen occasionally switched to
  Chinese mid-answer; it regenerates once with a language reminder, then cuts.
- Engine bugs fixed along the way: HF `repetition_penalty` applied to the
  prompt (garbled copied names), Ollama provider sent the label instead of the
  model tag and never set `num_ctx` (long prompts silently truncated).

Configured defaults: `ENGINE_DEFAULT_MODEL=qwen7b-local`; grounding settings
`chat_mode=grounded`, `grounded_use_cloud=false`, `chat_top_k=3`,
`local_prompt_style=complete`. Requires Ollama running locally with
`qwen2.5:7b-instruct` pulled.

Tried and **not adopted** (measured, kept in the code as options):
- quote-first prompt — the 3B wrongly refused 52% of answerable questions;
- **Knowledge Compiler** (`src/core/knowledge_compiler.py`,
  `scripts/compile_knowledge.py`): compile each passage into verified
  question/answer/exact-quote facts at build time (1,194 of 1,448 kept after
  quote + support checks). Answering *directly* from matched facts was
  unreliable (paraphrased questions often match another fact); feeding facts
  + raw passages lifted the 3B from 0.32 to 0.44 but did not help the 7B
  (0.40 vs 0.44 short-prompt) — plain reading with the 7B wins.
- P4 grounded fine-tune of the 0.5B — no gain.

Remaining weakness: ~12% of answerable questions are refused although the
passage was retrieved (the model is conservative), and ~3 of 20 questions miss
their passage in the top 3.

## Hard questions: graph-walk retrieval (2026-09-30)

A harder companion eval set (`data/eval/aprocrifos-v2-hard.json`, built by
`scripts/eval_grounded.py --set hard --build-set`): 12 paraphrased questions
(roles instead of names, no copied wording), 8 chained multi-step questions
(identify an entity through passage A, ask what passage B says about it),
5 why/how questions, 5 near-domain out-of-corpus questions (verified absent
from the books). Offline 7B baseline: accuracy 0.32, but only 10% wrong when
answering — it mostly *refused*, because retrieval never found the passage.

Diagnosis (per missed passage): the cross-encoder demoted passages that both
first-stage rankers had at #1 (6 of 15), and paraphrases never entered the
candidate pool at all (8 of 15: "el gobernador romano" vs "Pilatos").

**Graph-walk retrieval** (`src/core/graph_retrieval.py`): one graph over
passages + knowledge-graph entities (fact links, literal mentions, story
order: 1,350 nodes / 5,198 edges for aprocrifos-v2), seeded by matching the
question to entity *fact descriptions* with the cross-encoder, ranked by
Personalized PageRank (d = 0.5). Used as a **candidate source**: its passages
join the reranker's pool and one final slot is reserved for its best passage
— **adaptively**, only when retrieval is unsure (top rerank score < 0).

| offline 7B, complete, top-3 | original set acc | hard set acc | hard: wrong when answering |
|---|---|---|---|
| without graph | 0.71 | 0.32 | 0.10 |
| graph slot always on | ~0.52 | 0.44 | 0.00 |
| **adaptive graph slot (default)** | **0.68** | **0.40** | 0.08 |

Retrieval-level (gold passage in top-3, hard set): 0.42 → 0.60 always-on
(paraphrase 0.25 → 0.50, chained 0.44 → 0.62). Chained questions still fail
at the *answer* level (0.12): the 7B gets both passages more often but can't
reliably combine them. Out-of-corpus refusal stays 100% everywhere.

Measured and **rejected**: reciprocal-rank fusion of all rankers (hard hit@3
0.24 — the reranker is the best single judge), a Spanish stemmer for TF-IDF
(`src/core/es_stem.py`, 0.60 → 0.56), reading passages 4–6 on refusal (only
2 of 25 hard gold passages sit there).

Setting: `chat_graph_slot` (default true). Needs a built knowledge graph for
the corpus (Knowledge Graph page); without one it's a no-op. First unsure
question after a server start builds the graph index (~1 min; entity
embeddings are cached on disk in `data/graph_index/<corpus>/`).
