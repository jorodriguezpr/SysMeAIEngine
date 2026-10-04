# Training guide

Three ways to make the model smarter offline:

| Method | Use when | Cost |
|---|---|---|
| **SFT** (`train finetune --method sft`) | you have instruction/output JSONL pairs | moderate — full model updated |
| **LoRA** (`--method lora`) | large HF model, limited RAM | small adapter, low VRAM |
| **QLoRA** (`--method qlora`) | very large model on consumer hardware | quantized 4-bit + LoRA |
| **Distill** (`train distill`) | you have a PDF → want QA pairs injected | cheap, safe for TinyGPT |

## Dataset format

JSONL, one row per example:

```json
{"instruction":"Answer briefly","input":"What is the sky?","output":"The sky is the atmosphere above Earth."}
```

`build_dataset.py` (or `ingest build-dataset`) produces this from chunks.

## TinyGPT vs HF

- `--model minilm` → loads `TinyGPT` reference model, full-FT (~2M params), saves `minilm.pt` to `output/`.
- `--model Qwen/Qwen3-0.6B` → HF provider; needs ~2 GB free RAM + patience.

## Weight editing

The `weights edit` endpoint (`/api/weights/edit`) and the dashboard's *Weights* tab
perform surgical ops on the loaded TinyGPT checkpoint:

- `scale` — multiply all weights by a factor
- `bias` — add a constant
- `clip` — clamp to ±value
- `sparsify` — zero the fraction of smallest-magnitude weights (e.g. 0.1 = remove 10%)

These edits persist to `models/weights/<name>/minilm.pt` and are picked up on next load.

## Train-from-Folder: chunking modes and dataset preview

`scripts/train_from_folder.py --chunking {window,structured,paragraph}`:

- `window` (default, unchanged behavior): fixed character sliding window, can cut mid-sentence.
- `structured`: packs whole sentences per chunk.
- `paragraph`: packs whole paragraphs per chunk (falls back to sentence-packing for any
  single paragraph bigger than the chunk size). Requires `clean_text()` to have preserved
  blank-line paragraph breaks as `"\n\n"` — it does, as of the fix that made this mode
  possible (previously *every* run of 2+ whitespace chars, including real paragraph breaks,
  collapsed to a single space, so there were no paragraph boundaries left to chunk on).

**False paragraph breaks in PDFs.** pdfminer emits a blank line between its layout
boxes, which often falls mid-sentence ("Ellos engendran⏎⏎a otras Personas"). Measured
on the real aprocrifos PDFs: 1046 of 1825 (57%) of all "paragraph breaks" were such
mid-sentence splits. `extract_pdf()` now runs `repair_paragraph_breaks()` on pdfminer's
output, joining a break only when the text before it has no sentence-ending punctuation
AND the text after starts lowercase (headings, list items and capitalized paragraph
starts are never merged). Same PDFs afterwards: 49 of 800. Existing extracted text is
only affected once a source is re-extracted (edited source file, or a new job).

`--chunking` affects the training dataset. The RAG index always uses paragraph
chunking (`rag.build_index()` calls `chunk_text_paragraphs`), independent of this flag.

### Dataset preview (rows, exact tokens, truncation)

Before starting a real job, the *Train from Folder* UI's "Preview dataset size" button
(`POST /api/train/from-folder/preview`) samples the first few documents, extracts+chunks+
filters them the same way the real job would, then scales the result up for the rest of
the folder. Token counts for the sample are **exact**: the rows are tokenized with the
selected base model's own tokenizer (tokenizer files only, ~1s to load) counting exactly
the text training will see (QA rows include the prompt prefix). TinyGPT (`minilm`) counts
UTF-8 bytes, which is exact for it. If the tokenizer can't be loaded (e.g. offline with an
uncached hub model) it falls back to a chars/4 estimate and says so in
`token_count_method`.

Why exact: chars/4 measured −36% (gpt2 on Spanish, 2.55 chars/token) to +19% (Qwen2.5 on
English, 4.78 chars/token) off on this project's own corpora.

The preview also reports what **`max_seq` truncation** would discard
(`estimated_tokens_trained`, `rows_truncated_pct`, `text_truncated_pct`, and a
`truncation_warning` above 10%). Measured on the aprocrifos folder with paragraph chunking
at the previous hardcoded `max_seq=128`: **64% of the text (gpt2) / 56% (Qwen2.5) was
never trained on** — only the first 128 tokens of each ~330-token row. At 512 almost
nothing is cut. `max_seq` is now a field on the form / `TrainFolderIn` (default still
128, so existing behavior and the RAM pre-flight are unchanged unless you raise it; the
pre-flight now uses the value you pick).

## OCR: scanned PDFs and images

Optional — install on the master node only (extraction never runs on cluster slaves):

```
pip install -r requirements-ocr.txt
```

That's `rapidocr` (3.x) + `pypdfium2`. RapidOCR was chosen over Tesseract because it's
pip-only (no separate Windows binary) and its models ship inside the wheel (no download on
first use). It runs on whatever `onnxruntime` module is importable — including the one
`onnxruntime-directml` provides — and, unlike `rapidocr_onnxruntime`, does **not** pull in
a separate plain `onnxruntime` wheel that would conflict with the DirectML build.

When installed (and `ENGINE_OCR` isn't `off`):

- PDFs: every page whose text layer has fewer than 25 alphanumeric characters is rendered
  (~144 dpi) and OCR'd; pages with a real text layer are still read by pdfminer. A mixed
  PDF only OCRs its scanned pages. An OCR failure never loses the text layer.
- Image files (`.png .jpg .jpeg .tif .tiff .bmp .webp`, any case) become supported
  inputs. Without OCR they're simply not picked up, as before.
- Paragraphs survive: text boxes are grouped into lines by position, and a vertical gap
  noticeably larger than normal line spacing becomes a blank line, so `paragraph`
  chunking works on OCR'd text too.

Measured on a real Spanish PDF page: accents correct, ~3s/page on CPU. The *Train from
Folder* page shows an OCR status card; job logs start with `OCR: enabled` / `OCR: off
(<reason>)`.

Note: some real PDFs have a text layer that exists but is scrambled (e.g. two-column
layouts extracted as word salad — seen in parts of the aprocrifos Nicodemo PDF). OCR only
triggers for *empty* text layers, so those pages are still read from the broken layer.

## Chat answers stuck repeating the same sentence

**Confirmed live**: a real chat answer (`aprocrifos` model, before the fix
below) repeated the exact same sentence 15+ times verbatim instead of
producing a normal answer. Root cause: `HFProvider.generate()`
(`src/core/providers/hf.py`) called `model.generate()` with no
`repetition_penalty` or `no_repeat_ngram_size` at all — a small/undertrained
model reliably locks onto a high-probability phrase and re-emits it for the
rest of the token budget without one. Made worse by this same file's own
"keep generating if the budget was used without a natural stop" logic
(there to avoid cutting off long legitimate answers) — a repetition loop
looks exactly like that condition, so it kept extending the loop instead of
ending it.

Fixed by always passing `repetition_penalty=1.15` and
`no_repeat_ngram_size=3` to `model.generate()`. Live-verified: the same
model, the same question, asked again after the fix — no repetition,
normal varied prose. This is independent of training quality or RAG
grounding; fix it regardless of what else you're trying about the
training data.

## Too few real training steps on a small dataset

**Confirmed live**: fine-tuning `Qwen2.5-0.5B-Instruct` (LoRA) on a
201-row dataset with the defaults (`--batch-size 16`, grad accumulation 8)
produced an effective batch size of 128 — `ceil(201/128) = 2` optimizer
steps for the *entire* training run. A LoRA adapter with 2 real gradient
updates is barely different from the untouched base model; the resulting
chat answers reflected that.

Both `finetune_model.py` and `distributed_finetune.py` now call
`resolve_grad_accum()` (`src/core/finetune_plan.py`) right after loading
the dataset, which automatically lowers `gradient_accumulation_steps` so a
small dataset still gets a reasonable number of steps (floor of 20 by
default), logging when it does so:

```
[finetune/HF] auto-lowered --grad-accum 8 -> 1 (236 rows x 1 epoch(s) / batch_size 16 ...)
[finetune/HF] ~14 optimizer step(s) expected (236 rows x 1 epoch(s), effective batch 16)
```

It only ever lowers grad_accum, never raises it, and if even grad_accum=1
can't reach the floor (a very small dataset relative to batch_size), it
settles for the best achievable step count rather than erroring out.
Raising `--epochs` is the other lever for a small dataset — the resolver
already accounts for epochs when deciding whether to intervene.

## Evaluating a RAG index

Measures whether a corpus's RAG index actually retrieves the right passage — no separate
labeled eval set needed: eval questions are generated from a held-out slice of the
corpus's own already-indexed chunks. Three ways to run it:

- UI: *Train from Folder* → "Evaluate a RAG corpus" (runs in the background, polls).
- API: `POST /api/rag/eval` → `{eval_id}`, then `GET /api/rag/eval/{eval_id}`;
  `GET /api/rag/corpora` lists evaluable corpora. Results are in memory until restart.
- CLI: `python -m src.cli rag eval --corpus <name> [--question-mode keywords] [--model <name>] [--details]`
  (or `scripts/evaluate_rag.py`, same options).

Metrics: `recall_at_k` (source chunk anywhere in top-k), `hit_at_1` (ranked first) and
`mrr` (mean reciprocal rank). recall@k alone saturates easily — hit@1/MRR distinguish
"found at #1" from "found at #5".

Question modes:

- `first_sentence` (default, original behavior): the chunk's first sentence, verbatim.
- `keywords`: 6 distinct exact words (5+ letters) from the chunk, no sentence. This is a
  **lexical** test, not a paraphrase test — useful for comparing retrieval methods.

Measured on aprocrifos-v2 (20 questions each):

| method | mode | recall@5 | hit@1 | MRR |
|---|---|---|---|---|
| hybrid | first_sentence | 1.00 | 0.80 | 0.892 |
| hybrid | keywords | 1.00 | 0.90 | 0.950 |
| tfidf | first_sentence | 1.00 | 0.75 | 0.875 |
| tfidf | keywords | 1.00 | 1.00 | 1.000 |
| semantic | first_sentence | 0.45 | 0.30 | 0.360 |
| semantic | keywords | 0.25 | 0.05 | 0.129 |

i.e. on this Spanish corpus the semantic half contributes little — hybrid's quality comes
from TF-IDF (+ rerank).

**Correction (2026-09-29):** an earlier version of this note blamed an English-only
embedder (`all-MiniLM-L6-v2`). Wrong: the indexes were already built with the
multilingual `paraphrase-multilingual-MiniLM-L12-v2` (verified by re-embedding stored
chunks: cosine 1.000), and a re-run gives identical numbers. The multilingual MiniLM itself
is weak on this corpus. It doesn't hurt answers (hybrid recall@5 = 1.00); a stronger
multilingual embedder (e5 / bge-m3) is a possible upgrade, at the cost of re-embedding
every corpus.

Pass a model to additionally score generated answers: for each held-out question it
generates a real RAG-grounded answer and reports `avg_keyword_overlap` (fraction of the
expected answer's 4+-letter words — any script, accents included — that appear as words
in the generated text) and `refusal_rate` (answers that are the grounding prompt's "does
not contain this information" refusal). Crude and dependency-free (not BLEU/ROUGE); a
directional signal only. Slow: one real generation per question.

## CPU training crash: STATUS_ACCESS_VIOLATION (exit code 3221225477)

**Confirmed live on a real Windows machine** (16 logical cores): both
`finetune_model.py` and `distributed_finetune.py` (local and distributed
alike, LoRA and full SFT alike) crashed with exit code `3221225477`
(`0xC0000005`, Windows' STATUS_ACCESS_VIOLATION) right at the very first
real training step — every time, reproducibly — after loading the model
and building the dataset successfully. No Python traceback, no Windows
Error Reporting entry either; the process just silently dies.

Root cause: `torch.set_num_threads(os.cpu_count())` gives PyTorch its own
intra-op thread pool sized to every logical core, while MKL/OpenMP
independently size their own BLAS thread pool the same way unless told
otherwise — the two pools oversubscribing each other on this Windows/CPU
combination is what crashes. Both scripts now set `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` (via `os.environ.setdefault`, so an explicit value you
already set isn't overridden) at the very top of the file, before numpy or
torch is ever imported — MKL/OpenMP size their thread pools once at
library-load time, so setting this any later (e.g. right before
`torch.set_num_threads()`) does not work. Live-verified this fixed it: the
same job that crashed immediately then completed cleanly (real loss curve,
real checkpoint, `DONE in 359.3s`) with no other change.

If you've verified a higher value works fine on your own machine, set
`OMP_NUM_THREADS`/`MKL_NUM_THREADS` yourself before launching — the
`setdefault` won't touch it.

## Self-learning safety

- The agent **writes staging files only** — it never triggers training.
- Use the dashboard *Training* tab to review the new chunks, then start a FT run.
- Configure `ENGINE_AGENT_ALLOWLIST` to bound the crawlable domains.
