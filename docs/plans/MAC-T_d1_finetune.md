# MAC-T — fine-tune d1-omni-600M on the pro decisions (spec for a Sonnet implementer)

Author: Opus · 2026-11-03 · Implements D-039 (user go-ahead for D-022), D-026. Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
tools/train/__init__.py           (new, empty)
tools/train/train_d1.py           (new)
tests/test_train_d1.py            (new)
tools/eval/eval_d1_zero_shot.py   (add --model only)
```

## 0. What d1 is internally (verified by Opus from the cached remote code)
`model = AutoModel.from_pretrained("LiquidAI/d1-omni-600M", trust_remote_code=True, dtype=torch.float32)`; `mod = importlib.import_module(type(model).__module__)` exposes `encode`, `as_question`, `QTYPES` (imported from the bundled `prompt.py`). For a **text** request the model does (see `probabilities_batch` / `_forward` in the cached `modeling_d1.py`):
```python
ids, markers = mod.encode(model.tokenizer, state, q, model.config.max_length, None, False)   # one sequence per question
h = model.encoder.embed_tokens(ids)            # per row, then pad_sequence(batch_first=True)
pad = arange(L)[None] < lengths[:, None]
h = model.encoder(h, pad, offsets)             # offsets = zeros (no media prefix)
logits = model.head(h, pad, marker_pos, marker_mask, qtype)   # (B, K) one logit per option marker; qtype = QTYPES[q.type]
z = logits[i][:q.options]                      # option order = render_options order; a noul is [false, true]
# inference then divides z by config.temperatures[temperature_key(q)] and softmaxes
```
Trainable text path: `model.encoder` (16 layers in `model.encoder.layers`, embeddings `model.encoder.embed_tokens`) + `model.head`. `model.vision` / `model.audio` are unused here.

## 1. `tools/train/train_d1.py`
```
$PY tools/train/train_d1.py --variant joint|factorized|act [--data DIR] [--out DIR] [--init ID_OR_PATH]
        [--epochs 3] [--lr 2e-5] [--weight-decay 0.01] [--warmup 0.05] [--max-tokens 6144] [--accum 4]
        [--train-layers 16] [--balance] [--limit N] [--val-limit N] [--device mps] [--seed 0]
```
Defaults: `--data <dataset>/d1/<variant>`, `--out db/models/d1-ft-<variant>`, `--init LiquidAI/d1-omni-600M`, `--balance` on for `act`.
1. **Load** as in §0 on `--device` (fp32; never bf16 — the model card reports it flips answers). `torch.manual_seed`, `random.seed`.
2. **Freeze**: every parameter `requires_grad=False`; then enable `model.head` and the top `--train-layers` entries of `model.encoder.layers` (16 = all) and `model.encoder.embedding_norm` if `train-layers == 16`. `embed_tokens`, `vision`, `audio` always stay frozen. Log the trainable parameter count.
3. **Rows** (`build_rows(lines, model, mod) -> list[Row]`): every line is `{"id","state","questions","answers",...}` (format of `dataset/v1/d1/*/*.jsonl`); every question becomes one row: `q = mod.as_question(questions[name])`, `ids, markers = mod.encode(...)` as in §0, `target = list(q.criteria).index(answers[name])` for `choice`, and for `noul` `target = 1 if answers[name] == "true" else 0` (raw order [false, true]). `Row = (ids, markers, qtype, n_options, target, line_id, name)`. `--limit` / `--val-limit` cut the number of **lines** (first N in file order).
4. **Differentiable forward** `option_logits(model, rows) -> list[Tensor]`: exactly the §0 computation for a batch of text rows (copy the padding/marker code of `_forward`), returning `z[:n_options]` per row with gradients (no temperature, no softmax, no `.tolist()`).
5. **Batches** (`make_batches(rows, max_tokens, rng) -> list[list[int]]`): shuffle row indices with `rng`, split into chunks of 64, sort each chunk by length descending, then greedily cut batches so that `len(batch) * longest <= max_tokens` (a single row longer than the budget forms its own batch). Every row appears exactly once per epoch.
6. **Train**: AdamW over trainable params (`lr`, `weight_decay`); linear warmup over `warmup × total_steps` then cosine decay to 0; gradient accumulation over `--accum` batches; `clip_grad_norm_(…, 1.0)`. Loss = mean cross-entropy over the rows of a batch; with `--balance` use per-class weights = inverse class frequency on the training rows (only meaningful for `act`). Every 50 optimizer steps print `step, mean loss, rows/s, elapsed, eta`.
7. **Evaluate** after each epoch on the val file (`no_grad`, `model.eval()`, same batching without shuffle): per-question accuracy; **joint**: `play` accuracy; **factorized**: `card`, `lane`, `zone`, `all_three` (all three right for the line); **act**: accuracy, balanced accuracy, precision/recall of `false` (= wait); plus val loss. The **primary metric** (joint `play` acc, factorized `all_three`, act balanced accuracy) selects the best epoch; keep the best state dict in CPU memory (`{k: v.detach().cpu().clone()}` for trainable params only) and restore it at the end. Call `torch.mps.empty_cache()` after eval when on mps.
8. **Temperature re-fit** on val with the best weights: for each `mod.temperature_key(q)` present, choose `T` in `numpy.arange(0.5, 3.0001, 0.05)` minimising the mean NLL of `softmax(z / T)`; write it into `model.config.temperatures[key]` (keep the other keys).
9. **Save** to `--out`: `model.save_pretrained(out)`; copy from the source snapshot (`huggingface_hub.snapshot_download(init, local_files_only=True)` when `--init` is a hub id, else the `--init` dir) the files `modeling_d1.py prompt.py encoder.py vision.py audio.py tokenizer.json tokenizer_config.json LICENSE` (those that exist). Then **reload check**: `AutoModel.from_pretrained(out, trust_remote_code=True)` on cpu, `system_one` on the first val line → probabilities sum to 1 ± 1e-3 → `reload_ok`.
10. **Report** `out/train_report.json`: args, rows per split, trainable params, per-epoch `{train_loss, val_loss, val_metrics, seconds}`, `best_epoch`, `temperatures`, `reload_ok`, `peak_mps_bytes` (`torch.mps.driver_allocated_memory()` if available), total seconds. Print a final markdown table of the per-epoch val metrics.
`db/` is gitignored (D-013): checkpoints never enter git.

## 2. `tools/eval/eval_d1_zero_shot.py`
Add `--model PATH` (default `None`) → `D1Ranker(model_path=PATH, backend=args.backend)`; when given, the default `--out` becomes `<dataset>/eval/d1_eval_<backend>_<basename(PATH)>.json` and the JSON records `"model_path"`. Nothing else changes.

## 3. Tests — `tests/test_train_d1.py` (no real model; fast)
Build a tiny fake with the same interface: `FakeModel(nn.Module)` with `encoder.embed_tokens` (Embedding 50×8), `encoder.layers` (2 small Linear blocks) and `encoder(h, pad, offsets)` applying them, `encoder.embedding_norm`, `head(h, pad, marker_pos, marker_mask, qtype)` = gather at markers → Linear(8, 1), `config.max_length`, `config.temperatures = {}`; and a fake `mod` with `encode` (one token per character mod 50, a marker before each option), `as_question` (dict → simple object with `type`, `criteria`, `options`), `QTYPES`, `temperature_key`.
1. `build_rows`: choice target = index of the answer key; noul `"true"` → 1, `"false"` → 0; factorized line → 3 rows.
2. `make_batches`: every row exactly once; no batch exceeds the budget unless it is a single over-long row; deterministic for a fixed rng.
3. Training on a toy separable set (answer determined by a token in the state) reaches ≥ 95% train accuracy within 200 steps; loss decreases.
4. Freezing: with `--train-layers 1`, the bottom fake layer's weights are bit-identical after training; head weights changed.
5. Temperature fit: calibrated logits → T ≈ 1.0 (± 0.1); logits ×3 → T ≈ 3.0.
6. Balanced weights for an imbalanced act set (9:1) → minority weight ≈ 9× majority.
7. `eval_d1_zero_shot.main([... "--model", "x"])` with a monkeypatched `D1Ranker` records `model_path` and the default out name.
8. `@pytest.mark.slow` + `CLASHBOT_SLOW_TESTS=1`: real model, 8 joint lines, 1 epoch, `--train-layers 2` on cpu, finishes and saves/reloads.

## 4. Smoke (allowed once, after the symlinks; the user may be using the GPU — keep it short)
`$PY tools/train/train_d1.py --variant joint --limit 64 --val-limit 32 --epochs 1 --out runs/smoke/T/joint` — paste the report table, rows/s and peak memory, and the estimated hours for the full joint run (3,641 lines × 3 epochs).

## 5. Heavy runs (Opus, serialised)
Full joint → eval on val/test with `eval_d1_zero_shot.py --model db/models/d1-ft-joint` (transformers backend; MLX conversion afterwards) vs zero-shot and the placement prior; then act (dense data, MAC-V), then factorized.

## 6. Report
Per MAC-00 §5, plus the smoke numbers and the time estimate.
