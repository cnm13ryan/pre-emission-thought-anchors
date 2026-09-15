# Stage 6 investigation and execution plan (2026-09-15, H100 PCIe, branch `codex/stage6` at `abf7ea5`)

Status of this document: investigation and plan only. No Stage 6 or Stage 7 execution
occurred. Two bounded, disposable GPU probes (two trajectories, scratchpad output only,
all files deleted) and CPU-only tokenisation audits were run. No canonical artifact was
touched. Evidence tags: CONFIRMED (spec/code/artifact/measurement), INFERRED (derived),
AMBIGUOUS/BLOCKED (needs a scientific or design decision).

Naming note (CONFIRMED): SPEC §11 defines **Stage 6 = activation-extraction smoke test**
(16 boundaries of `330:correct`, invariance tests, logit features) and **Stage 7 = full
activation and feature extraction** (all 40 trajectories). The request's "full Stage 6
extraction" is SPEC Stage 7. This plan covers Stage 6 fully and Stage 7's extraction
half up to its admission gate.

## A. Stage 6 admission status

| Gate | Result | Evidence |
|---|---|---|
| Stage 5 canonical run complete | PASS | run `20260915T144926Z-bb9ba695de-9fca4c21`, state `labels_constructed`, 256/256 valid checksums, 0 failures, `fresh_labels.parquet` (5 rows, sha `d60d8ddc…`) |
| vLLM stopped, GPU released | PASS | `nvidia-smi`: 0 MiB used, no compute processes |
| Production doctor on this host | PASS | Stage 5 `doctor.json`: all checks ok, `gpu_stage_admitted: true`, `fit_test: "Stage 6 still required"` |
| Stage 3 inputs present, checksums valid | PASS | `sentences.parquet` `83cf8c20…` (byte-identical to canonical), `trajectories.parquet` `4fd5fbee…`, 40/6,474 rows |
| Stage 4 labels present, checksums valid | PASS | `labels.parquet` `fab9886e…`, 6,474 rows, 4,019 eligible, 344 anchors, `canonical_fallback` |
| Model cache pinned revision | PASS | `1df8507…` snapshot, 4 shards, 29,540,067,328 B; loads with 14,770,033,664 params, 48 layers, d=5120 |
| Environment lock | PASS | worktree `.venv` synced from `uv.lock` (hash `1126f52a…`), torch 2.7.0+cu126, transformers 4.53.2; 189 CPU tests pass |
| Stage 5 durable backup verified (AGENTS §9 "back up after every major stage") | UNRESOLVED | manifest `backup.verified: false`; live checkpoint `secondary_verified: false`. Stage 2→3 precedent waited for the MacBook receipt before the next stage. |
| Resource projection for Stage 6/7 (AGENTS §8) | PENDING | must be written from the smoke run; projected below from probes only |
| Worktree submodules | FIXED | were uninitialised in this worktree (would break `provenance()`); `git submodule update --init` now at pinned commits |
| `PREANCHOR_HOURLY_RATE` | PASS | 2.49 USD/h in `.env` (SPEC §7.1 table says 3.29; billed rate unverified) |

Stage 5 outputs are not a data input to Stage 6 or 7 (both read Stage 3 traces and
Stage 4 eligibility); Stage 5 gates Stage 6 only by ordering and GPU release.

## B. Input contract

All under `/lambda/nfs/preanchor/preanchor` (virtiofs UUID `8dd29228-…`).

| Input | Path | SHA-256 | Producer | In scientific identity? |
|---|---|---|---|---|
| Traces | `data/processed/stage3/trajectories.parquet` | `4fd5fbee159c0b54…f39913d` | Stage 3 recomputed on H100 (`reconstruction/20260915T145500Z-…json`) | file hash: no (embeds local `source_path`); `full_cot_sha256` per trajectory: **yes** |
| Sentence spans | `data/processed/stage3/sentences.parquet` | `83cf8c20bda3ba32…143cf8` | Stage 3 | **yes** (char spans define boundaries) |
| Labels/eligibility | `data/processed/stage4/labels.parquet` | `fab9886e478922ac…a61b31` | Stage 4 run `20260915T141929Z-45b6f6e0cc-07aa7070` | eligibility mask only; not needed if all rows are stored (see K3) |
| Model + tokenizer | HF cache `models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/1df8507…` | shard blobs `9a080ebf…`, `b8bc9837…`, `557341a0…`, `6ba8a9c8…` | pinned | **yes** (revision) |
| Config | `configs/mvp.yaml` (Stage 7) / `configs/smoke.yaml` (Stage 6) | resolved hash `9bad4b36…960ff` / scientific `14e279c5…fabd7` for mvp | repo | **yes** |
| Code | commit of the implementing branch, clean tree | – | – | yes (manifest identity) |

CONFIRMED trace facts: `full_cot` starts with the stored `prompt` in 40/40; every prompt
ends with `<think>\n`; tokenising `full_cot` with the pinned fast tokenizer and
`add_special_tokens=False` (the Stage 5 convention, `ADD_SPECIAL_TOKENS = False`) round-trips
exactly via `decode`; 187,579 tokens total, min 1,537, max 13,087 (`3935:incorrect`),
mean 4,689. Tokenizer: `LlamaTokenizerFast`, `add_bos_token: true` in config, BOS
`151646`; the model's `bos_token_id` is `151643`. Stage 5 sent no BOS; Stage 6 must match.

## C. Output contract

Specified (SPEC §5, §11 Stage 7):

- `artifacts/runs/<run_id>/activations/<trajectory_id>.safetensors` with tensors
  `resid: [n_sentences, 49, 5120]` float16 and `final_prev_position: [n_sentences, 5120]`
  float16 (`layer=48` at `boundary_token_idx − 1`; NaN row and index flag when that index
  is negative — never occurs here: minimum boundary index is 60).
- `activations/index.parquet` with the §5 "Activation index" columns
  (`sentence_id, trajectory_id, shard_path, row_idx, n_layers=49, d_model=5120,
  dtype=float16, activation_site=block_post_pre_final_norm, final_prev_position_available,
  model_id, model_revision, tokenizer_revision, trace_sha256, token_ids_sha256`), written
  with `write_parquet` (schema sidecar + `.sha256`).
- Stage 6 smoke: one shard, a boundary/token audit, a max-difference report.
- Stage 7: `data/processed/features.parquet`, `features/feature_audit.json` (observables,
  out of scope here), `stages/stage7/resource_projection.json`.
- Every file atomic with `.sha256` sidecar; manifest `input_hashes`, `output_hashes`,
  `stages.stage6/7.state`, hardware, peak GPU memory (AGENTS §6).

Open implementation choices (not scientific): shard filename (`330_correct.safetensors`
rather than a colon, matching Stage 5's `4682_correct`); storing per-shard metadata
(`token_ids_sha256`, boundary indices, `target_token_start`) in safetensors header
metadata plus a `boundaries.parquet`; UnitStore keying per trajectory.

## D. Activation extraction design

CONFIRMED definitions (SPEC §3.8, AGENTS §5.1–5.2, `configs/mvp.yaml`):

- Sequence: the exact stored `full_cot` (prompt + `<think>` trace) tokenised once, fast
  tokenizer, offsets, `add_special_tokens=False`. No chat template, no BOS.
- Boundary: `target_token_start = min k : offset_end[k] > char_start`;
  `boundary_token_idx = target_token_start − 1`. A token overlapping `char_start` is target.
- Sites: `layer 0` = `model.model.embed_tokens` output; `layer i (1..48)` =
  `model.model.layers[i-1]` output `[0]` (transformers 4.53.2 returns a tuple), i.e. the
  residual stream after the block and before `model.model.norm`. bf16 compute, fp16 store.
- Replay: one full causal pass per trajectory, batch 1, `inference_mode`, `use_cache=False`,
  `output_attentions=False`, no `output_hidden_states`; hooks index `[0, boundaries, :]`
  and move to host immediately.
- Load: `AutoModelForCausalLM.from_pretrained(id, revision, torch_dtype=bf16,
  attn_implementation="sdpa", device_map={"": 0})`, assert no CPU/meta params, 48 layers,
  d=5120, `tie_word_embeddings=False`.

CONFIRMED by CPU audit of all 6,474 rows: 0 negative boundaries; sentence 0 always starts
after the prompt (min `char_start` 168); 2,844 rows (1,930 eligible) begin after a space
where the full-trace token is `" Word"`, so the boundary token is the last token of the
previous sentence and the space belongs to the target (spec rule). Eligible boundary
indices span 62–8,921.

Proposed modules (SPEC §6 layout): `model/load.py`, `model/boundaries.py`,
`model/hooks.py`, `model/logits.py`, `features/extract.py`, `features/store.py`, CLI
`extract-activations --config … [--trajectory --limit-sentences] [--resume --run-id]`.

## E. Resource model

Tensor arithmetic (CONFIRMED counts): 6,474 rows × 49 × 5,120 × 2 B = 3,248,271,360 B
(3.03 GiB) for `resid` over all rows; 2,016,747,520 B (1.88 GiB) if only the 4,019
eligible rows; `final_prev_position` adds 66.3 MB. Per trajectory 1.5–195 MB (measured:
`3935:incorrect`, 389 rows → 199,168,176 B, fp16 round-trip max error 3e-8, no overflow,
bf16 |max| 936).

Measured on this H100 (probe, disposable):

| Quantity | Value |
|---|---|
| Model load from NFS cache | 49.0 s |
| VRAM after load | 28,172 MiB |
| Peak VRAM, 13,087-token replay with 389-row capture | 29,974 MiB allocated / 31,156 MiB reserved |
| Forward, 13,087 tokens | 3.66 s cold, 3.01 s warm (3.6–4.3k tok/s) |
| Forward, 2,973 tokens | 0.40 s (7.4k tok/s) |
| Repeat replay bitwise equal | yes |
| Host RSS | 9.2 GB (221 GB available) |
| fp16 write / reload, 199 MB | 0.30 s / 0.16 s (local) |
| NFS write / read (1 GiB, fsync) | 230 MB/s / 176 MB/s; local root 1.0 GB/s |
| SDPA backend actually used | EFFICIENT_ATTENTION; FLASH unavailable ("No available kernel") |

INFERRED: Stage 7 forward time ≈ 187,579 tokens ÷ ~3.6k tok/s ≈ 55 s; safetensors
writes ≈ 3.3 GB ÷ 230 MB/s ≈ 15 s on NFS; total extraction ≈ 3–5 min wall including
load. Batch size is fixed at 1 by config; VRAM headroom is ~50 GB so no OOM path is
expected at ≤ 13k tokens (eager fallback would need 40 × 13,087² × 2 B ≈ 13.7 GB, still
fits). Bottleneck is model load, then NFS write; GPU compute is negligible.

## F. Measured-fit / smoke plan (SPEC Stage 6)

Workload: `configs/smoke.yaml` → `330:correct`, first 16 sentences (boundaries
69, 85, 111, 115, 159, 171, 181, 208, 235, 256, 268, 317, 335, 347, 364, 374; 2,973 tokens),
plus the single longest trajectory `3935:incorrect` as the memory-envelope unit. Must
establish: load and layer-count asserts; hook capture shapes `[16,49,5120]` fp16 and
`[16,5120]`; token/boundary audit (prefix ids equal full ids, every target char ≥
`target_token_start`); serialization + `.sha256` + reload equality; peak VRAM/RAM logged;
throughput; the two invariance tests and the logit-identity assert; kill (SIGTERM and
SIGKILL) during the second unit and `--resume` leaving the first unit's hashes unchanged
and quarantining the partial; `resource_projection.json` for Stage 7 from the measured
seconds per token and bytes per row. GPU time ≈ 3 min; cost ≈ $0.15.

## G. Resume and integrity design

Reuse existing primitives unchanged: `UnitStore.produce` (lock, quarantine, `SUCCESS.json`
with per-file hashes), `write_safetensors`/`write_parquet`/`write_json` (temp file, fsync,
rename, sidecar), `create_run`/`check_resume` (identity = resolved config, git SHA, diff
hash, lock hash, `input_hashes`), `Runtime` (heartbeat 30 s, SIGINT/SIGTERM →
`StageInterrupted`), `start_tracking`. Unit = one trajectory; key =
`cache_key(stage="stage7_activations", scientific_config_hash, model_revision,
tokenizer_revision, trace_sha256, segmentation_version, boundary_mapping_sha256=sha256 of
the sorted (sentence_idx, boundary_token_idx) list, generation_parameters={}, fork_idx=None,
sample_seed=None, code_version, input_hashes, unit_id=trajectory_id)`. Order of work is
the sorted trajectory list; `index.parquet` is rebuilt from validated units at the end so
partial runs never publish an index. Corrupt or partial unit → quarantine and recompute
that trajectory only; valid units are never rewritten. Manifest stage state:
`prepared → running → extracted → validated`, `output_hashes` filled from unit
`SUCCESS.json` entries, `failure.json` on exception.

## H. Validation gates for completion

1. Manifest identity (config, code, lock, input hashes) matches; model and tokenizer
   revision `1df8507…`; `attn_implementation == "sdpa"`, no CPU/meta parameter.
2. Per trajectory: `full_cot_sha256` equals the stored trace hash; `token_ids_sha256`
   recorded; decode round-trip equals `full_cot`; every sentence's boundary satisfies
   §10.2 (prefix ids == full ids prefix, target chars ≥ `target_token_start`, boundary ≥ 0).
3. Shapes `[n_rows, 49, 5120]` fp16, `[n_rows, 5120]` fp16; no NaN/Inf; row order equals
   ascending `sentence_idx`; `index.parquet` rows == stored rows == 6,474 (or 4,019 if
   eligible-only is chosen; see K3) with every eligible `sentence_id` present exactly once.
4. Every shard and the index pass `validate_artifact`/`validate_parquet`; `SUCCESS.json`
   hashes match; replaying one trajectory again is bitwise identical (measured true).
5. Scientific: mutation invariance exact (measured 0.0); prefix-vs-full within the frozen
   tolerance (see K1, currently failing for short prefixes); `lm_head(norm(layer48))`
   reproduces the model's logits at a boundary within the same forward pass.
6. Stage 8 coverage preconditions already satisfied by the label table: eligible train
   2,650 / validation 680 / test 689; anchors 265 / 54 / 25.

## I. Production execution plan (ordered)

1. Decide K1–K5 with the user; write any amendment under `artifacts/amendments/`.
2. Implement `model/*`, `features/extract.py`, `features/store.py`, CLI, unit tests with a
   tiny random Qwen2 (fp32, tolerance 1e-5), resume/quarantine tests; run
   `tests/unit tests/integration`; commit on `codex/stage6`.
3. `uv run preanchor doctor --config configs/mvp.yaml` (production scope, GPU on).
4. Stage 6 smoke in tmux: `extract-activations --config configs/smoke.yaml --trajectory
   330:correct --limit-sentences 16`, then the memory-envelope unit, kill/resume drill,
   `tests/scientific/test_preemission_invariance.py`; write `resource_projection.json`;
   record `docs/stage6_validation.md` and `docs/decisions.md`; commit.
5. Stage 7 extraction in tmux: `extract-activations --config configs/mvp.yaml --resume`
   (new run ID), 40 units; `validate` pass; index; backup snapshot; MacBook receipt.
6. Only then Stage 7 observables/features (`build-dataset`, `validate-features`).

## J. Time, storage, cost

| Item | Measured | Extrapolated | Assumptions |
|---|---|---|---|
| Smoke GPU time | load 49 s + 0.4 s + 3.0 s | ≈ 3 min with invariance replays and drills | one load per process |
| Stage 7 extraction | – | 3–5 min wall (≈ 55 s forward, ≈ 15 s NFS writes, 49 s load) | 3.6k tok/s, 230 MB/s |
| Output storage | 199 MB per 389 rows | 3.31 GB all rows (+ index, sidecars, audits < 20 MB) | fp16 |
| Temporary headroom | – | ≤ 2× output for quarantine/rewrites: 7 GB; NFS has 1.8 PB nominal, local 933 GB | – |
| Cost | – | smoke < $0.15, Stage 7 < $0.25, both < $0.50 at 2.49 USD/h (3.29 USD/h per SPEC table: < $0.65) | H100 PCIe hourly billing continues while the instance runs; wall clock dominated by human steps |

## K. Ambiguities and blockers

1. **BLOCKED — bf16 prefix-vs-full tolerance.** `configs/mvp.yaml` freezes
   `invariance_atol_bfloat16: 0.02`. Measured on the smoke trajectory: boundaries with
   prefix length 70 and 86 tokens differ from the full replay starting at layer 1 (max
   abs 4.5 and 2.0 at layer 48, 90,926 of 250,880 elements above 0.02); all 14 boundaries
   with prefix ≥ 112 tokens are bitwise identical; repeat runs are deterministic; the MATH
   backend differs by 4.0 as well; the mutation test (same length) is exactly 0 for every
   boundary. So the boundary state depends only on the prefix, but GPU bf16 kernels chosen
   for short sequences round differently, and 0.02 is below bf16 resolution at the observed
   residual magnitudes (ulp 2.0 at |x| in [256,512)). The spec forbids loosening the
   tolerance. Options for the user: (a) amend the test to compare at equal sequence length
   (prefix + mutated future tokens, which is the existing mutation test) and keep the
   truncated-prefix comparison as a reported diagnostic; (b) amend the tolerance to a
   relative or float32-reference criterion (fp32 weights fit on this H100, 59 GB); (c) run
   the truncated-prefix check only for boundaries ≥ 112 tokens. Any option is a frozen-item
   amendment needing approval. Extraction itself is unaffected.
2. **AMBIGUOUS — stage naming.** SPEC Stage 6 is the smoke; full extraction is Stage 7.
   Confirm whether the user authorises Stage 7 extraction under this plan.
3. **AMBIGUOUS — rows to store.** SPEC shape `[number_of_sentences, …]` and the §9
   estimate (5,768 rows pre-filter) imply all rows; the Stage 7 validation says "all
   eligible rows have an activation". Recommendation: all 6,474 rows (3.3 GB, supports the
   K=4 convergence sensitivity whose eligible set differs, 3,902 rows); eligibility stays a
   join on `labels.parquet`.
4. **INFERRED conflict to record, not a blocker.** For the 2,844 space-initial sentences,
   the label-generating prefixes (Stage 4 released forks and Stage 5 vLLM prompts
   `full_cot[:char_start]`) tokenise to the full-trace prefix plus one `" "` token (2,816
   cases; Stage 5 forks 6, 7, 12, 14, 47 affected), while the Stage 6 boundary is one token
   earlier. This is the spec's own choice (§3.8) and belongs in
   `docs/alternative_explanations.md` under tokenisation artifacts.
5. **UNRESOLVED gate — Stage 5 secondary backup** not verified (`backup.verified: false`).
   Per precedent the next stage waited for the MacBook receipt.
6. Minor: FLASH SDPA kernel is unavailable with transformers 4.53.2's explicit mask; the
   efficient kernel is used. Record in the run, no action.

## L. Expected code changes

- New: `src/preanchor/model/load.py` (pinned bf16 load, device/layer asserts, tokenizer
  checks), `model/boundaries.py` (offset mapping → boundaries, audits), `model/hooks.py`
  (boundary-selective capture), `model/logits.py` (fp32 `lm_head(norm(h))`, next-token
  statistics, no full-vocab storage), `features/extract.py` (per-trajectory units, resume,
  heartbeat, projection), `features/store.py` (shard writer/reader, index builder,
  validator).
- CLI: `extract-activations`, `validate-activations` in `cli.py`; `Makefile` targets.
- Tests: `tests/unit/test_boundaries.py`, `tests/unit/test_hooks_tiny_qwen2.py` (random
  tiny Qwen2, fp32 tolerance 1e-5), `tests/integration/test_activation_resume.py`
  (SIGTERM/SIGKILL, quarantine, no rewrite), `tests/scientific/test_preemission_invariance.py`
  (real model, GPU, skipped without CUDA) — its tolerance semantics depend on K1.
- Docs: `docs/stage6_validation.md`, `docs/data_dictionary.md` (activation index),
  `docs/decisions.md`, `docs/alternative_explanations.md` (K4), `docs/continuation.md`.
- No change to `configs/mvp.yaml` unless K1 is amended; `configs/smoke.yaml` unchanged.
