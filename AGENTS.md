# AGENTS.md

Persistent operating rules for coding agents in this repository.
`IMPLEMENTATION_SPEC.md` (cited as `SPEC §n`) owns the execution plan,
schemas, commands, stage gates and acceptance criteria. This file states
only what must hold in every session. Where the two overlap, `SPEC` is
the detailed authority; the rules here are guardrails, not a second plan.

## 1. Purpose and scope

This repository implements the MVP of `research/project-brief.md`
(*Predicting High-Leverage Reasoning Transitions Before They Occur*):
can the model's internal state immediately before any token of a
reasoning sentence `S_i` is emitted predict whether `S_i` will later have
high counterfactual importance for the final answer?

The MVP is the replay-first experiment of `SPEC §3`: released Thought
Anchors trajectories and labels, exact replay through the pinned model,
boundary-only activation extraction, problem-grouped linear probes
against strict observable baselines, one small fresh resampling pilot, a
locked test evaluation, and a reproducible report. Implement it exactly
before adding extensions. Do not expand the project into a general
interpretability platform.

The probe is a passive, offline, correlational observer. It must never
become a training or reward signal (`SPEC §1, §2`).

## 2. Authoritative documents

Read, in order: `AGENTS.md`, `IMPLEMENTATION_SPEC.md`,
`research/project-brief.md`.

- The brief is authoritative for the research question, estimand and
  scope. If the spec conflicts with it on the estimand, preserve the
  brief's meaning and record the conflict.
- If implementation evidence shows the spec is technically wrong or
  infeasible, do not work around it silently: record the issue, choose
  the smallest credible resolution, and follow the amendment protocol
  (§4).
- Do not regenerate or replace `AGENTS.md`. Amend it only when
  explicitly instructed.
- Maintain `docs/decisions.md` (every result-affecting choice, tagged
  `fact | assumption | choice | interpretation`, with commit and the
  alternative rejected) and `docs/alternative_explanations.md` (for each
  positive finding: position, prefix-length, uncertainty,
  tokenisation-artifact and label-noise explanations, and the experiment
  that tests each). Other required documents are listed in `SPEC §6`;
  Stage 0 creates stubs for all of them.

## 3. Execution order and fail-fast policy

Follow the stage order of `SPEC §11`. For each stage: make one minimal
example work, validate every intermediate artifact against its schema
and checksum, run the smoke configuration end to end, prove
kill-and-resume, and only then run at MVP scale.

Order work by information gain per unit time. The riskiest question is
whether pre-emission activations beat the combined observable baseline;
a small end-to-end answer to it precedes polishing any single stage. A
null on the smoke run triggers a bug hunt before any conclusion; a
genuine null is a valid, reportable outcome (`SPEC §13`). Never tune,
hide or omit negative results.

Prefer simple implementations and the pinned upstream code over new
abstractions. Scientific logic lives in `src/preanchor/`, never only in
notebooks.

## 4. Frozen scientific choices and amendments

`configs/mvp.yaml` freezes: model, revision, dtype, context length,
`quantization: null`; dataset, revision, the 20 problems and the fixed
train/validation/test split; sentence eligibility and convergence rules;
the importance definition, the three label fields and the audit gate;
anchor-threshold construction; the pre-emission boundary and the 49
activation sites; the primary feature schema and its exclusion list;
probe families, grids, grouped CV and selection; the primary metric; the
minimum experiment set (`SPEC §3.2-3.12`).

Never change any of these silently. A necessary change MUST be written to
`artifacts/amendments/<utc-timestamp>.md` with reason, evidence and the
alternative rejected; MUST produce a new resolved-config hash and a new
`run_id`; MUST be logged in `docs/decisions.md`; and, for the items
above, MUST be approved by the user before execution.

## 5. Scientific invariants (hard constraints)

### 5.1 Pre-emission means genuinely pre-emission

- The boundary for `S_i` is `target_token_start − 1`, computed on the
  token ids of the exact stored trace with offset mappings; a token
  overlapping `char_start` belongs to the target and never enters the
  prefix (`SPEC §3.8`).
- Tokenise the exact stored trace once, with the pinned fast tokenizer
  and the upstream special-token behaviour, and no new chat template.
  Never rebuild a trace by joining sentences; never re-tokenise a
  sentence substring to locate its tokens.
- Primary features MUST contain nothing from `S_i` or later tokens,
  from completed-trajectory length, or from eventual correctness. The
  allow-list is `SPEC §3.9`; excluded quantities exist only as
  `diagnostic__*` columns.
- `tests/scientific/test_preemission_invariance.py` MUST pass:
  prefix-only replay equals full-trace replay at the boundary within the
  stated tolerances, and mutating every target and future token leaves
  the boundary state unchanged. Never loosen a tolerance to pass.

### 5.2 One hidden-state convention

`layer=0` is the embedding output; `layer=1..48` are decoder-block
outputs before the final RMSNorm (`SPEC §3.8`). Assert the layer count at
load and that `lm_head(final_norm(layer48))` reproduces the model's
logits at one boundary. Next-token statistics are computed in float32
from stored final-layer boundary states; full-vocabulary logits are never
stored.

### 5.3 Preserve the target variable

- `importance_canonical` is exactly the smoothed KL of `SPEC §3.6`, the
  brief's `KL(P(A | different S_i) ‖ P(A | S_i))`, over the union of
  normalised answer categories including `<NO_ANSWER>`.
- Counterfactuals are model-resampled continuations at the boundary
  with the manifest decoding parameters and the similarity filter.
  Deletion, blanking, or paraphrase by another model are prohibited
  substitutes.
- `importance_primary` follows the Problem 4682 audit gate; if the gate
  fails, recompute canonical labels and do not change the formula.
- Never substitute a superficially similar attribution metric without
  an amendment record.

### 5.4 Leakage-safe splits and the test lock

- Splits are by `problem_id`, both trajectories together, asserted
  disjoint; no row-wise random splitter may exist in the codebase.
- Every scaler, threshold, feature selection, residualisation,
  hyperparameter and layer choice is fitted on training folds only; the
  calibrator reads validation rows only.
- Never inspect test metrics before `frozen_selection.json` exists.
  `evaluate-test` runs once against the recorded freeze hash; any
  post-freeze selection change invalidates the run and requires a new
  `run_id` (`SPEC §11 Stage 9`).
- The four test problems are an exploratory held-out engineering test
  and every report says so (`SPEC §3.4`).

### 5.5 Baselines, controls, simple models

- Implement probes, baselines, grouped CV, calibration, metrics,
  controls and sensitivities exactly as `SPEC §3.10-3.12`. No probe
  result is reported without the same-split observable baselines (given
  the same tuning budget), the per-layer curve, the residualised
  control, lead-time one, the sensitivities and the permutation null.
- Linear and logistic probes only; no nonlinear predictor in the MVP.
- With four test problems, bootstrap intervals are descriptive; no
  strong significance claim.

### 5.6 Claim hygiene

Tag every metrics file `stage: exploratory | confirmatory` and
`preregistered: true | false`. Separate facts established by experiment,
implementation assumptions, methodological choices and interpretations.
State the final claim at the tier the evidence supports (one model, one
dataset, twenty problems) and as exploratory construct validation, never
as evidence of a deployable safety monitor (`SPEC §13`). Qualitative
examples are randomly sampled with the seed logged.

## 6. Model and library handling

- The MVP uses Hugging Face `transformers` with PyTorch forward hooks for
  extraction and the pinned vLLM container for generation. TransformerLens
  and nnsight are later work (`SPEC §12`); do not introduce them.
- Load with explicit `revision`, bf16, `attn_implementation="sdpa"`, an
  explicit single-GPU `device_map`, and fail if any parameter lands on
  CPU or meta. No quantisation for any measured quantity.
- Production extraction never uses `output_hidden_states=True`; hooks
  select boundary rows immediately and retain no full hidden-state
  tensors. One causal replay per trajectory, batch size one,
  `inference_mode`, `use_cache=False`, `output_attentions=False`.
- Generation and extraction run in separate processes; never hold two
  copies of the model. Stop vLLM and confirm with `nvidia-smi` that
  memory is released before extraction. Log peak GPU memory per stage.
- vLLM requests carry `temperature`, `top_p`, `max_tokens` and `seed`
  explicitly from the config; verify host and vLLM tokenisation agree
  before the pilot. Prefix caching may change speed, never outputs.
- Fetch the model by pinned revision and the dataset by explicit
  allow-list only; never download the whole dataset repository.
  `HF_HOME` lives on the persistent filesystem.

## 7. Validation gates

The tests of `SPEC §10` define "the smallest relevant test" for each
stage. Run the relevant ones after every material change, `tests/unit`
and `tests/integration` before every commit, and `tests/scientific`
before every MVP-scale stage. A stage MUST NOT run at MVP scale while its
gate in `SPEC §11` fails. Do not weaken a test to make an implementation
pass. Add a regression test for every bug that could have affected
results.

## 8. Compute and cost discipline

- The human launches and terminates Lambda instances manually. The agent
  MUST NOT use the Lambda API to create, resize, duplicate or terminate
  infrastructure, and MUST NOT issue `shutdown`, `poweroff` or `reboot`
  (Linux shutdown does not end billing). Never request a larger GPU class
  or more GPUs on its own initiative. Never create `SAFE_TO_TERMINATE`
  before its conditions in `SPEC §7.7` hold.
- Treat GPU time as metered. Before Stage 5, Stage 7, and any job
  expected to exceed one hour, run the representative benchmark named in
  `SPEC` (for example `pilot-labels --benchmark-only 8` or the Stage 6
  smoke extraction) and write
  `artifacts/runs/<run_id>/stages/<stage>/resource_projection.json`:
  unit count, measured seconds per unit with hooks or caches active,
  projected runtime, peak VRAM and RAM, disk growth, and projected cost
  at the hourly rate the user supplies. Refuse to start the stage if the
  projection cannot be computed, is older than the current commit, or
  exceeds the runtime or budget the user has authorised.
- Hardware admission is by measurement: `preanchor doctor` plus the
  Stage 6 smoke run on the actual GPU decide fit, not a table entry.
  The A100 40 GB configuration is admissible only at generation
  concurrency 1 and after that fit test (`SPEC §7.1`).
- CPU-only stages (probes, calibration, metrics, bootstrap, permutation,
  figures) run with `CUDA_VISIBLE_DEVICES=""` from stored features, one
  layer in float32 at a time.

## 9. Persistent storage and artifacts

- The instance root disk is ephemeral. Everything valuable lives under
  `$PREANCHOR_ROOT` on `/lambda/nfs/`; `bootstrap_lambda.sh` refuses
  production execution otherwise. Local SSD holds only regenerable
  scratch.
- Expensive outputs are written atomically with checksum sidecars and
  content-addressed cache keys exactly as `SPEC §9`; every table is
  Parquet with a sidecar schema and SHA-256.
- Back up per `SPEC §7.7` after every major stage and every 15 minutes
  during generation; never back up model caches or secrets; never upload
  raw activations to W&B.
- Never delete anything under `$PREANCHOR_ROOT`, including quarantined
  or superseded runs, without explicit user authorisation.

## 10. Observability, long-running jobs, resumability

- Long jobs run detached in `tmux` and never depend on an attached
  terminal or an interactive agent command.
- `heartbeat.json` is written from the main loop every 30 s with stage,
  progress, last durable output, measured rate, ETA and last error;
  `preanchor status` prints it with recent log lines. A stale heartbeat
  is a diagnosis trigger for the human, never a self-kill; low GPU
  utilisation is never a kill criterion.
- W&B carries the contents of `SPEC §8.1`; fall back to
  `WANDB_MODE=offline` automatically and sync later.
- Every expensive unit has a stable identity, an atomic completion
  record with a checksum, and a `--resume` path that validates completed
  parts, quarantines corrupt or partial files, refuses to mix runs with
  different config, code or input hashes, and reruns only missing work.
  Completed, checksum-valid samples are never rewritten. Failures write
  `failure.json` with enough context to diagnose without rerunning.

## 11. Environment and reproducibility

- `uv.lock` is the authority. If it fails on the actual Lambda image,
  change only the incompatible packages, rerun smoke tests, and record
  both lock hashes in `docs/environment_deviations.md`. Never run
  `apt full-upgrade` or `apt dist-upgrade`; never upgrade drivers or
  CUDA. GH200 (arm64) is not used unless the full lock is first verified
  on arm64.
- `preanchor doctor` MUST pass before any GPU stage. Verify the
  environment on the instance (`nvidia-smi`, `torch.utils.collect_env`)
  rather than assuming it.
- Every run's `manifest.json` carries the provenance fields of
  `SPEC §6` (commit and dirty diff, resolved config hash, all revisions,
  lock hash, hardware and driver, seeds, commands, stage timings, W&B
  URL, backup verification).
- All randomness derives from `seed_root` per `SPEC §3.13`; seeds are
  stored per sample and per fold. Stochastic generation is reproducible
  in configuration, not bit-for-bit; report counts and intervals
  accordingly.
- Reported metrics and figures MUST regenerate from stored predictions
  and metrics without the model or raw activations (`SPEC §10.7`).

## 12. Git, upstream code, secrets, external actions

- Work on `codex/mvp`, created from `main` after the Stage 0 bootstrap
  commit. Commit validated milestones with descriptive messages; inspect
  `git status` and diffs before substantial changes. Do not rewrite
  published history, force-push, merge into `main`, delete branches with
  useful work, or push to external repositories unless instructed.
- `vendor/thought-anchors` and `vendor/thought-anchors.com` stay at their
  pinned commits and are never patched. Wrap or port upstream routines
  only after checking the licence and retaining notices; without a
  permissive licence, reimplement independently. Record every reuse and
  behavioural difference in `docs/upstream_audit.md`.
- Secrets (`HF_TOKEN`, `WANDB_API_KEY`, SSH keys, backup credentials)
  come only from the untracked `.env` or environment variables. Never
  commit, print, log, or write them into manifests, W&B config, reports
  or artifacts.
- Network use: documentation, pinned dependencies, the pinned model and
  dataset files, the vLLM image, W&B. Streamlit and vLLM bind to
  localhost only; no port other than SSH is exposed. Do not publish
  results, create repositories, open pull requests, send messages, or
  modify upstream resources without explicit authorisation.

## 13. When to ask, definition of done

Work autonomously on ordinary implementation, debugging, tests, refactors
needed for correctness, retries within a stage, and non-frozen
amendments under §4.

Ask the user before: changing any frozen item, the research question or
the estimand; proceeding when two scientifically meaningful alternatives
cannot be distinguished with available evidence; any operation that could
destroy or overwrite persistent data; any stage whose projection exceeds
the authorised runtime or budget, or generation beyond the pilot's
32 (adaptive 100) samples per fork; any change of GPU class, count or
instance; credentials or identifiers the agent lacks (repository URL,
instance IP, SSH key, filesystem name, W&B key, backup destination);
publishing, pushing, merging or other external side effects. Do not ask
for approval of routine coding choices that can be safely tested and
reverted.

The MVP is complete only when every criterion in `SPEC §13` holds, with
concrete evidence per criterion (test output, artifact path and hash, or
metric file). There is no required positive effect size. When claiming
completion, provide the evidence table, the report contents required by
`SPEC §13`, and unresolved limitations, including public-data reuse,
four held-out problems, and the exploratory nature of the test split.
