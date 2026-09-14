# Lambda Mechanistic Interpretability Workload Runbook

**Layer:** Specialised workload runbook
**Depends on:** `Lambda Platform Specification.md`
**Also depends on:** `Lambda LLM Training and Fine-Tuning Runbook.md`, for any controlled fine-tuning phase
**Last reviewed:** 22 August 2026
**Scope:** Mechanistic interpretability experiments on Lambda Cloud, primarily using TransformerLens, NNsight, Hugging Face/PyTorch, SAELens, and related tooling.
**Primary workloads:** activation extraction, probing, logit attribution, activation and path patching, attribution patching, steering and ablation, sparse autoencoders, crosscoders/model diffing, and controlled fine-tuning used to create model organisms.

## 0. Inheritance

This runbook **inherits the Lambda Platform Specification**. That document is a hard dependency.

```text
Lambda Platform Specification
        │
        ├── Lambda LLM Training and Fine-Tuning Runbook
        │
        └── Lambda Mechanistic Interpretability Workload Runbook   <- this document
```

**Governing question:** given an execution environment already governed by the Lambda Platform Specification, how should a mechanistic-interpretability experiment be represented and executed in terms of compute, activations, interventions, storage, provenance, and resumability?

### 0.1 Inherited without restatement

This runbook does **not** define, and MUST NOT redefine:

```text
price discovery                storage lifecycle
Lambda API provisioning        watchdog design
cost authorisation             provider lifecycle
firewall policy                hard-TTL mechanism
authentication and SSH         termination verification
launch idempotency             post-run sweeping
```

> **Every MechInterp run MUST satisfy the launch authorisation, external hard-TTL, verified-termination, and residual-resource audit requirements defined by the Lambda Platform Specification.** The authorised-exposure relation \(C_{\max}=n\,r\,T_{\max}\) — with \(n\) the authorised instance count, \(r\) the live per-instance hourly price, and \(T_{\max}\) the hard TTL — is a platform invariant, as is the rule that no billable resource may exist without an owner, purpose, maximum exposure, and externally enforceable termination deadline.

Cost containment is therefore independent of whether the interpretability result is obtained. This runbook contributes only the **workload-specific inputs** to that machinery: the resource projection of section 12, the manifest of section 13, and the durable-output validation invoked by the platform success path.

### 0.2 Controlled fine-tuning

When a MechInterp project includes controlled fine-tuning to create a model organism, that training phase is executed under the **Lambda LLM Training and Fine-Tuning Runbook**. This runbook does not specify a second training workflow. The interpretability phase consumes the resulting model or adapter by its recorded revision, checkpoint identity, and training manifest.

Where the scientific question permits, prefer a LoRA model organism and preserve the adapter plus its training manifest, rather than a full fine-tune, unless full fine-tuning is scientifically required.

### 0.3 Owned by this runbook

```text
internal-state experiment specification    steering / ablation
activation memory                          SAEs
hooks / caches                             crosscoders / model diffing
probes                                     activation retention
activation patching                        intervention scaling
attribution patching                       MechInterp-specific resumability
```

---

## 1. Operating principle

Mechanistic interpretability places unusual demands on infrastructure. Ordinary inference mainly requires model weights and temporary forward-pass state. MechInterp deliberately exposes, retains, compares, modifies, or differentiates through intermediate states: residual-stream activations, attention-head outputs, attention patterns, gradients, clean/corrupted activation pairs, intervention sweeps, SAE features, and model-to-model activation comparisons.

The infrastructure therefore optimises for four properties:

1. **rapid experimental feedback**, especially during exploratory work;
2. **controlled access to model internals** without indiscriminately materialising them;
3. **reproducible causal interventions** whose hook state, model state, prompts, and activations cannot silently contaminate one another;
4. **bounded economic exposure**, even when an experiment crashes or produces unexpectedly large activation outputs.

The core operating principle is:

> **Treat GPU instances as disposable execution workers, activations as expensive intermediate state, experimental manifests and final evidence as durable state, and cost control as an independent system outside the research process.**

Research practice moves from fast feedback and high information gain during exploration toward increasingly rigorous hypothesis tests and skeptical validation. Infrastructure should support that progression rather than forcing every exploratory experiment into the cost structure of a confirmatory sweep.

### 1.1 Default architecture

For most projects, one Lambda On-Demand Cloud instance suffices; multi-node execution is a platform-level service decision (Platform Specification 3).

```text
Versioned repository + execution manifest
                 │
                 ▼
       Pre-launch resource estimator
   model + activation + intervention cost
                 │
                 ▼
      Lambda ODC GPU instance
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
Local ephemeral SSD    Lambda filesystem
hot activation scratch model cache
temporary caches       prompt/data manifests
intervention buffers   durable metrics
                       selected activations
                       checkpoints/SAEs
                              │
                              ▼
                       independent backup
```

The independent controller sitting outside this execution path is the platform cost-control plane, defined once in the Platform Specification (section 1) and not restated here.

---

## 2. MechInterp workload classes

Six materially different workload types exist. **"MechInterp experiment" is not a sufficient workload specification**; the class determines provisioning.

| Workload | Main bottleneck | Typical infrastructure risk |
|---|---|---|
| Logit lens, direct logit attribution, exploratory hooks | GPU inference | Keeping an oversized GPU alive interactively |
| Activation extraction and probes | Activation RAM/storage | Caching far more tensors than required |
| Activation/path patching | Number of forward passes | Combinatorial intervention sweep |
| Attribution patching | Forward + backward memory | Gradient and activation retention |
| SAE/crosscoder work | Token throughput + activation stream | Huge activation datasets and long training |
| Large-model NNsight-style interventions | Weight VRAM + transfer | Returning large internal tensors unnecessarily |

---

## 3. MechInterp-specific invariants

Platform invariants apply in full and are not repeated. The following are additional and specific to internal-state work.

```text
MUST treat full activation caches as opt-in rather than the default.

MUST specify exactly which activations, layers, positions, heads,
gradients, or statistics are required before a large collection run.

MUST estimate model memory and activation memory separately.

MUST estimate attention-pattern memory with its S^2 dependence.

MUST microbenchmark hooks and interventions before projecting runtime.

MUST calculate intervention cardinality before patching sweeps.

MUST perform a backward-memory fit test for gradient-based methods.

MUST establish the behavioural effect before expensive MechInterp.

MUST scope and reset hooks between experiments.

MUST avoid transferring internal tensors that can be reduced remotely.

MUST preserve raw experimental metadata sufficient to reconstruct
every causal intervention, including prompt IDs, token positions,
hook definitions, and model revisions.

MUST separate reusable scientific artifacts from disposable
activation scratch.

MUST make large intervention sweeps resumable.

SHOULD use scalable attribution to screen before exhaustive patching.

SHOULD separate expensive activation generation from cheap
downstream analysis.

SHOULD stream SAE activation/data pipelines unless persistent
materialisation has a demonstrated reuse case.
```

The declaration each run must make — `project`, `run_id`, `owner`, `instance type`, `instance count`, `expected runtime`, `hard TTL`, `expected cost`, `maximum authorised cost` — is the platform launch authorisation record (Platform Specification 11.2). This runbook adds `model revision` and `workload type` (section 2) as required workload fields of the execution manifest.

---

## 4. Size workloads from tensors, not parameter count alone

### 4.1 Model weights

For a model with \(P\) parameters and \(b_w\) bytes per parameter,

\[
M_{\text{weights}}\approx P b_w,
\]

so BF16 or FP16 weights alone require approximately

\[
M_{\text{weights}}\approx 2P\ \text{bytes}.
\]

An 8B model therefore requires roughly 16 GB for weights alone, before framework overhead, temporary buffers, activations, KV cache, hooks, or intervention state. A 70B model requires about 140 GB of BF16 weights before overhead; NNsight documentation similarly notes roughly 140–280 GB depending on representation, which motivates remote or distributed execution.

> **"The model weights fit" is an insufficient admission criterion for a MechInterp workload.**

Require **measured headroom after the actual hooks and caches are active**.

### 4.2 Residual-stream activation memory

A residual activation typically has shape \([B,S,d]\), with \(B\) the batch size, \(S\) the sequence length, and \(d\) the residual-stream width. Caching one such tensor at every layer requires approximately

\[
M_{\text{resid}}
=
B S d L b_a,
\]

where \(L\) is the number of layers and \(b_a\) is bytes per activation element.

For an illustrative 32-layer model with \(d=4096\), \(B=8\), \(S=512\), and BF16 activations,

\[
M_{\text{resid}}\approx 1\ \text{GiB}
\]

for **one residual-stream location per layer**. Caching `resid_pre`, `resid_mid`, and `resid_post` multiplies this by roughly three before Q/K/V, head outputs, MLP activations, logits, or gradients are included.

### 4.3 Attention patterns

Attention patterns have shape approximately \([B,H,S,S]\), giving all-layer memory

\[
M_{\text{attn}}
=
B H S^2 L b_a.
\]

The quadratic dependence on context length makes attention-pattern caching substantially more dangerous than residual-stream caching. For \(B=8\), \(H=32\), \(L=32\), BF16:

- \(S=512\): approximately **4 GiB**;
- \(S=2048\): approximately **64 GiB**.

These are derived tensor-storage calculations before framework overhead: planning quantities, not measurements.

```text
DO NOT cache full attention matrices by default.

Prefer selected:
    layers,
    heads,
    destination positions,
    source positions,
    or summary statistics.
```

This matters particularly because TransformerLens patching utilities permit interventions down to layer, head, destination position, and source position.

---

## 5. Activation retention

Every run MUST declare an **activation retention plan**:

```yaml
activations:
  required:
    - hook: blocks.*.hook_resid_pre
      layers: [12, 13, 14, 15]
      positions: [-1]
      dtype: bfloat16

  gradients: false
  attention_patterns: false

  persistence:
    mode: summary_only

  scratch_ttl_hours: 6
```

Valid persistence modes:

```text
none
summary_only
selected_activations
full_required_cache
materialized_activation_dataset
```

The default is `summary_only`. Persist activation tensors only when they will be reused enough to justify the storage and transfer cost. Scratch retention is bounded by its own TTL; this is a data-hygiene control that operates inside the run and does not replace the platform hard TTL.

### 5.1 Durable versus disposable

Normally durable:

```text
experiment configuration
Git commit and patch
environment manifest
model/tokenizer revision
prompt/example manifest
token IDs where position-level analysis matters
clean/corrupted pairing
hook/intervention specification
behavioural baselines
patching/attribution result matrices
probe/steering directions
SAE/crosscoder checkpoints if scientifically valuable
selected validation activations
plots
statistical analyses
logs sufficient to diagnose failures
actual resource/cost record
```

Normally disposable:

```text
full unfiltered activation caches
attention matrices used once
temporary CPU-offload buffers
download staging
duplicate model copies
compiled scratch
failed-run partial tensors
temporary intervention outputs already reduced to metrics
```

**The distinction MUST be decided before launch rather than after the filesystem fills.** Placement of each class follows the platform storage policy (Platform Specification 7.1).

---

## 6. TransformerLens

TransformerLens is appropriate when the target model is supported and local execution provides enough VRAM.

TransformerLens defaults to caching every matching hook when no `names_filter` is supplied, and provides `names_filter` and `pos_slice` specifically to restrict what is retained. Therefore:

> **Never use full `run_with_cache` blindly on a large model or large batch.**

Use these controls deliberately:

```text
names_filter
    cache only relevant hook names.

pos_slice
    retain only relevant token positions.

incl_bwd
    enable only when gradients are scientifically required.

device
    choose activation-cache location deliberately.

reset_hooks_end
    normally keep enabled so interventions do not leak into later runs.
```

Both `names_filter` and `pos_slice` MUST reflect the exact hypothesis being tested.

TransformerLens warns that moving cached activations to a device different from the model can cause significant performance degradation. CPU offloading can therefore relieve VRAM pressure while becoming a PCIe/host-RAM throughput bottleneck. Benchmark it rather than assuming it is free.

### 6.1 Hook scoping and reset

Temporary hooks SHOULD be scoped to context managers and reset after execution; TransformerLens implements automatic hook removal when `reset_hooks_end=True`. Hook leakage across experiments is a contamination failure, not a performance issue.

### 6.2 TransformerLens provenance

Record:

```text
TransformerLens version
model identifier
model revision
tokenizer revision
TransformerLens model configuration
hook names
hook layers
hook positions
dtype
device mapping
any TL weight-processing/folding options
```

---

## 7. NNsight and the local-versus-NDIF decision

NNsight constructs an intervention graph that can read and modify internal states and can send that graph to remote execution. Two conceptually different uses exist:

```text
NNsight local execution
    Lambda owns the model and GPU.
    remote=False.

NNsight/NDIF remote execution
    NDIF owns the expensive model execution.
    Lambda may be unnecessary.
```

Before provisioning a large Lambda instance, ask:

> **Can the scientific question be answered through an available NNsight/NDIF model instead?**

If yes, paying for a Lambda VM merely to act as a client is usually unnecessary. This decision precedes platform `preflight`, because it determines whether a billable Lambda resource is needed at all.

### 7.1 Save the smallest possible object

Reduce tensors inside the tracing context before saving them. Saving an entire vocabulary-sized logit vector when only an argmax or scalar metric is required creates needless transfer. Prefer:

```text
scalar metric
top-k indices
selected position
selected layer
selected feature values
aggregated statistic
```

### 7.2 Keep coupled interventions in one session

NNsight sessions allow multiple interventions to reuse intermediate outputs without transferring those outputs to the client between traces. This suits clean/corrupted comparisons, cross-prompt patching, intervention sweeps, and iterative causal tests.

Use `scan=True, validate=True` during development where supported: scan mode performs fake-input shape and debug execution without the normal model-memory cost, allowing indexing and intervention errors to be found before expensive execution.

---

## 8. Stage experiments by information per dollar

Cloud execution mirrors the research progression from cheap exploration to expensive confirmation.

### Stage A: behavioural and functional baseline

Before caching internals:

```text
verify prompts;
verify tokenization;
measure behavioural/logit effect;
establish clean vs corrupted contrast;
check metric sign and scale.
```

This is cheap and prevents expensive interpretability analysis of a nonexistent phenomenon.

### Stage B: cheap localization

Prefer inexpensive analyses first:

```text
logit difference
direct logit attribution
logit lens
selected residual-stream extraction
difference-in-means
linear probe
small steering test
```

### Stage C: scalable causal screening

Ordinary activation patching requires a separate forward pass per patched activation, while attribution patching can estimate all candidate contributions using roughly two forward passes and one backward pass. The default cost-efficient strategy is therefore:

```text
broad attribution screen
        ↓
top candidate layers/heads/positions
        ↓
exact activation patching
        ↓
path-specific or finer-grained intervention
```

Do not begin with an exhaustive exact patching grid on a large model unless the scientific design specifically requires it.

---

## 9. Intervention scaling and patching budgets

### 9.1 Intervention-count estimation

Generic patching implementations iterate over intervention indices and run the relevant patch for each candidate site. With \(L\) layers, \(H\) heads, and \(S\) token positions, a layer-head-position sweep may require on the order of

\[
N_{\text{patch}}\sim LHS
\]

patched runs, and patching individual attention entries may reach

\[
N_{\text{patch}}\sim LHS^2 .
\]

Before launch, calculate:

```yaml
patching:
  candidate_layers: ...
  candidate_heads: ...
  candidate_positions: ...
  total_interventions: ...
  measured_seconds_per_intervention: ...
  projected_gpu_hours: ...
```

### 9.2 Runtime estimation from measured intervention cost

After a microbenchmark,

\[
T_{\text{patch}}
\approx
N_{\text{patch}}\,t_{\text{measured}}.
\]

\(t_{\text{measured}}\) MUST be measured under the actual hook and cache configuration. A projection built from unhooked throughput is invalid.

If the projected cost exceeds the experiment budget:

1. reduce candidate space using theory or cheap attribution;
2. batch interventions where semantically valid;
3. restrict positions;
4. restrict layers;
5. use attribution patching to screen;
6. run exact patching only on candidates that affect the claim.

Path patching is more operationally complex still: it may require clean and corrupted caches, frozen components, sender interventions, and additional downstream runs. Narrow the candidate graph before executing it.

### 9.3 Attribution patching is a gradient workload

Attribution patching changes the resource regime. It requires:

```text
clean activations
corrupted activations
corrupted gradients
backward graph state
```

with the attribution statistic

\[
\nabla a_{\text{corrupt}}
\cdot
(a_{\text{clean}}-a_{\text{corrupt}})
\]

computed from cached outputs and gradients across layers.

Therefore:

- perform a **dedicated backward-pass memory fit test**;
- do not infer fit from forward-only inference;
- restrict stored activation sites;
- immediately reduce gradients to the required attribution statistic where possible;
- release the autograd graph and unused caches between batches.

Attribution patching is a better **screening** workload than exhaustive activation patching. Its linear approximation still requires exact causal validation for conclusions that depend on precise effect size: scalability and approximation error are separate properties.

---

## 10. Probes, feature directions, steering, SAEs, and model diffing

### 10.1 Probes and feature directions are two-phase jobs

```text
Phase 1: expensive model forward passes
    ↓
selected activation extraction
    ↓
Phase 2: cheap statistical analysis
```

Do not keep an expensive GPU running while fitting small linear models on a cached activation matrix if the operation can run on CPU or inexpensive hardware. If activations fit in RAM, perform the analysis before writing them permanently.

If they must be reused:

```text
chunk
preserve prompt/example IDs
preserve token positions
preserve layer/hook metadata
preserve dtype
write manifest + hashes
```

**The feature direction alone is not sufficient provenance.** Store the dataset and extraction specification that generated it.

### 10.2 Steering and ablation

Batch interventions within a single loaded-model session rather than reloading per intervention. Steering strength and ablation type are experimental-unit dimensions (section 16) and MUST be recorded in the manifest.

### 10.3 SAE workloads

SAE latent dimension can be far larger than the model residual width; one documented example has \(d_{\text{in}}=768\) and \(d_{\text{SAE}}=24576\). SAE projects therefore combine base-model inference, high-volume activation generation, SAE training, large feature spaces, and evaluation over large token corpora.

Use:

```text
pre-tokenized dataset where available;
streaming when repeated full local materialization is unnecessary;
on-the-fly activation generation when cheaper than persistent activation storage;
materialized activations only when they will be reused enough to amortize generation.
```

For a materialized activation corpus,

\[
M_{\text{dataset}}
=
N_{\text{tokens}}\,d\,b_a .
\]

At one billion tokens, \(d=4096\), and BF16, a single residual-stream activation per token requires roughly

\[
8.2\times10^{12}\ \text{bytes},
\]

several tebibytes. **"Save all activations and decide later" is an unacceptable default for large SAE work.** Use streaming or chunked generation and write only the activation stream the SAE or analysis requires.

### 10.4 Crosscoders and model diffing

Model diffing treats activation-space differences, SAEs, crosscoders, logits, weights, and behaviour as different levels of comparison. Paired activation analysis may require \(a_A(x)\) and \(a_B(x)\) for models \(A\) and \(B\), which creates three execution choices.

| Strategy | Advantage | Cost |
|---|---|---|
| **Concurrent models** — load both simultaneously | simpler pairing; minimal intermediate disk; high throughput if VRAM permits | roughly additive weight footprint |
| **Sequential with persisted activations** — run A, store, release, run B | lower peak VRAM | potentially enormous storage and I/O |
| **Sequential streaming comparison** — generate corresponding chunks, compute the statistic, discard raw activations | bounded storage | requires the quantity to be reducible online |

Prefer sequential streaming comparison whenever the scientific quantity can be reduced online.

Always preserve the paired-model alignment metadata:

```text
shared prompt/example ID
tokenized input for each model
token-position alignment rule
model revision A
model revision B
hook definition A
hook definition B
normalization/preprocessing
comparison statistic
```

**A cross-model "same layer" assumption MUST NOT be implicit if architecture or depth differs.**

---

## 11. Internal-state data-transfer policy

MechInterp can produce far more internal data than it needs to export. Generic staging, filesystem lifecycle, and backup rules are platform-level (Platform Specification 7); the following is the internal-state specialisation.

**Into Lambda** — stage before premium GPU launch where possible:

```text
repository
container/environment lock
prompt datasets
small metadata
model weights or reusable HF cache
SAE weights
pretokenized datasets
```

Persistent model caching and the local-SSD copy-at-startup pattern are platform staging behaviour (Platform Specification 7.4); what is MechInterp-specific is which artifacts are worth caching, listed above.

**Within the instance** — prefer local NVMe/root SSD for:

```text
temporary activation chunks
attention matrices
intervention buffers
temporary model copies
compilation caches
large disposable tensors
```

and persistent storage for:

```text
experiment manifests
prompt datasets
probe weights
feature directions
SAE/crosscoder checkpoints
patching result matrices
selected validation activations
final metrics
logs needed for reconstruction
```

**Out of Lambda** — export evidence, not every intermediate tensor:

```text
scalar metrics
heatmaps / patch matrices
top-k components
feature statistics
probe weights
selected exemplar activations
small reproducibility subsets
experiment manifests
logs
final model/SAE artifacts where required
```

Saving the smallest possible object is the correct general rule for all remote MechInterp infrastructure.

---

## 12. MechInterp resource projection

The platform authorises exposure; this runbook supplies the projection that makes the authorisation meaningful.

Total cost is better represented as

\[
C_{\text{total}}
=
C_{\text{model execution}}
+
C_{\text{intervention multiplicity}}
+
C_{\text{activation storage}}
+
C_{\text{data transfer}}
+
C_{\text{training}}
+
C_{\text{failed/idle time}} .
\]

For patching,

\[
C_{\text{patch}}
\approx
r\,N_{\text{patch}}\,t_{\text{patch}} ,
\]

and for activation extraction,

\[
C_{\text{extract}}
\approx
r\,\frac{N_{\text{tokens}}}{q_{\text{measured}}},
\]

where \(q_{\text{measured}}\) is measured token throughput **under the actual hook and cache configuration**. Throughput measured with no hooks is not a valid estimate for a run that transfers every layer's activations to CPU. \(r\) is the live hourly price obtained through platform discovery.

Before full execution, the workload MUST report:

```text
Expected temporary activation volume: ...
Expected persistent activation volume: ...
Expected model-cache volume: ...

Intervention count: ...
Measured interventions/sec: ...
Expected runtime: ...
Projected compute cost: ...
```

These lines are read together with the platform authorisation record, which supplies the live rate, hard TTL, maximum compute exposure, and account-wide burn before and after launch (Platform Specification 8.4, 11.2).

**Refuse the launch if the projection cannot be computed.** This is the workload-side instance of the platform rule that a resource whose maximum exposure is undetermined MUST NOT be launched.

---

## 13. The MechInterp execution manifest

The execution manifest is the **central abstraction of this runbook**. It is the machine-readable statement of what internal state a run will touch, how many times, at what cost, and what evidence it must leave behind. No expensive MechInterp execution proceeds without one.

Infrastructure identity and cost authorisation live in the platform contracts (`platform_snapshot.yaml`, `launch_authorisation.yaml`, `resource_registry.yaml`); the manifest references the run identity rather than restating the cost envelope.

```yaml
method: activation_patching        # one of the workload classes in section 14

model:
  name: ...
  revision: ...
  tokenizer_revision: ...
  dtype: ...
  device_map: ...

tooling:
  pytorch: ...
  transformers: ...
  transformer_lens: ...
  nnsight: ...
  sae_lens: ...

inputs:
  dataset_manifest: ...
  prompt_hash: ...
  token_ids_hash: ...
  sequence_length: ...
  chat_template: ...
  prepend_bos: ...
  clean_condition: ...
  corrupted_condition: ...

baseline:
  behavioural_effect: ...
  clean_reference: ...
  corrupted_reference: ...

metric:
  name: ...
  definition: ...
  expected_sign: ...

interventions:
  hook_name: ...
  layers: ...
  heads: ...
  positions: ...
  patch_direction: clean_to_corrupt
  ablation_type: ...
  strengths: ...
  total_interventions: ...

cache:
  names_filter: ...
  pos_slice: ...
  dtype: ...
  persistence: none | summary_only | selected_activations |
               full_required_cache | materialized_activation_dataset
  gradients: true | false
  attention_patterns: true | false
  scratch_ttl_hours: ...

resource_estimate:
  weight_memory: ...
  activation_memory: ...
  attention_pattern_memory: ...
  backward_memory: ...
  measured_seconds_per_intervention: ...
  projected_gpu_hours: ...
  expected_temporary_activation_volume: ...
  expected_persistent_activation_volume: ...

outputs:
  required_artifacts: ...
  export_reduction: ...

resumability:
  unit_key: ...
  completion_record: ...
  validation: ...

provenance:
  run_id: ...
  git_commit: ...
  dirty_patch_hash: ...
  hook_definitions: ...
  model_pair: ...
```

For an activation-patching workload, the manifest is therefore capable of expressing model revision, clean and corrupted inputs, the behavioural or logit metric, the hook, layers, heads, token positions, patch direction, cache dtype, activation-retention mode, intervention count, estimated resource burden, and required result artifacts.

**For clean/corrupted patching, the exact token IDs MUST be persisted.** Patching is defined in terms of corresponding internal states, so an unnoticed tokenization or positional mismatch can invalidate the interpretation even when the code executes without error.

---

## 14. Standard workload classes

Each class is a logical interface, not an implementation.

### `extract_activations`

- **Inputs.** Model revision; prompt/dataset manifest; hook set (`names_filter`); position selection (`pos_slice`); dtype; persistence mode.
- **Principal resource risk.** Activation RAM and storage growth; caching far more than required.
- **Durable outputs.** Selected activations or their summary statistics; extraction specification; prompt/example IDs; token positions; layer/hook metadata; dtype; manifest and hashes.
- **Resumability unit.** Prompt batch (× layer where chunked).

### `fit_probe`

- **Inputs.** Extracted activation matrix with its extraction specification; labels; probe family and hyperparameters; seed.
- **Principal resource risk.** Holding expensive GPU time for cheap statistical work.
- **Durable outputs.** Probe weights or direction; fit statistics; the generating dataset and extraction specification.
- **Resumability unit.** Probe fit (per layer/hook/seed).

### `run_attribution_patch`

- **Inputs.** Clean and corrupted inputs; metric; hook sites; gradient requirement; backward-memory fit-test result.
- **Principal resource risk.** Forward + backward memory; gradient and activation retention.
- **Durable outputs.** Attribution matrix over candidate sites; the attribution statistic definition; measured per-site cost.
- **Resumability unit.** Prompt batch × layer block.

### `run_activation_patch`

- **Inputs.** Clean and corrupted inputs with persisted token IDs; metric; candidate layers/heads/positions; patch direction; measured seconds per intervention.
- **Principal resource risk.** Combinatorial intervention count, \(N_{\text{patch}}\sim LHS\) or \(LHS^2\).
- **Durable outputs.** Patch result matrix; per-unit completion records; intervention specification.
- **Resumability unit.** `run / model / prompt batch / layer / head / position`.

### `run_steering_sweep`

- **Inputs.** Steering or ablation direction; hook site; strength grid; prompts; metric.
- **Principal resource risk.** Intervention multiplicity across strengths and prompts; model reload overhead if not batched.
- **Durable outputs.** Metric per (site, strength, prompt batch); direction provenance.
- **Resumability unit.** `run / model / prompt batch / site / strength`.

### `generate_sae_stream`

- **Inputs.** Base model revision; pre-tokenized or streamed dataset; hook site; token budget; chunking policy.
- **Principal resource risk.** Activation-corpus volume \(M_{\text{dataset}}=N_{\text{tokens}} d b_a\).
- **Durable outputs.** Activation chunks (only where materialisation is justified) or the streamed pipeline specification; chunk manifest and hashes.
- **Resumability unit.** Token/activation chunk.

### `train_sae`

- **Inputs.** Activation stream or corpus; SAE architecture including \(d_{\text{in}}\) and \(d_{\text{SAE}}\); training configuration; seed.
- **Principal resource risk.** Long training over a high-volume activation stream; large feature spaces.
- **Durable outputs.** SAE checkpoints; training metrics; the generating activation specification.
- **Resumability unit.** Training checkpoint over the chunk sequence.

### `compare_model_pair`

- **Inputs.** Model revisions A and B; hook definitions for each; alignment rule; shared prompt IDs; comparison statistic; execution strategy (concurrent / sequential persisted / sequential streaming).
- **Principal resource risk.** Additive weight footprint or paired activation storage.
- **Durable outputs.** Comparison statistic; full alignment metadata of section 10.4.
- **Resumability unit.** `run / model pair / prompt batch / layer`.

Any class that runs a controlled fine-tuning phase to produce a model organism delegates that phase to the Training Runbook.

---

## 15. Reliability stages

```text
no-GPU / static validation
        ↓
model + hook smoke test
        ↓
intervention sanity test
        ↓
micro-sweep
        ↓
full execution
```

### Stage 0: no-GPU / static validation

- **Establishes.** That the manifest is coherent and the run is affordable before any billable resource exists.

```text
parse configuration;
validate prompt files;
validate model identifiers;
validate requested hooks;
calculate intervention count;
calculate activation-storage upper bound;
calculate cost envelope.
```

Where possible, use NNsight scan/validate or lightweight model stubs to catch shape and index errors before real execution.

- **Progression evidence.** A complete manifest and a computable resource projection (section 12). An incomputable projection blocks platform `authorize`.

### Stage 1: model + hook smoke test

- **Establishes.** That the model and the requested internal access work on the provisioned hardware.

```text
load model;
tokenize representative prompts;
run unmodified forward pass;
record reference logits;
test selected hook;
verify tensor shape;
verify no unexpected dtype/device conversion.
```

- **Progression evidence.** Correct hook shapes and reference logits; measured headroom **with hooks and caches active**, not weights-only fit.

### Stage 2: intervention sanity test

- **Establishes.** That the causal machinery is wired correctly and does not leak.

```text
clean baseline behaves as expected;
corrupted baseline behaves as expected;
patch metric has correct sign;
identity patch produces approximately no change;
known-positive or intentionally large intervention produces detectable change;
hook disappears after its scoped execution.
```

- **Progression evidence.** All six checks pass. A wrong metric sign or a surviving hook invalidates any sweep built on it.

### Stage 3: micro-sweep

- **Establishes.** Measured per-intervention cost and real storage growth.

Run a small layer/head/position subset and measure:

```text
seconds per intervention
VRAM
host RAM
local disk growth
persistent disk growth
bytes exported
GPU utilization
```

- **Progression evidence.** The full cost recomputed from measured values, and found to fit inside the authorised envelope. If it does not, revise the candidate space or return to platform `authorize`.

### Stage 4: full execution

Only now run the expensive sweep, under the resumability requirements of section 16.

---

## 16. Large-sweep resumability

Many MechInterp experiments are collections of independent units. Define a **stable execution-unit identity** over the relevant dimensions:

```text
run
model
prompt batch
layer
head
position
feature
intervention strength
seed
model pair
```

for example

```text
<run_id>/<model>/<prompt_batch>/<layer>/<head>/<position>
```

Requirements:

1. **Atomic completion records.** A unit is complete only when its result and completion marker are durably written; partial writes MUST NOT be observable as complete.
2. **Completed-unit discovery.** On restart: load manifest → identify completed units → validate their hashes → continue only missing units.
3. **Validation of retained outputs.** Hash validation is part of discovery; an unverifiable unit is treated as missing.
4. **Selective continuation.** Only missing units execute.
5. **Deterministic aggregation.** Results aggregate identically regardless of completion order.

> **A large independent-unit sweep MUST NOT require restarting from zero because a later unit failed.** A 10,000-intervention patching sweep that fails at intervention 9,600 resumes at 9,600.

---

## 17. Interactive versus batch execution

Interactive visualisation and exploration create a specific cost risk: an instance remains running while the researcher reads papers, inspects plots, edits code, or leaves for the day. Use two declared modes.

**Interactive exploration**

```text
short TTL
cheap sufficient GPU
small prompts
small batches
selected layers
aggressive scratch cleanup
```

**Batch experiment**

```text
fully specified execution manifest
non-interactive process
structured outputs
resume support
heartbeat, soft deadline, hard TTL, and automatic termination
    (platform-enforced; see Platform Specification 8.5-8.7 and 9)
```

Do not leave an expensive GPU provisioned merely because a Jupyter kernel may be used again later. TTL registration and enforcement are platform mechanisms; the mode determines only which TTL and which hardware class is requested.

---

## 18. Method-specific default policy

| Method | Default cloud policy |
|---|---|
| Direct logit attribution / logit lens | Cheapest GPU that fits the model |
| Small probe / direction discovery | Extract selected activations, then terminate the GPU and analyse cheaply |
| Activation patching | Attribution or cheap screen first, exact top candidates second |
| Path patching | Narrow the candidate graph before execution |
| Attribution patching | Dedicated backward-memory microbenchmark |
| Steering / ablation | Batch interventions in one loaded-model session |
| SAE evaluation | Stream activations where possible |
| SAE training | Pretokenize/stream dataset; avoid a persistent raw activation corpus unless justified |
| Crosscoder / model diffing | Reduce paired activations online when possible |
| 70B+ internal access | Evaluate NNsight/NDIF before provisioning a large Lambda configuration |
| Controlled LoRA model organism | Execute under the Training Runbook; save adapter + training manifest; avoid full fine-tuning unless scientifically required |

---

## 19. Scientific reliability dominates infrastructure convenience

Causal intervention is stronger evidence than correlational localization; baselines, skepticism, and validation matter. Activation patching is designed around controlled clean/corrupted contrasts, while attribution patching trades exactness for scalability.

Cloud optimisation MUST therefore never silently alter:

```text
model revision
precision where numerically consequential
prompt/tokenization
intervention definition
hook location
clean/corrupted direction
batch composition
random baseline
normalization
metric
```

Hardware changes may be treated as infrastructure only after verifying that they leave the scientific quantity acceptably invariant. Where the primary comparison requires identical conditions, keep GPU class, library stack, dtype, and execution implementation fixed across conditions — the workload consequence of the platform prohibition on silent hardware substitution (Platform Specification 6.1–6.2).

---

## 20. Canonical MechInterp workflow

Platform operations are shown in `code` style.

```text
 1. Specify behavioural phenomenon and metric.
              ↓
 2. Verify phenomenon without internal instrumentation.
              ↓
 3. Pin model, tokenizer, prompt set, libraries, and hook semantics.
              ↓
 4. Define exactly which internals/interventions are required.
              ↓
 5. Estimate weight VRAM, activation VRAM, attention S^2 cost,
    intervention count, activation storage.
              ↓
 6. Decide Lambda-local vs NNsight/NDIF.
              ↓
 7. Stage code, model cache, prompts, and reusable artifacts.
              ↓
 8. Complete the execution manifest and resource projection.
              ↓
 9. `preflight` → `authorize` → `launch`     (Platform Specification 12)
              ↓
10. Model + hook smoke test.
              ↓
11. Intervention sanity test.
              ↓
12. Micro-sweep; re-estimate runtime/storage from measurements.
              ↓
13. Run cheapest useful localization method.
              ↓
14. Narrow candidate components.
              ↓
15. Run exact causal interventions where required, resumably.
              ↓
16. Persist compact evidence + provenance.
              ↓
17. Validate result reconstruction.
              ↓
18. `terminate` → `audit`                    (Platform Specification 12)
```

The audit step includes attention to large activation dumps, stale activation scratch, and obsolete model-cache copies, because a single accidental full-cache export can dwarf ordinary result storage. Classification of that debris is a MechInterp input to the platform sweep; the sweep itself, its enumeration procedure, and its terminal report are platform-defined.

---

## 21. Core MechInterp invariants

> **Localize cheaply, cache selectively, intervene narrowly, persist evidence rather than tensors, and make cost containment independent of experimental success.**

1. **Establish the behavioural effect before instrumenting internals.**
2. **Size from tensors, not parameter count: weights, activations, and attention patterns are separate budgets.**
3. **Declare retention before collection; selective caching is the default.**
4. **Measure intervention cost before projecting a sweep.**
5. **Preserve the provenance needed to reconstruct every causal intervention.**
6. **Make every large sweep resumable at the unit level.**
