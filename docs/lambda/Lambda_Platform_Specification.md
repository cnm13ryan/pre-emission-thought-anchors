# Lambda Platform Specification

**Layer:** Platform (root of the documentation hierarchy)
**Last verified:** 22 August 2026
**Scope:** Lambda Cloud platform, provisioning, security, storage, identity, lifecycle, and cost-control invariants for arbitrary GPU research workloads, primarily On-Demand Cloud (ODC) instances.
**Audience:** Anyone provisioning billable Lambda resources for research, and any automation, runbook, or tool that does so on their behalf.

## 0. Position in the documentation hierarchy

```text
Lambda Platform Specification            <- this document
        │
        ├── Lambda LLM Training and Fine-Tuning Runbook
        │
        └── Lambda Mechanistic Interpretability Workload Runbook
```

**Governing question:** given an arbitrary research workload, what Lambda platform, provisioning, security, storage, lifecycle, reproducibility, and cost-control invariants must hold independently of the scientific workload?

This document is the sole complete definition of provider lifecycle and cost-control behaviour. The specialised runbooks inherit it. They MUST NOT restate its mechanisms; they MAY state workload-specific consequences of them.

### 0.1 Normative language

- **MUST** / **MUST NOT** — an invariant. Violating it invalidates the run's cost-safety or reproducibility guarantees.
- **SHOULD** / **SHOULD NOT** — a strong default. Departures MUST be recorded with a reason.
- **MAY** — a permitted option with no default preference.

### 0.2 Layer boundary

This document owns:

```text
provider                   cost authorisation
authentication             watchdog
security                   lifecycle
API                        termination
live discovery             residual-resource audit
provisioning
storage
resource identity
```

This document does **not** own, and MUST NOT redefine: full SFT versus LoRA versus QLoRA; effective training batch size; FSDP2 or ZeRO-3 procedure; training seed policy; training checkpoint contents; LLM-as-a-judge methodology; TransformerLens; NNsight; activation-memory calculation; activation, path, or attribution patching; probes; SAEs; crosscoders; model-diffing methodology. Those belong to the specialised runbooks.

Where this document uses a training or interpretability example, it is **illustrative of a platform mechanism only** and confers no requirement on the specialised runbooks.

---

## 1. Purpose and operating principles

A cloud migration MUST change the **execution substrate**, not the scientific question.

The platform objective is an execution environment in which:

1. hardware, software, data, model revisions, and experimental configurations are explicitly recorded;
2. experiments survive SSH loss, instance failure, and capacity changes;
3. irreplaceable outputs survive instance termination;
4. hardware choice follows measured memory and throughput requirements;
5. infrastructure changes are distinguished from scientific interventions;
6. the researcher can prove what code, data, model, environment, and hardware generated every result;
7. every billable resource has a bounded, auditable cost lifecycle and an external termination failsafe.

The default architecture is:

```text
Versioned code + configuration + data manifest
                    │
                    ▼
        Lambda On-Demand GPU instance
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 ephemeral root SSD      Lambda filesystem
 scratch / caches        data / checkpoints /
 compilation             logs / manifests
                                │
                                ▼
                      independent backup
                                │
                                ▼
                   analysis / evaluation
```

A separate **cost-control plane** sits outside this execution path. It records launch authorisation, watches provider inventory and job heartbeat, estimates accrued spend, and enforces the hard TTL independently of the GPU VM.

```text
Independent controller
    ├── live price query
    ├── authorised cost envelope
    ├── resource registry
    ├── heartbeat monitoring
    ├── soft deadline
    ├── unconditional hard TTL
    └── post-run resource sweep
```

Treat GPU instances as disposable compute workers. Store valuable state independently of the instance.

For very small disposable pilots the root SSD MAY be sufficient, provided all required results are copied and hash-verified before termination. For substantive experiments, persistent storage MUST be used by default.

### 1.1 Invariant research policy versus mutable provider facts

Two categories of statement appear throughout this document and MUST remain distinguishable.

| Category | Nature | Handling |
|---|---|---|
| **Invariant research policy** | Stable rules about authorisation, identity, verification, and audit | Stated normatively here; changes only by revising this document |
| **Mutable provider fact** | Prices, inventory, quotas, identifiers, schema versions, regional coverage | Discovered live per run and recorded in `platform_snapshot.yaml`; never frozen into automation |

Section 13 enumerates the facts that MUST remain dynamic. Any value in this document that names a current provider state is a dated observation, not a guarantee.

---

## 2. Non-negotiable cost-safety invariants

Billing attaches to the **provider resource lifecycle**, not to whether useful research work is occurring. Every automation, runbook, or downstream tool derived from this specification MUST enforce these invariants ahead of convenience-oriented behaviour.

```text
 1. Never launch without querying the live provider price.
 2. Never launch without a declared maximum dollar exposure.
 3. Never launch without a hard maximum runtime (TTL).
 4. The hard-TTL controller must run outside the GPU instance.
 5. Never silently increase GPU class, GPU count, instance count, or hourly price.
 6. Never retry an ambiguous launch until the provider inventory proves that the
    prior request did not create the intended resource.
 7. Never interpret SSH loss, process exit, Linux shutdown, or low GPU utilisation
    as provider-side termination.
 8. During normal completion, never destroy valuable compute before durable-output
    verification.
 9. The hard TTL overrides normal completion safeguards when the maximum authorised
    exposure is reached.
10. Always verify termination through the provider API after issuing a termination
    request.
11. Always enumerate residual billable resources after the run.
12. Never use displayed service-credit balance as the real-time budget ledger.
```

The controlling economic invariant is:

> **No billable resource may exist without an owner, declared purpose, maximum authorised cost exposure, and externally enforceable termination deadline.**

### 2.1 The authorised-exposure relation

For an intended launch with \(n\) instances, live per-instance hourly price \(r\), and authorised hard runtime \(T_{\max}\):

\[
C_{\max}=n\,r\,T_{\max},
\]

where

- \(n\) is the authorised instance count;
- \(r\) is the live per-instance hourly price, obtained from the live Lambda API immediately before provisioning rather than from a static price table;
- \(T_{\max}\) is the hard authorised runtime.

The expected-cost companion is

\[
C_{\mathrm{expected}}=n\,r\,T_{\mathrm{expected}},
\]

where \(T_{\mathrm{expected}}\) is the expected billed runtime.

**A billable resource MUST NOT be launched when any input required to determine its authorised maximum exposure is unknown.** If \(n\), \(r\), \(T_{\max}\), or the approving budget cannot be established, the launch is invalid and the system MUST fail closed. This applies equally when price, hardware identity, availability, or SKU identity cannot be established.

### 2.2 Two independent termination mechanisms

```text
NORMAL COMPLETION PATH
job complete
    ↓
checkpoint / output / backup verification
    ↓
terminate immediately

COST FAILSAFE PATH
hard TTL reached
    ↓
terminate regardless of experiment state
```

The normal path protects scientific outputs. The hard-TTL path protects the budget when the scientific workflow crashes, stalls, loses network connectivity, or never reaches its completion sentinel. **Cost containment MUST remain independent of successful scientific completion.**

---

## 3. Lambda service selection

For most individual research projects, use **On-Demand Cloud (ODC)**. ODC provides self-service Linux GPU virtual machines, regional persistent filesystems, SSH access, JupyterLab, and one- to eight-GPU configurations. Capacity is first-come and varies by region.

Use a **1-Click Cluster (1CC)** only when the research genuinely requires multi-node distributed computation. Lambda currently documents 1CC systems using H100 or B200 GPUs with NVIDIA Quantum-2 400 Gb/s InfiniBand, GPUDirect RDMA, two 100 Gb/s Ethernet links per node, two 100 Gb/s direct Internet connections, and three CPU management nodes.

Those networking guarantees apply to 1CC. **Do not infer equivalent RDMA, placement, or inter-instance performance for ordinary ODC instances.** ODC interconnect characteristics remain an unresolved platform variable requiring measurement.

For most fine-tuning, behavioural evaluation, model-diffing, interpretability, and local-judge workloads up to roughly the 27B scale, prefer one ODC VM whenever possible. The scale figure is a planning guide inherited from the source material, not a capability boundary.

---

## 4. Authentication, access control, and network security

Establish account and network security **before** provisioning.

### 4.1 Authentication and API keys

The Lambda API base is

```text
https://cloud.lambda.ai/api/v1
```

and new integrations MUST use

```text
Authorization: Bearer <LAMBDA_API_KEY>
```

The current API specification is **OpenAPI 1.10.0** (dated observation; see section 13). Ordinary requests are generally limited to one request per second; instance-launch requests are limited to one per 12 seconds or five per minute. Automation MUST respect these limits and MUST treat rate-limit behaviour as a mutable provider fact.

Lambda API keys MUST be treated as highly privileged secrets, because they currently grant full access to all Lambda API operations. Keys MUST NOT be written into cloud-init configuration, Git repositories, or stored manifests.

### 4.2 Workspaces and access control

Lambda's access model is relatively coarse. Both account-level **Admin** and **Member** roles can create API keys, launch and terminate resources, and retrieve audit logs. Admins additionally control account membership, billing, account details, and workspace memberships. Resources in different workspaces cannot directly access one another. Lambda retains audit events for six months.

Therefore:

- use a dedicated workspace when research projects have different collaborators or data-access requirements;
- accept that account administrators retain the ability to modify workspace membership;
- enable MFA.

### 4.3 SSH policy

- use a unique SSH key per research context;
- keep private SSH keys off the instance and out of repositories;
- register the intended key before launch, and verify its presence in preflight.

Connection is by ordinary SSH to the instance's public address:

```bash
ssh -i "$SSH_PRIVATE_KEY" \
  -o IdentitiesOnly=yes \
  "ubuntu@$INSTANCE_IP"
```

### 4.4 Firewall policy

Lambda permits only incoming ICMP and TCP/22 by default. It supports both global workspace firewall rules and region-scoped per-instance firewall rulesets. Per-instance rulesets MUST be attached during launch and cannot subsequently be detached, although modifications to a ruleset propagate to attached instances.

Prefer rules such as

```text
TCP/22
source = <your-public-IP>/32
```

rather than `0.0.0.0/0`.

Development servers — Jupyter, experiment dashboards, inference servers, TensorBoard — MUST NOT be exposed directly to the public Internet. Use Lambda's console-integrated JupyterLab or an SSH tunnel.

### 4.5 Session independence

Long-running workloads MUST be detachable from the interactive session, so that SSH loss does not terminate the experiment. `tmux`, a service manager, or an equivalent mechanism satisfies this.

```bash
tmux new-session -d -s experiment \
  "bash scripts/run_experiment.sh"

tmux attach-session -t experiment
```

Conversely, an interactive session MUST NOT be the enforcement point for any cost safeguard (section 8).

---

## 5. Live discovery of provider state

### 5.1 The discovery requirement

Automation MUST NOT embed a permanent Lambda price table, image ID, region list, or SKU list. Immediately before every substantive launch, query live platform state and record it as a dated **platform snapshot**.

At minimum, preflight MUST query:

```text
GET /instance-types
GET /regions
GET /images
GET /ssh-keys
GET /file-systems
GET /instances
```

The snapshot MUST contain at least:

```text
instance-type identifier
GPU model
GPU count
VRAM per GPU
host RAM
root storage
current price per hour
regions with capacity
running-instance inventory
available images
available filesystems
registered SSH keys
API schema version
query timestamp
```

Static prices MAY be used for rough human planning only. They become stale and create false confidence in an automated spending control. **The launch-time price returned by the current provider interface is the authoritative planning input for that run.**

If live price, capacity, or SKU identity cannot be established, the system MUST fail closed and MUST NOT provision.

### 5.2 Instance types, GPU configuration, and capacity

ODC capacity is first-come. Instance types differ across regions, provisioning can take several minutes, and new accounts have an instance-count quota that increases as invoices are paid. Capacity MUST therefore be treated as a runtime variable, never as a static assumption, and capacity probability is undocumented.

The currently documented ODC regions are:

```text
asia-northeast-1   Tokyo
asia-northeast-2   Osaka
asia-south-1       India
europe-central-1   Germany
me-west-1          Israel
us-east-1          Virginia
us-east-2          Washington DC
us-midwest-1       Illinois
us-south-1         Texas
us-south-2         North Texas
us-south-3         Central Texas
us-west-1          California
us-west-2          Arizona
us-west-3          Utah
```

This list is a dated observation and MUST be re-discovered per run.

Aggregate VRAM across GPUs is not a single address space. Two 80 GB GPUs do not automatically behave like one 160 GB GPU. Inter-GPU topology MUST NOT be inferred from total VRAM or GPU marketing names; record it from the instance:

```bash
nvidia-smi topo -m
nvidia-smi nvlink -s
```

### 5.3 Hardware selection principle

Hardware MUST be selected as the **least expensive configuration that satisfies measured memory and throughput requirements**, rather than either the cheapest nominal GPU or the fastest available GPU.

Any hardware-sizing table published by a workload runbook is a set of **starting configurations, not guaranteed minimum requirements**. Admission to a substantive run MUST rest on measurement under the actual workload, not on a table entry. Workload-specific memory models belong to the specialised runbooks.

---

## 6. Provisioning invariants

### 6.1 Cost-monotone provisioning

Provisioning automation MUST be cost-monotone unless a human explicitly authorises an exception:

```text
MUST NOT automatically increase GPU class.
MUST NOT automatically increase GPU count.
MUST NOT automatically increase instance count.
MUST NOT launch duplicate copies across regions to race for capacity.
MUST NOT choose a fallback whose hourly price exceeds its approved ceiling.
```

### 6.2 Hardware fallback policy

Hardware MUST be declared explicitly in configuration, and fallbacks MUST be explicit and price-bounded:

```yaml
hardware:
  primary:
    instance_type: <SKU>
  approved_fallbacks:
    - instance_type: <SKU>
      max_price_per_hour: <USD>
    - instance_type: <SKU>
      max_price_per_hour: <USD>
  region_preferences:
    - <region>
    - <region>
```

Do not silently change GPU type because the preferred configuration is unavailable. Hardware changes can alter numerical behaviour, kernels, throughput, distributed execution, and reproducibility. Where a primary experimental comparison requires hardware equivalence, all conditions in that comparison MUST run on the same hardware class, and migration to another GPU MUST be treated as a separate block or sensitivity analysis.

### 6.3 Duplicate-launch prevention and experiment-level idempotency

Launch retries MUST be idempotent at the experiment level. A network timeout or ambiguous HTTP response does not prove that Lambda failed to create an instance.

Before retrying an ambiguous launch, the system MUST query the running-instance inventory and search for the intended `run_id`, project tag, hostname, or other unique launch identity. **A second billable resource MUST NOT be created merely because the first launch response was unclear.**

### 6.4 Declarative launch

The current launch endpoint is

```text
POST /api/v1/instance-operations/launch
```

The API supports, among other fields:

```text
region_name
instance_type_name
ssh_key_names
file_system_names
file_system_mounts
hostname
name
image
user_data
tags
firewall_rulesets
```

A research-oriented launch request is conceptually:

```json
{
  "region_name": "<REGION>",
  "instance_type_name": "<INSTANCE-TYPE>",
  "ssh_key_names": ["research-key"],
  "file_system_names": ["research-filesystem"],
  "name": "project-run-001",
  "image": { "id": "<PINNED-IMAGE-ID>" },
  "tags": [
    {"key": "project", "value": "experiment-name"},
    {"key": "run_id", "value": "run-001"},
    {"key": "owner", "value": "researcher"},
    {"key": "purpose", "value": "train"}
  ],
  "firewall_rulesets": [ {"id": "<RULESET-ID>"} ]
}
```

Image IDs MUST be retrieved from the live `/images` endpoint rather than assumed.

`user_data` permits cloud-init launch-time configuration and can make repeat deployments more reproducible. Keep the bootstrap small and auditable, and keep secrets out of it.

Some Lambda examples still contain a legacy `"quantity": 1` field while the current launch-body schema does not enumerate it. New automation MUST follow the live OpenAPI schema and MUST test the exact request in preflight rather than depending on undocumented fields.

### 6.5 Pinned image and environment at the platform boundary

Lambda currently offers:

| Image | Family | Architecture |
|---|---|---|
| Lambda Stack 24.04 | `lambda-stack-24-04` | x86-64, arm64 |
| Lambda Stack 22.04 | `lambda-stack-22-04` | x86-64, arm64 |
| Lambda Stack Legacy 22.04 | `lambda-stack-legacy-22-04` | x86-64 |
| GPU Base 24.04 | `gpu-base-24-04` | x86-64 |
| GPU Base 22.04 | `gpu-base-22-04` | x86-64 |
| Ubuntu Server 24.04 | `ubuntu-24-04` | x86-64 |
| Ubuntu Server 22.04 | `ubuntu-22-04` | x86-64 |

Lambda Stack includes NVIDIA drivers, CUDA, cuDNN, NCCL, the NVIDIA Container Toolkit, ML frameworks, Docker, JupyterLab, and common development tools. GPU Base provides a thinner GPU-ready environment. Lambda Stack Legacy is used where the older closed-source NVIDIA driver stack is required, including V100 instances.

For ordinary research work, start with `lambda-stack-24-04` unless a dependency requires otherwise. Image families and IDs are mutable provider facts.

The environment MUST be pinned as either:

```text
Pinned Lambda base image        Pinned Lambda base image
        +                 or            +
Pinned virtual environment      Container referenced by immutable digest
```

General operating-system upgrades MUST NOT be performed merely to "update everything". Lambda currently documents a known `apt full-upgrade` / `apt dist-upgrade` issue on Lambda Stack 24.04 and GPU Base 24.04 and provides a specific workaround; more importantly, uncontrolled system upgrades undermine reproducibility.

Every run MUST record at minimum the platform-boundary environment identity:

```text
Lambda image ID/family
CPU architecture
GPU model and count
NVIDIA driver
CUDA runtime
Python
PyTorch
Transformers
Accelerate
PEFT / TRL if used
bitsandbytes if used
vLLM if used
FlashAttention if used
NCCL
DeepSpeed if used
Git commit + dirty patch
model/tokenizer revision where applicable
dataset hashes where applicable
```

Workload runbooks extend this list with their own tooling and with the configuration of any distributed or sharding framework in use; they do not replace it.

After the instance becomes active, the actual environment MUST be verified rather than assumed:

```bash
nvidia-smi
python -m torch.utils.collect_env
```

```python
import platform, torch
print("arch:", platform.machine())
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("bf16:", torch.cuda.is_bf16_supported())
print("gpu:", torch.cuda.get_device_name(0))
```

An isolated experiment environment MUST be created. One low-friction approach on Lambda Stack is

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
```

which retains Lambda's CUDA-compatible installed PyTorch stack. A fully pinned environment or immutable container is preferable when reproducibility outweighs convenience.

### 6.6 CPU architecture compatibility

**GH200 hosts are ARM64.** Any Python package with native binaries, custom CUDA extensions, compiled attention kernels, inference engines, or other compiled dependencies therefore requires an arm64 compatibility check before committing to GH200. Preflight MUST fail when the image architecture mismatches the GPU host.

---

## 7. Storage

### 7.1 Ephemeral versus persistent storage

ODC root storage is **ephemeral**. When the instance is terminated, local non-filesystem data is permanently destroyed and Lambda cannot recover it.

A Lambda filesystem persists independently of the instance and normally mounts at

```text
/lambda/nfs/<FILESYSTEM-NAME>
```

Persistent storage and compute are **independent billing and lifecycle objects**.

| Storage | Use for |
|---|---|
| Ephemeral root SSD | temporary caches, compilation, intermediate scratch, disposable preprocessing, large disposable tensors |
| Lambda filesystem | datasets, run configurations, logs, manifests, checkpoints, metrics, final artifacts |
| Independent backup | important milestones and final artifacts |

### 7.2 Filesystem lifecycle

A filesystem MUST be in the same region and workspace as the instance. It MUST be attached when the instance is launched and cannot later be attached to an already running instance.

Lambda currently allows up to 24 filesystems and documents nominal capacity of 8 EB in most regions; `us-south-1` is limited to 10 TB. These counts and capacities are dated observations and MUST be re-checked live. Lambda bills filesystems according to used GiB-month in one-hour increments and continues billing while the filesystem exists, whether or not it is mounted. Lambda's documentation gives `$0.20/GiB-month` only as an illustrative example; the actual current price is shown when the filesystem is created and MUST be discovered live. Lambda does not currently charge ingress or egress fees.

The canonical storage hierarchy is:

```text
/lambda/nfs/research/
├── cache/
│   └── huggingface/
├── data/
│   ├── raw_immutable/
│   └── processed/
├── runs/
│   └── <run_id>/
│       ├── config/
│       ├── logs/
│       ├── checkpoints/
│       ├── eval/
│       └── system/
└── exports/
```

Workload runbooks MAY add subtrees under `/lambda/nfs/research/` and under `runs/<run_id>/` for workload-specific artifacts. They MUST NOT relocate the run-scoped root or reuse another run's directory.

Filesystem deletion MUST remain deliberate, because data loss is irreversible. Automated sweepers classify resources and request deletion only when retention policy and backup state permit it.

### 7.3 Independent backup expectations

A Lambda filesystem protects against instance termination. **It MUST NOT be treated as the only backup**, and its persistence MUST NOT be interpreted as an independent backup or an unstated durability guarantee. Important milestones and final artifacts MUST have an independent copy in an object store or institutional repository.

The **Filesystem S3 Adapter** exposes Lambda filesystems through S3-compatible tooling. It is an **access interface to a Lambda filesystem, not an independent object-storage backup.** As of current documentation, endpoints are available in:

```text
us-east-2      Washington DC
us-east-3      Washington DC
us-midwest-2   Ohio
```

and the adapter is designed for filesystems created in **March 2025 or later**; older filesystems might not be supported. Regional coverage is a mutable provider fact. Elsewhere, use ordinary tools such as `rsync`, `rclone`, or `s5cmd`.

### 7.4 Data staging

Expensive accelerator time SHOULD begin as close as practical to actual accelerator work. Before provisioning premium compute, confirm that code, environment or container, model weights, datasets, credentials, and preprocessing outputs are already available or can be staged without keeping the target GPU idle.

Where a remote transfer path is available, populate the persistent filesystem before launching premium compute. Where this is not possible, use the least expensive suitable instance for long transfers, decompression, tokenisation, compilation, or other CPU/IO-heavy staging work.

For repeatedly used artifacts, retain a persistent cache so each new experiment does not re-download large weights or datasets. Where high-throughput local reads matter, copy the required shard from persistent storage to local SSD at startup and remove it at teardown.

Lambda may charge no filesystem ingress or egress fee while the **source provider** can still charge data-transfer-out fees. Cross-cloud transfer MUST be treated as a separate cost item and checked against the source provider before moving large artifacts.

### 7.5 Storage hygiene

Persistent storage requires active hygiene. Applications such as JupyterLab can place deleted data into hidden `.Trash-*` directories, where it still occupies billable filesystem capacity. Every storage audit MUST include hidden trash, stale caches, abandoned checkpoints, and orphaned project directories.

### 7.6 Export and independent verification

Exports of results MUST be hash-verified before the compute resource is destroyed:

```bash
sha256sum artifact.tar.gz > artifact.tar.gz.sha256
sha256sum -c artifact.tar.gz.sha256
```

A compact export typically contains analysis outputs, metrics, raw records, logs, configuration, environment manifest, Git revision, data hashes, telemetry, and artifact metadata. Large tensors MAY be transferred separately. Any research artifact that cannot be cheaply regenerated MUST be hash-verified and independently copied.

---

## 8. Identity, authorisation, and cost containment

### 8.1 Resource identity

Every billable resource MUST carry, at launch, an identity sufficient for attribution and reconciliation:

```text
owner
project
run_id
purpose
```

These MUST be applied as provider tags (or the equivalent unique launch identity) so that inventory reconciliation and duplicate detection can operate from provider state alone. Untagged or unknown instances MUST be treated as incidents requiring immediate ownership and cost attribution, not assumed safe.

### 8.2 Launch preflight

Preflight MUST fail before launch if:

```text
requested SKU is unavailable
requested region is invalid
image architecture mismatches the GPU host
SSH key is missing
filesystem is in the wrong region/workspace
firewall ruleset is missing or too permissive
estimated cost exceeds run budget
projected account-wide hourly burn exceeds policy
projected concurrent instance/GPU count exceeds policy
a previous instance with the same run identity is still active
hard TTL or external watchdog registration failed
```

### 8.3 Launch authorisation

Before any billable launch, a machine-readable **launch authorisation record** MUST exist and MUST be internally consistent. Its schema is defined in section 11.2. The launch MUST fail if the record cannot be completed consistently.

### 8.4 Account-wide burn and concurrency limits

Per-run budgets alone do not protect against duplicate, forgotten, or concurrently launched instances. Compute the account-wide proposed burn before launch:

\[
R_{\mathrm{after}}
=
\sum_{i\in\mathrm{running}} r_i
+
n_{\mathrm{new}} r_{\mathrm{new}}.
\]

The new resource MUST NOT be authorised when \(R_{\mathrm{after}}\) exceeds the configured account or project ceiling. Concurrent-instance and concurrent-GPU ceilings MUST be evaluated in the same way.

### 8.5 External watchdog registration

Before sending the launch request, the run and its hard termination deadline MUST be registered with an **external watchdog** that is independent of the GPU VM. The watchdog MUST know:

```text
run_id
expected instance identity
launch time
soft deadline
hard deadline
maximum authorised exposure
```

**If watchdog registration fails, the launch MUST NOT proceed.**

After launch, the returned provider instance ID MUST be bound to the watchdog record. This prevents a workload-side crash, dead SSH session, corrupted environment, or lost API credential inside the VM from disabling the budget failsafe.

The watchdog MUST run on a control plane expected to remain available through the entire authorised lifetime. A sleeping laptop or an interactive SSH session MUST NOT be the only hard-TTL enforcement mechanism.

### 8.6 Soft deadline and unconditional hard TTL

Define a soft deadline \(T_{\mathrm{soft}}\) and a hard deadline \(T_{\mathrm{hard}}\) with

\[
T_{\mathrm{soft}}<T_{\mathrm{hard}}.
\]

At the soft deadline: warn, inspect progress, and attempt checkpoint or export if useful.
At the hard deadline: the independent watchdog terminates the authorised resource **regardless of experiment state**.

The hard TTL MUST NOT depend on any of:

```text
workload reporting completion
notebook or kernel alive
SSH connection
GPU utilisation
the research process writing a sentinel
```

because the purpose of the TTL is to protect against exactly those systems failing.

### 8.7 Heartbeat semantics

Every long-running job MUST emit a **progress heartbeat** to durable storage or another observer outside the process. The platform-level minimum payload is:

```json
{
  "run_id": "experiment-042",
  "timestamp": "2026-08-22T18:55:00Z",
  "status": "running",
  "progress_marker": "<workload-defined>",
  "last_durable_output": "<workload-defined>"
}
```

Workload runbooks define the semantic content of the progress and durable-output fields; they do not redefine the mechanism.

Define

\[
\Delta t_{\mathrm{heartbeat}}=t_{\mathrm{now}}-t_{\mathrm{last\ heartbeat}}.
\]

Use separate thresholds for warning and for failure investigation. Missing heartbeats can indicate a crashed process, deadlock, filesystem failure, or stalled workload. They MUST trigger diagnosis and possibly graceful cleanup; they MUST NOT replace the hard TTL.

**GPU utilisation alone is never a safe automatic kill criterion.** Legitimate phases such as preprocessing, evaluation, checkpointing, compilation, or remote synchronisation can leave the accelerator idle while useful work continues.

---

## 9. Provider-resource lifecycle

### 9.1 States

Every billable instance MUST be modelled as an explicit lifecycle:

\[
\text{PLANNED}
\rightarrow
\text{AUTHORIZED}
\rightarrow
\text{LAUNCHING}
\rightarrow
\text{RUNNING}
\rightarrow
\{\text{SUCCEEDED},\text{FAILED},\text{STALLED},\text{TTL\_EXPIRED}\}
\rightarrow
\text{TERMINATING}
\rightarrow
\text{TERMINATED}
\rightarrow
\text{AUDITED}.
\]

Operationally:

```text
PLANNED
    live price + capacity + storage + scientific configuration discovered

AUTHORIZED
    maximum cost exposure approved
    hard TTL registered externally

LAUNCHING
    provider request sent with unique run identity

RUNNING
    heartbeat + system telemetry + accrued-cost estimate

SUCCEEDED
    validate outputs and backup → terminate immediately

FAILED
    preserve diagnostics/checkpoint when feasible → terminate

STALLED
    warn / inspect / attempt graceful collection → terminate before hard TTL

TTL_EXPIRED
    terminate unconditionally

TERMINATING
    poll provider state; do not assume the request itself ended billing

TERMINATED
    prove the instance is no longer active/billable

AUDITED
    enumerate residual instances, filesystems, and other continuing costs
```

### 9.2 Success path

Automated termination on the success path MUST require an explicit sentinel:

```text
workload reports completion
AND
workload-defined durable-output validation passed
AND
artifact hashes verified
AND
independent backup verified
AND
exact instance ID matches the authorised run
```

Then call the provider termination endpoint **immediately**. Waiting until a researcher manually remembers to shut down converts completed work into idle billable time.

The content of "workload-defined durable-output validation" is owned by the specialised runbooks.

### 9.3 Failure and stall path

A failed or stalled experiment SHOULD preserve a bounded set of diagnostics and the latest safe resumable state, then terminate. Do not keep expensive compute alive indefinitely merely to permit later debugging: export the evidence required for diagnosis and reproduce on cheaper compute where possible.

### 9.4 Hard-TTL path

```text
T >= T_hard
→ terminate unconditionally
```

A termination gate that requires a completion sentinel cannot protect spending when the workload crashes before writing that sentinel. The hard-TTL path therefore overrides the normal completion safeguards of section 9.2.

### 9.5 Provider-side termination verification

Stopping Linux does **not** terminate the Lambda billing resource. Do not rely on:

```bash
sudo shutdown
poweroff
```

to end charges. Use the Lambda console or provider termination API, then verify:

```text
1. poll the Lambda instance inventory;
2. confirm the target instance is terminating or absent;
3. continue until it is no longer an active billable instance;
4. record the observed termination timestamp/state;
5. only then mark compute cleanup complete.
```

**Successful API submission MUST NOT be equated with proven termination.** The observation MUST be recorded in a termination receipt (section 11.4).

---

## 10. Post-run reconciliation and audit

### 10.1 Residual-resource sweep

Every completed, failed, or TTL-expired run MUST end with an independent resource sweep:

```text
 1. List all running instances.
 2. Compare them with the authorised resource registry.
 3. Flag untagged, unknown, duplicated, or expired instances.
 4. List persistent filesystems.
 5. Identify project filesystems with no active owner/run.
 6. Measure filesystem usage and large directories.
 7. Inspect hidden .Trash-* directories.
 8. Inspect stale caches and obsolete checkpoints.
 9. Identify failed-run scratch accidentally written to persistent storage.
10. Estimate remaining compute and storage exposure.
11. Record final accrued compute cost.
12. Emit an explicit final status.
```

Useful primitives:

```bash
df -h
du -sh /lambda/nfs/<filesystem>/*
```

The preferred terminal report is:

```text
Remaining unexpected billable compute resources: NONE
Remaining expected persistent resources: <explicit list>
Estimated continuing storage exposure: <value or UNKNOWN>
```

**If an unexpected resource remains, cleanup is incomplete even if the experiment succeeded.** The terminal state MUST state explicitly whether any unexpected billable resource remains.

### 10.2 Independent accrued-cost accounting and reconciliation

Maintain an independent accrued-cost ledger during every run. The displayed Lambda service-credit balance MUST NOT be treated as a real-time spending limit, because displayed credits and recently accrued usage can be updated on different billing cadences.

Track at minimum:

```text
run_id
provider instance ID
launch timestamp
live launch price/hour
elapsed billed time estimate
accrued compute estimate
persistent-storage estimate where measurable
external API spend
data-transfer-out charges from source providers
maximum authorised exposure
```

For running compute, a simple real-time estimate is

\[
\widehat C_{\mathrm{compute}}(t)
=
\sum_i r_i\,[t-t_{i,\mathrm{start}}],
\]

with provider billing granularity applied when producing the final reconciliation.

The project cost model is

\[
C_{\text{total}}
=
C_{\text{compute}}
+
C_{\text{storage}}
+
C_{\text{external APIs}}
+
C_{\text{failed runs}}
+
C_{\text{other}},
\qquad
C_i=r_i h_i,
\]

where \(r_i\) is the current node cost per hour for phase \(i\) and \(h_i\) is the measured billed duration. Runtime MUST NOT be estimated from parameter count alone; it MUST be measured, and the measured throughput MUST be obtained under the actual workload configuration.

The major controllable cost factors are hardware choice, actual utilisation, workload volume, context length, repetition count, adaptation strategy, evaluation frequency, checkpoint overhead, idle setup time, failed-run probability, and storage retention.

Final reconciliation compares actual accrued spend against `launch_authorisation.yaml` and records the difference.

---

## 11. Canonical machine-readable contracts

These five records are the implementation boundary of this specification. They are **research-control-plane schemas**. They are not claims that Lambda exposes these objects, and they MUST NOT be extended with invented provider API fields.

Every contract MUST be durable, append-only in effect (superseding versions rather than silently overwriting), and stored outside the ephemeral instance.

```text
platform_snapshot.yaml
launch_authorisation.yaml
resource_registry.yaml
termination_receipt.yaml
post_run_audit.yaml
```

### 11.1 `platform_snapshot.yaml`

**Purpose.** Freeze the mutable provider facts observed for one launch decision, so that every later cost and reproducibility claim can be traced to what was actually true at query time.

**Created/updated.** Created during `preflight`, before authorisation. Never edited afterwards; a new preflight creates a new snapshot.

**Required fields.**

```yaml
snapshot_id: ...
query_timestamp: ...
api_schema_version: ...

instance_types:
  - name: ...
    gpu_model: ...
    gpu_count: ...
    vram_gb_per_gpu: ...
    host_ram_gb: ...
    root_storage: ...
    price_per_hour: ...
    regions_with_capacity: [...]

regions: [...]
images:
  - id: ...
    family: ...
    architecture: ...
filesystems:
  - name: ...
    region: ...
    workspace: ...
ssh_keys: [...]
running_instances:
  - instance_id: ...
    instance_type: ...
    price_per_hour: ...
    tags: {...}
account_hourly_burn_observed: ...
```

**Invariants.**

- MUST be produced from live queries, never from a stored price table.
- MUST carry a query timestamp; a snapshot older than the launch decision MUST NOT be reused.
- `price_per_hour` in this snapshot is the only admissible source of \(r\) for the accompanying authorisation.
- MUST include the running-instance inventory, because both duplicate detection and \(R_{\mathrm{after}}\) depend on it.

**Lifecycle relation.** Enables `PLANNED`. Absence of a valid snapshot makes `AUTHORIZED` unreachable.

### 11.2 `launch_authorisation.yaml`

**Purpose.** Record that a specific bounded exposure was approved for a specific resource before any billable action.

**Created/updated.** Created during `authorize`, after preflight passes and before `launch`. Immutable thereafter; a changed envelope requires a new authorisation.

**Required fields.**

```yaml
launch_authorisation:
  owner: <researcher>
  project: <project>
  run_id: <unique-id>
  purpose: <train|eval|debug|transfer|...>

  platform_snapshot_id: <snapshot>

  instance_type: <live SKU>
  instance_count: <n>
  region: <region>
  image_id: <pinned>
  current_price_per_instance_hour: <queried>

  approved_fallbacks:
    - instance_type: <SKU>
      max_price_per_hour: <USD>

  expected_runtime_hours: <T_expected>
  soft_deadline_hours: <T_soft>
  hard_ttl_hours: <T_max>

  expected_compute_cost_usd: <C_expected>
  maximum_compute_cost_usd: <C_max>

  max_account_hourly_burn_usd: <policy>
  max_concurrent_instances: <policy>
  max_concurrent_gpus: <policy>

  watchdog_registration_id: <external>
```

**Invariants.**

- MUST be internally consistent: \(C_{\max}=n r T_{\max}\) and \(C_{\mathrm{expected}}=n r T_{\mathrm{expected}}\), with \(r\) taken from the referenced snapshot.
- \(T_{\mathrm{soft}}<T_{\mathrm{hard}}\).
- MUST NOT be created when any of \(n\), \(r\), \(T_{\max}\), or the approving budget is unknown.
- MUST NOT be created when \(R_{\mathrm{after}}\) exceeds policy.
- MUST reference a successful external watchdog registration.
- A fallback MUST NOT be exercised above its `max_price_per_hour`.

**Lifecycle relation.** Its existence defines `AUTHORIZED`. It is the reference against which the final cost reconciliation is computed.

### 11.3 `resource_registry.yaml`

**Purpose.** Bind authorised runs to actual provider resources, so that inventory can be reconciled and duplicates detected from provider state.

**Created/updated.** Created at `launch` when the provider request is sent; updated when the provider instance ID returns, on every lifecycle transition, and at each accrued-cost update.

**Required fields.**

```yaml
resources:
  - run_id: ...
    project: ...
    owner: ...
    purpose: ...
    authorisation_ref: ...

    provider_instance_id: ...
    instance_type: ...
    region: ...
    launch_request_timestamp: ...
    launch_observed_timestamp: ...
    price_per_hour_at_launch: ...

    lifecycle_state: PLANNED|AUTHORIZED|LAUNCHING|RUNNING|
                     SUCCEEDED|FAILED|STALLED|TTL_EXPIRED|
                     TERMINATING|TERMINATED|AUDITED
    soft_deadline_at: ...
    hard_deadline_at: ...
    watchdog_registration_id: ...

    last_heartbeat_at: ...
    accrued_cost_estimate_usd: ...

    attached_filesystems: [...]
```

**Invariants.**

- Exactly one active registry entry per `run_id`. A second live resource with the same `run_id` is a duplicate incident.
- An entry MUST exist before a launch request is sent, so an ambiguous response can be reconciled against intent.
- `provider_instance_id` MUST be bound to the watchdog record as soon as it is known.
- Every running provider instance MUST reconcile to exactly one registry entry; unmatched instances are incidents.

**Lifecycle relation.** It is the authoritative map from `LAUNCHING` through `AUDITED`, and the reconciliation target of the post-run sweep.

### 11.4 `termination_receipt.yaml`

**Purpose.** Provide evidence that billing for a resource actually stopped, distinct from evidence that a termination request was accepted.

**Created/updated.** Created when `terminate` is invoked; completed only after provider-side polling confirms the resource is no longer active.

**Required fields.**

```yaml
termination_receipt:
  run_id: ...
  provider_instance_id: ...
  termination_path: success|failure|stall|ttl_expired
  requested_at: ...
  request_accepted: true|false

  completion_gate:
    workload_completion_reported: true|false
    durable_output_validation_passed: true|false
    artifact_hashes_verified: true|false
    independent_backup_verified: true|false
    instance_id_matches_authorisation: true|false

  provider_polling:
    - polled_at: ...
      observed_state: ...
  observed_terminated_at: ...
  verified_not_billable: true|false

  final_elapsed_billed_hours_estimate: ...
```

**Invariants.**

- `verified_not_billable: true` MUST be set only from observed provider inventory state, never from request acceptance, SSH loss, Linux shutdown, or low utilisation.
- On `termination_path: success`, every `completion_gate` field MUST be true.
- On `termination_path: ttl_expired`, the receipt MUST be produced regardless of the completion gate.
- A run MUST NOT be marked cleaned up without a receipt whose `verified_not_billable` is true.

**Lifecycle relation.** Records the transition `TERMINATING → TERMINATED`.

### 11.5 `post_run_audit.yaml`

**Purpose.** Prove that no unexpected billable resource survives the run, and reconcile actual spend against authorised exposure.

**Created/updated.** Created after every completed, failed, or TTL-expired run, following the termination receipt.

**Required fields.**

```yaml
post_run_audit:
  run_id: ...
  audited_at: ...

  instance_inventory:
    - provider_instance_id: ...
      reconciled_to_run: <run_id|NONE>
      classification: authorised|duplicate|expired|untagged|unknown

  filesystem_inventory:
    - name: ...
      region: ...
      used_gib: ...
      owning_run: <run_id|NONE>
      large_directories: [...]
      hidden_trash_found: true|false
      stale_caches_found: true|false
      obsolete_checkpoints_found: true|false
      misplaced_scratch_found: true|false

  residual_billable_compute: NONE|<explicit list>
  persistent_resources_intentionally_retained: <explicit list>
  disposable_data_remaining: NONE|<explicit list>
  estimated_continuing_storage_exposure: <value|UNKNOWN>

  cost_reconciliation:
    maximum_authorised_cost_usd: ...
    expected_cost_usd: ...
    actual_accrued_compute_usd: ...
    variance_usd: ...
    variance_explanation: ...
```

**Invariants.**

- MUST state `residual_billable_compute` explicitly; silence is not a pass.
- Any `classification` other than `authorised` is an incident requiring ownership and cost attribution before the audit closes.
- Deletion of persistent data MUST NOT be automated without retention policy and verified backup state.
- Cost reconciliation MUST use the independent ledger, not displayed credits.

**Lifecycle relation.** Its existence defines `AUDITED`, the terminal state.

---

## 12. Canonical control surface

Any implementation of this specification MUST expose the following seven operations with the semantics below. This is a **behavioural contract**, not a CLI syntax.

```text
preflight → authorize → launch → status → watch → terminate → audit
```

### 12.1 `preflight`

- **Inputs/prerequisites.** Intended workload identity (`owner`, `project`, `run_id`, `purpose`); intended hardware, region, image, filesystem, SSH key, firewall ruleset; intended \(T_{\mathrm{expected}}\), \(T_{\mathrm{soft}}\), \(T_{\max}\); account and project policy ceilings; valid API credentials.
- **State transition.** `— → PLANNED`.
- **Required checks.** Live discovery per section 5.1; every failure condition in section 8.2; architecture compatibility (6.6); filesystem region/workspace correctness (7.2); duplicate run identity in the live inventory (6.3); computation of \(C_{\mathrm{expected}}\), \(C_{\max}\), and \(R_{\mathrm{after}}\).
- **Durable output.** `platform_snapshot.yaml`, plus a preflight result listing every check and its outcome.
- **Failure behaviour.** Fail closed. No billable action. Report the specific failing check. Never substitute a more expensive SKU to make a check pass.

### 12.2 `authorize`

- **Inputs/prerequisites.** A passing preflight and its snapshot; a human-approved or policy-approved budget.
- **State transition.** `PLANNED → AUTHORIZED`.
- **Required checks.** Internal consistency of the exposure relation; \(T_{\mathrm{soft}}<T_{\mathrm{hard}}\); \(R_{\mathrm{after}}\) within ceilings; concurrency ceilings; successful external watchdog registration with the hard deadline and maximum exposure.
- **Durable output.** `launch_authorisation.yaml`; a watchdog registration ID; an initial `resource_registry.yaml` entry in state `AUTHORIZED`.
- **Failure behaviour.** If any input to \(C_{\max}\) is unknown, or watchdog registration fails, refuse. No launch may proceed on an incomplete authorisation.

### 12.3 `launch`

- **Inputs/prerequisites.** A valid, unexpired authorisation whose snapshot still matches live capacity.
- **State transition.** `AUTHORIZED → LAUNCHING → RUNNING`.
- **Required checks.** Exact SKU, count, region, image ID, filesystem, SSH key, firewall ruleset, and identity tags match the authorisation; no active resource already carries this run identity; request conforms to the live API schema.
- **Durable output.** Updated `resource_registry.yaml` containing the provider instance ID, launch timestamps, and price at launch; the provider instance ID bound to the watchdog record.
- **Failure behaviour.** On an ambiguous or timed-out response, do **not** retry blindly: query the inventory for the run identity first. If a resource was created, adopt it into the registry. If capacity is unavailable, either stop or use an approved fallback within its price ceiling; never escalate class, count, or price silently.

### 12.4 `status`

- **Inputs/prerequisites.** `run_id` or provider instance ID; registry access.
- **State transition.** None; observation only.
- **Required checks.** Provider-reported instance state; \(\Delta t_{\mathrm{heartbeat}}\); telemetry availability; accrued-cost estimate against \(C_{\max}\); remaining time to \(T_{\mathrm{soft}}\) and \(T_{\mathrm{hard}}\).
- **Durable output.** Appended registry observations: last heartbeat, lifecycle state, accrued-cost estimate.
- **Failure behaviour.** If provider state cannot be read, report `UNKNOWN` rather than assuming the resource is gone; an unreadable resource is presumed billable.

### 12.5 `watch`

- **Inputs/prerequisites.** An authorised, registered run; a control plane expected to remain available for the full authorised lifetime.
- **State transition.** `RUNNING → {SUCCEEDED, FAILED, STALLED, TTL_EXPIRED}`.
- **Required checks.** Continuous evaluation of the hard deadline independent of workload state; soft-deadline warning and inspection; heartbeat staleness classification; account-wide burn monitoring; detection of resources absent from the registry.
- **Durable output.** Registry state transitions with timestamps and the triggering evidence.
- **Failure behaviour.** The watchdog MUST fail safe toward termination: if it cannot determine that a resource is within its authorised envelope, it escalates to `terminate` at \(T_{\mathrm{hard}}\) unconditionally. Watchdog unavailability MUST be treated as an incident, not as permission to continue. GPU utilisation alone MUST NOT trigger termination.

### 12.6 `terminate`

- **Inputs/prerequisites.** Provider instance ID from the registry; termination path (`success`, `failure`, `stall`, `ttl_expired`).
- **State transition.** `{SUCCEEDED, FAILED, STALLED, TTL_EXPIRED} → TERMINATING → TERMINATED`.
- **Required checks.** On the success path, the complete completion gate of section 9.2. On the failure or stall path, bounded diagnostic export and preservation of resumable state where valuable. On the TTL path, no checks: terminate unconditionally. In all paths, verify the instance ID matches the authorised run before issuing the request, then poll the provider until the resource is no longer active.
- **Durable output.** `termination_receipt.yaml`.
- **Failure behaviour.** If the termination request fails or provider state cannot be confirmed, retry and escalate; the run remains `TERMINATING` and MUST NOT be reported as cleaned up. Linux shutdown is never an acceptable substitute.

### 12.7 `audit`

- **Inputs/prerequisites.** A termination receipt; registry; live provider inventory; the independent cost ledger.
- **State transition.** `TERMINATED → AUDITED`.
- **Required checks.** The full sweep of section 10.1; reconciliation of every running instance against the registry; filesystem inventory including hidden trash and stale caches; residual billable-resource detection; cost reconciliation against the authorisation.
- **Durable output.** `post_run_audit.yaml` and the explicit terminal report.
- **Failure behaviour.** Unmatched, untagged, duplicate, or expired resources leave the audit **open** and are incidents. The audit MUST NOT close by omission, and MUST NOT delete persistent data whose retention policy or backup state is unverified.

---

## 13. Platform facts that must remain dynamic

The following MUST NOT be frozen into this specification or into any automation derived from it:

```text
GPU prices
GPU availability
instance-type identifiers
regional inventory
account quota
image IDs
filesystem price
S3 Adapter regional coverage
API schema/version
external model/API prices
third-party rate limits
displayed credit/billing refresh behaviour
provider-side budget or spending-control features
```

Every research launch MUST instead create a dated `platform_snapshot.yaml` containing the live API responses or equivalent machine-readable records.

At the time of this revision, Lambda's Cloud API advertises OpenAPI **1.10.0**, and the ODC price table, regions, image families, filesystems, S3 Adapter coverage, firewall model, and billing behaviour were rechecked on **22 August 2026**. These are observations of that date.

The most important remaining platform uncertainties are:

- ordinary ODC does not provide sufficiently detailed contractual per-SKU interconnect benchmarks for assuming distributed performance;
- filesystem price must be checked live;
- capacity probabilities are undocumented;
- Lambda filesystem persistence must not be interpreted as an independent backup or an unstated durability guarantee.

---

## 14. Platform failure modes

| Failure | First diagnostic | Typical remedy |
|---|---|---|
| Instance unavailable | Live `/instance-types`, quota, region | Approved fallback within its ceiling, or another region |
| SSH fails | Instance status, IP, key, firewall | Correct key/CIDR; wait until active |
| SSH loss kills experiment | Process ownership | `tmux` or service manager |
| CUDA import mismatch | `nvidia-smi`, `torch.utils.collect_env` | Restore pinned environment |
| Filesystem full | `df`, `du`, hidden trash | Remove verified disposable data |
| Unexpected bill | Instance/filesystem inventory | Terminate the exact resource |
| Job crashed but VM still runs | Heartbeat/process state + provider inventory | Export bounded diagnostics, then terminate |
| Duplicate instance after launch retry | `run_id`/tags + provider inventory | Keep one authorised instance; terminate the duplicate |
| Fallback unexpectedly expensive | Live SKU price + authorisation record | Reject fallback above approved ceiling |
| Hidden storage growth | `.Trash-*`, caches, stale checkpoints | Delete only verified disposable data |
| Displayed credits look sufficient but usage is stale | Independent accrued-cost ledger | Trust the independent ledger and provider bill |

Workload-specific failure modes belong to the specialised runbooks.

---

## 15. Minimal platform workflow

```text
 1. Freeze scientific configuration (workload runbook)
          ↓
 2. Estimate resource requirements (workload runbook)
          ↓
 3. Stage code, data, model weights, and environment artifacts
          ↓
 4. preflight: query live capacity, price, regions, images, storage, running instances
          ↓
 5. Compute run-level and account-wide cost envelope
          ↓
 6. authorize: exact SKU / quantity / fallbacks / maximum exposure
          ↓
 7. Register external soft deadline + hard TTL watchdog
          ↓
 8. Configure region, workspace, SSH, firewall, filesystem, and pinned image
          ↓
 9. Validate launch request and duplicate-run check
          ↓
10. launch: exact authorised resource with project/run/owner/purpose tags
          ↓
11. Bind returned provider instance ID to watchdog and cost ledger
          ↓
12. Record hardware/software environment
          ↓
13. Bootstrap pinned experiment environment
          ↓
14. Workload validation stages (workload runbook)
          ↓
15. Execute workload with heartbeat + telemetry + accrued-cost monitoring
          ↓
16. Workload-defined durable-output validation
          ↓
17. Export + hash + independent backup
          ↓
18A. Success → terminate immediately
   OR
18B. Failure/stall → bounded diagnostic export → terminate
   OR
18C. Hard TTL → unconditional external termination
          ↓
19. Poll provider API until termination is verified
          ↓
20. audit: sweep all remaining instances and persistent storage
          ↓
21. Reconcile actual versus expected spend
          ↓
22. Record residual billable resources explicitly
```

A job may fail scientifically without being allowed to fail economically: the hard TTL and the resource sweeper remain active even when the workload never reaches its normal completion logic.

---

## 16. Canonical Lambda documentation

For implementation, treat the live Lambda documentation and API as authoritative for platform-specific behaviour:

- On-Demand Cloud overview
- Cloud API and current OpenAPI specification
- Creating and managing instances
- Access and security
- Firewalls
- Filesystems
- Filesystem S3 Adapter
- Importing and exporting data
- Billing
- Billing management / credits
- Current GPU pricing
- 1-Click Clusters

---

## 17. Core platform invariants

> **Freeze the experiment; discover the infrastructure live.**
>
> **Authorise cost before launch; enforce termination independently of the workload.**

Scientific configuration is predetermined and versioned. Lambda's prices, inventory, API responses, and provisioned hardware are queried, recorded, validated, and treated as observed execution metadata. Every billable resource has an explicit owner, purpose, maximum exposure, external hard TTL, verified teardown record, and closed audit.
