# GPU environment readiness

Validation date: 2026-09-14. This is environment readiness evidence, not Stage 6 fit
admission and not an experimental result.

The current Codex execution environment can access the host GPU. `nvidia-smi` reports
one NVIDIA A100-SXM4-40GB, driver 580.105.08, 40,960 MiB, with no running processes.
The locked GPU extra is installed without changing `uv.lock`: torch `2.7.0+cu126`,
Transformers 4.53.2, Accelerate 1.8.1 and Sentence Transformers 5.0.0. A BF16 CUDA
matrix kernel succeeded. The pinned model revision resolves successfully.

Production doctor was run with `CUDA_VISIBLE_DEVICES=0`, persistent `PREANCHOR_ROOT`,
`HF_HOME`, and `WANDB_DIR`, with W&B offline. All checks passed, including the writable
virtiofs persistent mount, exact submodules, model revision and offline tracking.
The doctor explicitly leaves the Stage 6 measured extraction fit test pending. No
generation, activation extraction or other Stage 5–7 workload was started. A100 use
remains limited to the SPEC concurrency-one profile after that fit test.

The prior sandbox failure is recorded as an environment/tooling constraint, not a host
driver failure. No driver, CUDA toolkit or infrastructure change was made.

Doctor JSON is retained at `/tmp/preanchor-doctor.json`; the durable readiness report
must be generated under the next committed run if GPU work is later admitted.
