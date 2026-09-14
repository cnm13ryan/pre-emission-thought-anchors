# Decisions

Record every result-affecting fact, assumption, choice and interpretation with commit and rejected alternative.

Entries below are based on repository commit `85a5604`; their implementation commit is
the Stage 0 bootstrap commit containing this file (`git log --follow docs/decisions.md`).

| Tag | Decision/evidence | Rejected alternative |
| --- | --- | --- |
| fact | User directs code checkout to `/home/ubuntu/preanchor-repo`; persistent state remains `/lambda/nfs/preanchor/preanchor`. | Treating the ephemeral checkout as authoritative experiment storage. |
| fact | `nvidia-smi` fails inside Codex sandbox; user confirms it works in normal host SSH. | Diagnosing a host driver failure from the sandbox result. |
| choice | Stage 0 CPU validation uses Python 3.11, with the prescribed GPU packages in the `gpu` extra and cu126 torch source. All dependencies are resolved in one `uv.lock`. | Installing GPU packages or admitting GPU stages during CPU bootstrap. |
| choice | Smoke and pilot configs inherit all scientific choices from MVP, changing only execution scope. Splits resolve into the hashed configuration. | Copying divergent science settings into separate configs or hashing only a split filename. |
| interpretation | YAML keys give explicit names to the SPEC §3.9 observable fields; no field is added to its allow-list. | Adding target-token probabilities or completed-length fields to primary inputs. |
| choice | `doctor --cpu-only` reports unchecked production prerequisites and cannot grant GPU admission. | Treating passing skeleton tests as hardware admission. |
| fact | Both pinned upstream repositories carry MIT notices; no source routine has been reused at Stage 0. | Assuming licence availability without reading the pinned notices. |
| choice | User chose MacBook-initiated SSH pulls into `~/preanchor-backups/` for the second durable copy. Stage 0 bundles only committed source, notices and validation documentation. | Pushing to the MacBook, uploading model caches/secrets, or claiming verification before the local receipt. |

No research results, model selection or scientific amendments have occurred.

Follow-up to bootstrap commit `2852b60` (`git log --follow src/preanchor/storage.py`):
**fact:** the designated persistent filesystem is exposed through writable `virtiofs`
under escalation. **choice:** fix the overly narrow NFS-type guard while retaining
dedicated-mount/path/write/checksum requirements. Rejected alternatives: treating the
root disk as durable, bypassing mount validation, or misreporting the virtual mount
as native NFS. The rejected attempt wrote no snapshot; science and config are unchanged.

Stage 1 (base commit `ad5758b`; implementation commit is the commit adding `manifest.py`):

- **fact:** the user supplied a MacBook verification receipt for all six Stage 0 files,
  manifest SHA-256 `dcd5417acca481bce61e1e50fbf79d2305f3f7f12b879ffb8877a03615f9c22c`,
  verified at `2026-09-14T17:37:27.331293+00:00`. Rechecked all persistent files and saved
  the receipt beside the snapshot; alternative rejected: inferring secondary verification
  from the existence of the first copy.
- **choice:** work-unit completion requires an immutable checksum manifest; incomplete or
  corrupt directories are renamed to quarantine, never deleted. OS advisory locks prevent
  concurrent writers and are released after abrupt termination. Alternative rejected:
  trusting file existence or rewriting valid completed samples.
- **choice:** run manifests have immutable checksummed revisions and an atomic current
  view; recovery validates that view against its revision if its sidecar publication was
  interrupted. Live heartbeat is replaceable metadata. Alternative rejected: accepting
  an unchecked current view or losing provenance through in-place edits.
- **choice:** run resume requires identical resolved config, code/diff, lock and input
  hashes. Artifact cache identities exclude tracking presentation but include all
  scientific inputs. Alternative rejected: silently mixing outputs from different runs.
- **choice:** Stage 1 smoke uses three tiny synthetic CPU units and tests both SIGTERM
  and SIGKILL. W&B offline files stay inside the run for backup; automatic hardware/code
  collection is disabled until the user's GPU-access gate is resolved. Explicit CPU RAM,
  rate and elapsed-time fields remain available. Alternative rejected: model inference
  or implicit GPU admission during I/O validation.
- **interpretation:** implementation-validation runs are not research experiments or
  evidence about probe performance. No Stage 1 artifact is a test metric or anchor label.
- **fact (after `b002449`):** the persistent smoke preserved completed hashes/mtimes
  through SIGTERM and SIGKILL, quarantined two partial units and completed a no-op resume.
  Full evidence is in `docs/stage1_validation.md`. The status reader initially called an
  old completed heartbeat stale; it now limits that flag to running work, with a regression
  test. Alternative rejected: treating normal completion as a stalled-job diagnosis.
