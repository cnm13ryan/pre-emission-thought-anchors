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
