# Upstream audit

Record upstream reuse, paper/released/legacy/canonical differences and the primary-label gate.

Stage 0 inspection (implementation based on commit `85a5604`):

| Repository | Required commit | Licence |
| --- | --- | --- |
| thought-anchors | b53ed8c75d3f6112f68adfaec9a13d4d708c442e | MIT; `LICENSE.md`, copyright 2025 interp-reasoning |
| thought-anchors.com | 67071a30bd2173c56f1772a26f6a9e6855170035 | MIT; `LICENSE.md`, copyright 2025 interp-reasoning |

Both are unchanged submodules. No upstream routines have been copied or wrapped yet.
Retain licence notices with future ports. Label parity and primary-field selection have
not been audited and are not established by the bootstrap tests.
