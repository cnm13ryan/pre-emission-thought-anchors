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

Stage 2 source inspection (execution commit `700d348`):
`vendor/thought-anchors/selected_problems.json` exists at the pinned commit. It has
106 records, and its first twenty problem IDs exactly equal the frozen MVP set.
SHA-256: `9a89831a1bda3f880e560c4ee75908e6972305d7bce1717a377122854646481e`.
This corrects the earlier continuation record's failed search; it changes no selection.
The file remains unmodified and is included through the pinned submodule Git bundle.
Stage 2 independently implements acquisition; no upstream scientific routine is copied.
