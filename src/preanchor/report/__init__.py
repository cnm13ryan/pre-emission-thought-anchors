"""SPEC Stage 11: report, interface export and dashboard from saved Stage 3-10 artifacts.

Every consumer in this package reads the restored root strictly read-only: each input is
sidecar-verified before use and its SHA-256 recorded, no validator is invoked in a mode
that writes a record, and no model weights or activation shards are ever opened. Outputs
go only to a fresh Stage 11 output directory.
"""
