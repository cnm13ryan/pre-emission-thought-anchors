"""Portable, content-addressed backup and fail-closed restore of canonical state.

Scientific identity (content, configuration, run and rollout IDs) is preserved as exact
bytes; physical identity (region, filesystem, host, GPU, mount path) is provenance only.
"""
