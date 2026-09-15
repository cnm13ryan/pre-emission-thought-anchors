"""Stage 9 calibration policy helper (SPEC §3.10 items 4-5), kept testable from Stage 8.

The calibrator itself is fitted in Stage 9 on validation rows only; Stage 8 never reads
validation rows. Only the explicit skip rule lives here.
"""

from __future__ import annotations

from preanchor.config import ExperimentConfig


def calibration_decision(n_positive: int, n_classes: int, config: ExperimentConfig) -> dict:
    minimum = int(config.probes["calibration_minimum_positives"])
    classes = int(config.probes["calibration_minimum_classes"])
    if n_classes < classes:
        return {"calibrate": False, "reason": f"validation has {n_classes} class(es)"}
    if n_positive < minimum:
        return {"calibrate": False,
                "reason": f"validation has {n_positive} positives, fewer than {minimum}"}
    return {"calibrate": True, "reason": None}
