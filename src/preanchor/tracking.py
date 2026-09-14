"""W&B stage sessions with offline fallback and no raw activation uploads."""

import os
from pathlib import Path

from preanchor.logging import redact


def start_tracking(directory: Path, manifest: dict, config, *, stage="stage1"):
    import wandb

    mode = os.environ.get("WANDB_MODE", config.tracking.wandb_mode)
    if mode != "online" or not os.environ.get("WANDB_API_KEY"):
        mode = "offline"
    kwargs = {"project": config.tracking.wandb_project, "group": manifest["run_id"],
              "job_type": stage, "dir": str(directory / "logs"),
              "config": redact({"run_id": manifest["run_id"], "identity": manifest["identity"],
                                "resolved_config": config.model_dump(mode="json")}),
              # Stage 1 is CPU-only; defer automatic hardware probing to the GPU admission gate.
              "settings": wandb.Settings(init_timeout=20, quiet=True, x_disable_stats=True,
                                         x_disable_meta=True, save_code=False, disable_git=True)}
    try:
        session = wandb.init(mode=mode, **kwargs)
    except Exception:
        mode = "offline"
        os.environ["WANDB_MODE"] = mode
        try:
            session = wandb.init(mode=mode, **kwargs)
        except Exception:
            # Durable local telemetry remains available even if the local W&B service fails.
            return None, {"mode": "local_json_fallback", "url": None,
                          "reason": "wandb_initialization_failed"}
    if mode == "offline":
        os.environ["WANDB_MODE"] = mode
    return session, {"mode": mode, "id": session.id,
                     "url": session.url if mode == "online" else None,
                     "directory": str(Path(session.dir).parent)}
