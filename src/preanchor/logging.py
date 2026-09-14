"""Structured events with recursively redacted credential fields and values."""

import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

_SENSITIVE = re.compile(r"token$|api_key|password|secret|credential|authorization|ssh_key", re.I)
_ENV_SECRETS = ("HF_TOKEN", "WANDB_API_KEY", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if _SENSITIVE.search(key) else redact(item)
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        for name in _ENV_SECRETS:
            secret = os.environ.get(name)
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"time": datetime.now(UTC).isoformat(), "level": record.levelname,
                   "logger": record.name, "event": record.getMessage(),
                   **getattr(record, "fields", {})}
        return json.dumps(redact(payload), sort_keys=True, allow_nan=False)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("preanchor")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
