import json
import logging

from preanchor.logging import JsonFormatter


def test_structured_logs_redact_secrets_in_fields_and_messages(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "synthetic-secret-for-test")
    record = logging.LogRecord("preanchor", logging.INFO, "test", 1,
                               "failure synthetic-secret-for-test", (), None)
    record.fields = {"nested": {"WANDB_API_KEY": "test-key"}, "completed": 3}
    result = JsonFormatter().format(record)
    assert "synthetic-secret-for-test" not in result
    assert "test-key" not in result
    parsed = json.loads(result)
    assert parsed["completed"] == 3
    assert parsed["level"] == "INFO"
