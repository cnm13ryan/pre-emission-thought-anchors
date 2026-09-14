.PHONY: test lint doctor
test:
	CUDA_VISIBLE_DEVICES="" uv run --frozen pytest -q tests/unit tests/integration
lint:
	uv run --frozen ruff check .
doctor:
	uv run --frozen preanchor doctor --config configs/mvp.yaml
