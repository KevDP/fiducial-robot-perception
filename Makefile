.PHONY: install install-dev test lint format clean dataset baseline

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

dataset:
	python -m fiducial.generate --out data/phase0 --seed 0

baseline:
	python -m fiducial.baseline --dataset data/phase0 --out experiments/baseline_aruco.json

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
