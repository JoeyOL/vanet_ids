PYTHON ?= ./.venv/bin/python

.PHONY: install test preprocess train test-run

install:
	python3 -m venv .venv && $(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

preprocess:
	$(PYTHON) main.py --mode preprocess --config configs/default.toml

train:
	$(PYTHON) main.py --mode train --config configs/default.toml

test-run:
	$(PYTHON) main.py --mode test --config configs/default.toml
