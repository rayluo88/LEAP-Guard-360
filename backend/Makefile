VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

.PHONY: venv install test clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest

clean:
	rm -rf $(VENV) .pytest_cache __pycache__ */__pycache__
