PYTHON ?= python3

.PHONY: install lint test test-unit test-e2e build check extract filter rpv pipeline clean

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

lint:
	$(PYTHON) -m ruff check src tests scripts --select E9,F63,F7,F82

test:
	$(PYTHON) -m pytest tests/ -q

test-unit:
	$(PYTHON) -m pytest tests/ -q -k 'not e2e'

test-e2e:
	$(PYTHON) -m pytest tests/e2e/ -q

build:
	$(PYTHON) -m build

check:
	$(PYTHON) -m twine check dist/*

extract:
	$(PYTHON) -m oncerco_uav.pipelines.extract_data --config examples/config_file_example.yml

filter:
	$(PYTHON) -m oncerco_uav.pipelines.filtering --config examples/config_file_example.yml

rpv:
	$(PYTHON) -m oncerco_uav.pipelines.modelling --config examples/config_file_example.yml --band band1

pipeline: extract filter rpv

clean:
	rm -rf build dist *.egg-info .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
