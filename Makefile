PY=python3

.PHONY: install lint format docs extract filter rpv pre-commit

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt
	pre-commit install || true

lint:
	pre-commit run --all-files || true

format:
	black src
	isort src

extract:
	$(PY) src/01_main_extract_data.py --config src/config_file_example.yml

filter:
	$(PY) src/02_filtering.py --config src/config_file_example.yml

rpv:
	$(PY) src/03_RPV_modelling.py --config src/config_file_example.yml --band band1

