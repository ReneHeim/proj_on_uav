PY=python3
PIP=pip3

.PHONY: install lint format docs extract filter rpv test coverage clean build deploy help

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies and pre-commit hooks
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .
	pre-commit install || true

lint:  ## Run linting checks
	pre-commit run --all-files || true
	flake8 src/ tests/ || true

format:  ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

test:  ## Run all tests
	$(PY) -m pytest tests/ -v

test-unit:  ## Run unit tests only
	$(PY) -m pytest tests/ -v -k "not e2e"

test-e2e:  ## Run end-to-end tests only
	$(PY) -m pytest tests/e2e/ -v

test-smoke:  ## Run smoke tests
	$(PY) -m pytest tests/test_smoke.py -v

coverage:  ## Run tests with coverage report
	$(PY) -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

coverage-xml:  ## Generate coverage XML for CI
	$(PY) -m pytest tests/ --cov=src --cov-report=xml

clean:  ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	$(PY) -m build

check:  ## Check package quality
	twine check dist/* || true

deploy-test:  ## Deploy to test PyPI (requires TWINE_USERNAME and TWINE_PASSWORD)
	$(PY) -m build
	twine upload --repository testpypi dist/*

deploy:  ## Deploy to PyPI (requires TWINE_USERNAME and TWINE_PASSWORD)
	$(PY) -m build
	twine upload dist/*

docs:  ## Build documentation
	cd Documentation && jupyter-book build . || echo "Documentation build failed - check if jupyter-book is installed"

docs-serve:  ## Serve documentation locally
	cd Documentation/_build/html && python -m http.server 8000

extract:  ## Run data extraction
	$(PY) src/01_main_extract_data.py --config src/config_file_example.yml

filter:  ## Run data filtering
	$(PY) src/02_filtering.py --config src/config_file_example.yml

rpv:  ## Run RPV modeling
	$(PY) src/03_RPV_modelling.py --config src/config_file_example.yml --band band1

pipeline:  ## Run complete pipeline (extract -> filter -> rpv)
	$(PY) src/01_main_extract_data.py --config src/config_file_example.yml
	$(PY) src/02_filtering.py --config src/config_file_example.yml
	$(PY) src/03_RPV_modelling.py --config src/config_file_example.yml --band band1

security:  ## Run security checks
	bandit -r src/ || true
	safety check || true

deps-update:  ## Update dependencies
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt

deps-check:  ## Check for outdated dependencies
	$(PIP) list --outdated

ci:  ## Run CI checks locally
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) coverage-xml
	$(MAKE) build
	$(MAKE) check

dev-setup:  ## Complete development setup
	$(MAKE) install
	$(MAKE) lint
	$(MAKE) test
	@echo "Development environment setup complete!"

# Legacy targets for backward compatibility
pre-commit: lint  ## Alias for lint

