# Contributing to Multi-angular UAV Reflectance Extractor

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.


## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Code of Conduct](#code-of-conduct)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic knowledge of Python, geospatial data processing, and remote sensing

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/proj_on_uav.git
   cd proj_on_uav
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ReneHeim/proj_on_uav.git
   ```

## Development Setup

### 1. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .
```

### 2. Set Up Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run code formatting and linting on every commit.

### 3. Verify Setup

```bash
# Run all tests
python -m pytest

# Run linting
make lint

# Check code formatting
make format
```

## Code Style

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters maximum
- **Import sorting**: Use `isort` (configured in `pyproject.toml`)
- **Code formatting**: Use `black` (configured in `pyproject.toml`)

### Type Hints

- Use type hints for all function parameters and return values
- Use `typing` module for complex types
- Example:
  ```python
  from typing import List, Optional, Union
  from pathlib import Path

  def process_data(
      input_path: Union[str, Path],
      output_dir: Optional[Path] = None
  ) -> List[Path]:
      """Process data and return list of output files."""
      pass
  ```

### Documentation

- Use docstrings for all public functions and classes
- Follow Google-style docstrings
- Example:
  ```python
  def calculate_reflectance(
      band_data: np.ndarray,
      solar_angles: tuple[float, float]
  ) -> np.ndarray:
      """Calculate reflectance from band data and solar angles.

      Args:
          band_data: Input band data array
          solar_angles: Tuple of (elevation, azimuth) angles in degrees

      Returns:
          Reflectance values array

      Raises:
          ValueError: If solar angles are invalid
      """
      pass
  ```

## Testing

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Group related tests in classes
- Use fixtures for common setup

Example:
```python
import pytest
from pathlib import Path

class TestDataExtraction:
    """Test data extraction functionality."""

    def test_extract_single_band(self, sample_data):
        """Test extraction of single band data."""
        result = extract_band(sample_data, band=1)
        assert result.shape == (100, 100)
        assert result.dtype == np.float32

    def test_extract_invalid_band(self, sample_data):
        """Test extraction with invalid band number."""
        with pytest.raises(ValueError, match="Invalid band"):
            extract_band(sample_data, band=10)
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_extraction.py

# Run tests in parallel
python -m pytest -n auto

# Run only unit tests
make test-unit

# Run only E2E tests
make test-e2e
```

### Test Data

- Use synthetic data for unit tests
- Create realistic but small datasets for integration tests
- Store test data in `tests/data/` directory
- Use `pytest.fixture` for reusable test data

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, focused commits
- Use descriptive commit messages
- Follow the conventional commit format:
  ```
  type(scope): description

  [optional body]
  [optional footer]
  ```

  Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### 3. Test Your Changes

```bash
# Run the full test suite
make ci

# Or run individual checks
make lint
make test
make coverage
```

### 4. Update Documentation

- Update README.md if needed
- Add docstrings for new functions
- Update configuration examples if you changed config structure

### 5. Submit Pull Request

1. Push your branch to your fork
2. Create a pull request against the `dev` branch
3. Fill out the PR template
4. Request review from maintainers

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] E2E tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment information**:
   - Operating system and version
   - Python version
   - Package versions (`pip freeze`)

2. **Steps to reproduce**:
   - Clear, step-by-step instructions
   - Sample data if possible

3. **Expected vs actual behavior**:
   - What you expected to happen
   - What actually happened

4. **Error messages**:
   - Full error traceback
   - Log files if available

### Feature Requests

For feature requests, please include:

1. **Problem description**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Use cases**: Who would benefit from this?
4. **Alternatives considered**: What other approaches were considered?

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Use welcoming and inclusive language
- Be collaborative and open to feedback
- Focus on what is best for the community
- Show empathy towards other community members

### Enforcement

- Unacceptable behavior will not be tolerated
- Violations will be addressed promptly and fairly
- Maintainers have the right to remove, edit, or reject contributions

## Getting Help

- **Documentation**: Check the [README](README.md) and [Documentation](Documentation/) folder
- **Issues**: Search existing [issues](https://github.com/ReneHeim/proj_on_uav/issues)
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact maintainers directly if needed

## Recognition

Contributors will be recognized in:

- The project README
- Release notes
- Academic publications (if applicable)

Thank you for contributing to the Multi-angular UAV Reflectance Extractor project!
