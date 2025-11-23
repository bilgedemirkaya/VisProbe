# Contributing to Visfuzz

Thank you for your interest in contributing to Visfuzz! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project follows a code of conduct that all contributors are expected to adhere to. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, PyTorch version)
- Any relevant code snippets or error messages

### Suggesting Enhancements

We welcome feature requests! Please create an issue with:
- A clear description of the feature
- Use cases and examples
- Any implementation ideas you may have

### Pull Requests

1. **Fork the repository** and create a branch from the main branch
2. **Make your changes** following the code style guidelines below
3. **Test your changes** thoroughly
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description of your changes

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/Visfuzz.git
cd Visfuzz
```

2. Install in development mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

3. Verify installation:
```bash
visprobe --help
```

## Code Style Guidelines

### Python Style
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Use type hints where appropriate

### Formatting
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (Black formatter default)
- Use Black for code formatting:
```bash
black src/
```

### Documentation
- Update README.md for user-facing changes
- Update API_OVERVIEW.md for architectural changes
- Update COMPREHENSIVE_API_REFERENCE.md for API changes
- Add examples in test-files/ for new features

## Project Structure

```
Visfuzz/
├── src/visprobe/
│   ├── api/          # Core API and decorators
│   ├── cli/          # Command-line interface
│   ├── properties/   # Robustness properties
│   └── strategies/   # Perturbation strategies
├── src/test-files/   # Example tests
└── docs/             # Documentation
```

## Adding New Features

### Adding a New Perturbation Strategy

1. Create a new class in `src/visprobe/strategies/` inheriting from `Strategy`
2. Implement the `resolve(img, **kwargs)` method
3. Add documentation and examples
4. Create a test file in `test-files/` demonstrating usage

Example:
```python
from visprobe.strategies.base import Strategy

class MyNewStrategy(Strategy):
    """Description of the strategy."""

    def resolve(self, img, **kwargs):
        # Implementation
        return perturbed_img
```

### Adding a New Robustness Property

1. Create a new class in `src/visprobe/properties/` inheriting from `Property`
2. Implement the `evaluate(original, perturbed, **kwargs)` classmethod
3. Add documentation and examples
4. Create a test file demonstrating usage

Example:
```python
from visprobe.properties.base import Property

class MyNewProperty(Property):
    """Description of the property."""

    @classmethod
    def evaluate(cls, original, perturbed, **kwargs):
        # Implementation
        return result, metrics
```

## Testing

### Running Tests
Currently, Visfuzz uses example-based testing. Run example tests with:
```bash
visprobe run src/test-files/test_example.py
```

### Adding Tests
- Add new test files to `src/test-files/`
- Use descriptive names: `test_<feature>.py`
- Include comments explaining the test purpose
- Test edge cases and error conditions

## Documentation

### Docstring Format
Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """Brief description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
    """
    pass
```

## Commit Message Guidelines

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters
- Add detailed description in the body if needed

Examples:
```
Add FGSM epsilon parameter validation

Fix dashboard crash when no results available

Update README with new installation instructions
```

## Release Process

Releases are managed by maintainers. If you believe a release is needed:
1. Create an issue requesting a release
2. List notable changes since the last release
3. Suggest a version number following [Semantic Versioning](https://semver.org/)

## Questions?

If you have questions about contributing:
- Check existing [Issues](https://github.com/bilgedemirkaya/Visfuzz/issues)
- Create a new issue with your question
- Join [GitHub Discussions](https://github.com/bilgedemirkaya/Visfuzz/discussions)

## License

By contributing to Visfuzz, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Visfuzz!
