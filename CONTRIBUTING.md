# Contributing to Ontogentelechy

Thank you for your interest in contributing to Ontogentelechy! This document provides guidelines for contributing to the project.

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

---

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

### Suggesting Enhancements

We welcome suggestions for new features or improvements:
- Describe the enhancement clearly
- Explain why it would be useful
- Provide examples if possible

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Add tests** for new functionality
5. **Ensure tests pass** (`pytest`)
6. **Format code** (`black ontogentelechy/` and `isort ontogentelechy/`)
7. **Commit changes** (`git commit -m 'Add amazing feature'`)
8. **Push to branch** (`git push origin feature/amazing-feature`)
9. **Open a Pull Request**

---

## Development Setup

### Clone the repository

```bash
git clone https://github.com/o9nn/ontogentelechy.git
cd ontogentelechy
```

### Install in development mode

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
```

### Format code

```bash
black ontogentelechy/
isort ontogentelechy/
```

### Type checking

```bash
mypy ontogentelechy/
```

---

## Coding Standards

### Style Guide

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Add type hints where appropriate
- Write docstrings for all public functions and classes

### Documentation

- Update README.md if adding new features
- Add docstrings following Google style
- Update relevant documentation in `docs/`

### Testing

- Write tests for new functionality
- Maintain or improve code coverage
- Use descriptive test names
- Include both unit and integration tests

---

## Project Structure

```
ontogentelechy/
├── ontogentelechy/       # Main package
│   ├── core.py          # Core framework
│   ├── fitness.py       # Fitness functions
│   └── examples.py      # Example teloi
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Usage examples
├── setup.py             # Package setup
├── pyproject.toml       # Build configuration
└── README.md            # Main documentation
```

---

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
feat: add new teleological fitness function
fix: correct actualization gradient computation
docs: update README with new examples
test: add tests for phase transitions
refactor: simplify attractor gradient calculation
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or changes
- `refactor:` - Code refactoring
- `style:` - Formatting changes
- `chore:` - Maintenance tasks

---

## Questions?

If you have questions, feel free to:
- Open an issue for discussion
- Contact the maintainers
- Check existing documentation

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Ontogentelechy! 🎯
