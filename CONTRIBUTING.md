# Contributing to HVSR Progressive Layer Stripping

We welcome contributions to this project! This document outlines how to contribute effectively.

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/mersadfathizadeh1995/hvstrip-progressive.git
   cd hvstrip-progressive
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

## 🔧 Development Guidelines

### Code Style
- **Black**: Use `black` for code formatting
- **Flake8**: Follow PEP 8 guidelines
- **Type hints**: Use type annotations where appropriate
- **Docstrings**: Follow NumPy/SciPy docstring conventions

### Testing
- Write tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for good test coverage

### Documentation
- Update README.md for new features
- Add docstrings to all public functions/classes
- Include examples for complex functionality

## 📝 Contribution Process

### 1. Create an Issue
Before making changes, create an issue to discuss:
- Bug reports
- Feature requests
- Enhancement proposals

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes
- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes
```bash
# Run tests
pytest tests/

# Check code style
black hvstrip_progressive/
flake8 hvstrip_progressive/

# Test installation
pip install -e .
```

### 5. Submit Pull Request
- Provide clear description of changes
- Reference related issues
- Include test results
- Update CHANGELOG.md if applicable

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version
- Package version
- Operating system
- Minimal example to reproduce
- Full error traceback

## 💡 Feature Requests

For new features, please provide:
- Clear description of the feature
- Use case and motivation
- Proposed API design (if applicable)
- Willingness to implement

## 🔬 Scientific Contributions

We especially welcome:
- New peak detection algorithms
- Advanced visualization methods
- Performance optimizations
- Validation studies
- Documentation improvements

## 📚 Documentation

Help improve documentation by:
- Adding examples
- Clarifying existing docs
- Fixing typos
- Adding tutorials

## 🙏 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn
- Maintain professional communication

## 🏆 Recognition

Contributors will be acknowledged in:
- CHANGELOG.md
- README.md contributors section
- Git commit history

Thank you for contributing to HVSR Progressive Layer Stripping Analysis!
