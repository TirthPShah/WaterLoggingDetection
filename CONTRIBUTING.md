# Contributing to AI-CCTV Waterlogging Detection & Forecasting System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/waterlogging-detection.git
   cd waterlogging-detection
   ```
3. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install development dependencies
   ```

## 🔧 Development Workflow

1. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Test your changes**:
   ```bash
   python -m pytest tests/
   ```

4. **Format your code**:
   ```bash
   black src/ *.py
   isort src/ *.py
   flake8 src/
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

## 📝 Code Style

- Follow **PEP 8** style guide
- Use **type hints** for function signatures
- Write **docstrings** for all public functions/classes
- Keep functions **focused and small** (< 50 lines preferred)
- Use **meaningful variable names**

### Example:

```python
def calculate_risk_score(
    detection_risk: float,
    forecast_risk: float,
    weights: Dict[str, float]
) -> float:
    """
    Calculate weighted risk score from detection and forecast.
    
    Args:
        detection_risk: Risk from detection model (0-1)
        forecast_risk: Risk from forecasting model (0-1)
        weights: Dictionary with 'detection' and 'forecast' keys
        
    Returns:
        Weighted risk score (0-1)
    """
    return (
        weights['detection'] * detection_risk +
        weights['forecast'] * forecast_risk
    )
```

## 🧪 Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## 📚 Documentation

- Update README.md if adding new features
- Add docstrings to new functions/classes
- Update configuration documentation if adding new settings
- Include usage examples for new functionality

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description** of the bug
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment** (OS, Python version, GPU, etc.)
6. **Error messages/logs**
7. **Screenshots** if applicable

## 💡 Feature Requests

When requesting features:

1. **Describe the feature** clearly
2. **Explain use case** and benefits
3. **Provide examples** if possible
4. **Consider implementation** complexity

## 🎯 Areas for Contribution

### High Priority
- [ ] Real-time streaming support (RTSP/RTMP)
- [ ] Web dashboard for monitoring
- [ ] Mobile/edge deployment optimization
- [ ] Comprehensive test suite
- [ ] Performance benchmarking

### Medium Priority
- [ ] Multi-camera coordination
- [ ] Weather API integration
- [ ] Alert system implementation
- [ ] Model compression/quantization
- [ ] Docker deployment

### Low Priority
- [ ] Additional visualization options
- [ ] More data augmentation techniques
- [ ] Additional model architectures
- [ ] Tutorials and notebooks

## 🤝 Code Review Process

1. Maintainers will review your PR within 1-2 weeks
2. Address review comments and update PR
3. Once approved, maintainers will merge your PR

## 📋 Commit Message Guidelines

Use clear, descriptive commit messages:

- `Add: new feature or capability`
- `Fix: bug fix`
- `Update: improvements to existing code`
- `Refactor: code restructuring without behavior change`
- `Docs: documentation updates`
- `Test: test additions or modifications`

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

## 📞 Getting Help

- Open an issue for questions
- Join our community discussions
- Check existing issues and documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
