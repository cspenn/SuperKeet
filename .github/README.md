# SuperKeet CI/CD Documentation

## Overview

SuperKeet uses GitHub Actions for automated testing, code quality checks, and releases. This document describes the CI/CD workflows and how to use them effectively.

## Workflows

### 1. Continuous Integration (`ci.yml`)

**Triggers:** Push/PR to main or develop branches

**Jobs:**
- **Test**: Runs on multiple macOS versions (latest, 13, 12)
  - Installs system dependencies (PortAudio)
  - Runs pytest with coverage reporting
  - Uploads coverage to Codecov
  - Performs basic smoke test
  
- **Build**: Creates application binary (main branch only)
  - Uses PyInstaller to create macOS app bundle
  - Uploads build artifacts
  
- **Code Quality**: Runs on Ubuntu for speed
  - Advanced linting with ruff
  - Security scanning with bandit
  - Type checking with mypy (if configured)
  - Uploads security reports

### 2. Code Quality (`quality.yml`)

**Triggers:** Push/PR to any branch

**Features:**
- Pre-commit hook validation
- Code complexity analysis with radon
- Dead code detection with vulture
- Security vulnerability scanning
- Faster feedback loop for developers

### 3. Release (`release.yml`)

**Triggers:** 
- Release publication
- Manual workflow dispatch

**Features:**
- Builds production-ready macOS application
- Creates DMG installer (if create-dmg available)
- Attaches assets to GitHub releases
- Verifies build integrity
- Basic smoke testing

## Local Development Setup

### Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks for this repository
pre-commit install

# Run hooks on all files (optional)
pre-commit run --all-files
```

### Running Tests Locally

```bash
# Basic test run
python -m pytest

# With coverage
python -m pytest --cov=src --cov-report=html

# Run specific test categories
python -m pytest -m "unit"
python -m pytest -m "integration" 
python -m pytest -m "not slow"
```

### Code Quality Checks

```bash
# Linting
ruff check src/ tests/
ruff format src/ tests/

# Security scan
bandit -r src/ -ll

# Type checking (if configured)
mypy src/

# Application validation
./checkpython.sh
```

## Branch Strategy

- **main**: Production-ready code, triggers full CI/CD pipeline
- **develop**: Integration branch, runs all tests but no builds
- **feature/***: Feature branches, runs quality checks

## Coverage and Quality Metrics

### Test Coverage
- Target: >80% code coverage
- Reports uploaded to Codecov for main branch
- HTML reports generated locally in `htmlcov/`

### Code Quality Standards
- Line length: 88 characters (ruff)
- Security: Bandit scanning enabled
- Complexity: Monitored with radon (C+ rating target)
- Dead code: Detected with vulture

### Performance Benchmarks
- Application startup: <5 seconds
- Test suite: <2 minutes on macOS
- Build time: <10 minutes

## Troubleshooting

### Common CI Issues

1. **PortAudio installation fails**
   - macOS runner issue - usually retry fixes it
   - Check brew formulae availability

2. **Qt tests fail with display issues**
   - Uses QT_QPA_PLATFORM=offscreen
   - DISPLAY=:99.0 for virtual display

3. **PyInstaller build fails**
   - Check superkeet.spec configuration
   - Verify all dependencies included

4. **Coverage upload fails**
   - Requires CODECOV_TOKEN secret
   - Only runs on main branch with Python 3.11

### Local Development Issues

1. **Pre-commit hooks fail**
   ```bash
   pre-commit clean
   pre-commit install --install-hooks
   ```

2. **Tests fail locally but pass in CI**
   - Check Python version (3.11 required)
   - Verify system dependencies installed
   - Check QT environment variables

3. **Build artifacts missing**
   - Ensure PyInstaller spec file exists
   - Check file permissions and paths

## Security Considerations

- Bandit security scanning enabled
- No secrets stored in code
- Dependencies scanned with safety
- Private key detection in pre-commit

## Contributing

1. Create feature branch from develop
2. Make changes with tests
3. Run pre-commit hooks locally
4. Push and create PR to develop
5. CI must pass before merge
6. Releases created from main branch

For questions about the CI/CD setup, check the workflow files in `.github/workflows/` or create an issue.