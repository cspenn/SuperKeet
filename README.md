# SuperKeet

SuperKeet is a FOSS voice-to-text application for macOS using the Parakeet-MLX ASR model.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11+
- ffmpeg (required for audio processing):
  ```bash
  brew install ffmpeg
  ```

## Installation

### Quick Setup (Recommended)

The easiest way to get started:

```bash
./setup.sh
```

This will automatically detect if you have UV or Poetry installed and set up the dependencies.

### Manual Setup

#### Option 1: Poetry (Stable)

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

#### Option 2: UV (Fast)

1. Install UV:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Run the UV installer:
   ```bash
   ./install-uv.sh
   ```

### Additional Setup

3. Create credentials file from template:
   ```bash
   cp credentials.yml.dist credentials.yml
   ```

4. Make the check script executable:
   ```bash
   chmod +x checkpython.sh
   ```

5. Download the ASR model (optional - will auto-download on first use):
   ```bash
   poetry run python download_model.py
   # or with UV:
   source .venv/bin/activate && python download_model.py
   ```

## Usage

### Using the Launcher (Easiest)

Simply run:
```bash
./startkeet.command
```

The launcher will automatically detect and activate your virtual environment.

### Manual Launch

With Poetry:
```bash
poetry run python -m superkeet.main
```

With UV:
```bash
source .venv/bin/activate
python -m superkeet.main
```

## Default Hotkey

The default hotkey combination is: **Cmd + Shift + Space**

Hold down the hotkey to record, release to transcribe and inject the text.

## Configuration

Edit `config.yml` to customize:
- Hotkey combination
- Audio settings
- ASR model
- UI icons
- Logging level

## Dependency Management

SuperKeet uses isolated virtual environments to prevent dependency conflicts. We support both Poetry and UV for dependency management.

### Troubleshooting

If you encounter dependency errors (like `ImportError: Numba needs NumPy 2.2 or less`):

1. **Quick fix**: Run the setup script
   ```bash
   ./setup.sh
   ```

2. **Poetry fix**: Reset the environment
   ```bash
   poetry env remove --all
   poetry install
   ```

3. **UV fix**: Recreate the environment
   ```bash
   rm -rf .venv
   ./install-uv.sh
   ```

For detailed information about dependency management, troubleshooting, and best practices, see:

📖 **[Dependency Management Guide](docs/DEPENDENCY_MANAGEMENT.md)**

## Development

Run quality checks:
```bash
./checkpython.sh
```

## System Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11+
- ffmpeg (install via Homebrew: `brew install ffmpeg`)
- Microphone permissions
- Accessibility permissions (for text injection)

## Model Information

SuperKeet uses the Parakeet-TDT-0.6b-v3 model from Hugging Face:
- Model ID: `mlx-community/parakeet-tdt-0.6b-v3`
- Size: ~600MB
- Language: English only
- Performance: Up to 60x real-time on Apple Silicon