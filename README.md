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

This will install UV if needed and set up all dependencies.

### Manual Setup

1. Install UV:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   uv sync --all-groups
   ```

3. Create credentials file from template:
   ```bash
   cp credentials.yml.dist credentials.yml
   ```

4. Download the ASR model (optional - will auto-download on first use):
   ```bash
   uv run python download_model.py
   ```

## Usage

### Using the Launcher (Easiest)

Simply run:
```bash
./startkeet.command
```

The launcher will automatically use UV to run the application.

### Manual Launch

```bash
uv run python -m superkeet.main
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

SuperKeet uses UV for fast, reliable dependency management with isolated virtual environments.

### Troubleshooting

If you encounter dependency errors (like `ImportError: Numba needs NumPy 2.2 or less`):

```bash
# Remove and recreate the environment
rm -rf .venv
uv sync --all-groups
```

For detailed information about dependency management, see:

📖 **[Dependency Management Guide](docs/DEPENDENCY_MANAGEMENT.md)**

## Development

Run quality checks:
```bash
./checkpython.sh
```

Run tests:
```bash
uv run pytest
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
