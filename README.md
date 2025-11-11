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

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

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
   ```

## Usage

Run the application:
```bash
python -m superkeet.main
```

Or with Poetry:
```bash
poetry run python -m superkeet.main
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