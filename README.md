<div align="center">

# SuperKeet

**Privacy-First Voice-to-Text for macOS**

*Fast, accurate, and completely offline speech recognition for your Mac*

[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-000000?style=flat&logo=apple&logoColor=white)](https://www.apple.com/macos/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![FOSS](https://img.shields.io/badge/100%25-Open%20Source-green.svg)](https://github.com/cspenn/SuperKeet)

[Features](#-features) | [Installation](#-installation) | [Usage](#-usage) | [FAQ](#-faq)

</div>

---

## What is SuperKeet?

SuperKeet transforms your voice into text **anywhere on your Mac** with just a hotkey press. Unlike cloud-based dictation tools, SuperKeet runs **100% locally** on your machine—your voice never leaves your computer.

Perfect for:
- **Writers** who think faster than they type
- **Privacy-conscious professionals** who handle sensitive information
- **Developers** who want dictation in their IDE or terminal
- **Anyone** who wants fast, accurate voice input without cloud services

### Why SuperKeet?

| Feature | SuperKeet | Cloud Services |
|---------|-----------|----------------|
| **Privacy** | 100% offline, nothing sent to cloud | Your voice is uploaded |
| **Speed** | Up to 60x real-time on Apple Silicon | Network dependent |
| **Cost** | Free & Open Source | Subscription fees |
| **Internet** | Works offline | Requires connection |
| **Apps** | Works everywhere | Limited integration |

---

## Features

### Core Capabilities
- **Push-to-Talk Dictation** - Hold `Cmd + Shift + Space`, speak, release—text appears
- **Blazing Fast** - Up to 60x real-time transcription speed on Apple Silicon
- **100% Private** - All processing happens locally, no cloud, no tracking
- **Universal** - Works in any app that accepts text input
- **Accurate** - Uses NVIDIA's Parakeet ASR model with smart punctuation
- **Customizable** - Configure hotkeys, audio settings, and behavior

### Technical Highlights
- System tray app—stays out of your way
- Automatic text injection via clipboard
- Optional transcript logging
- Real-time audio waveform visualization
- Configurable audio devices
- Debug mode for troubleshooting

---

## System Requirements

**Required:**
- macOS with **Apple Silicon** (M1, M2, M3, M4)
- Python 3.11 or higher
- ~2GB available RAM
- ~600MB for AI model (one-time download)

**Permissions:**
- Microphone access (for recording)
- Accessibility access (for text injection)

> **Note:** SuperKeet requires Apple Silicon. Intel Macs are not supported due to the MLX framework requirement.

---

## Installation

### Quick Start (Recommended)

1. **Install ffmpeg** (required for audio processing):
   ```bash
   brew install ffmpeg
   ```

2. **Clone and setup** SuperKeet:
   ```bash
   git clone https://github.com/cspenn/SuperKeet.git
   cd SuperKeet
   ./setup.sh
   ```

3. **Launch** the application:
   ```bash
   ./startkeet.command
   ```

That's it! The first launch will download the AI model (~600MB, one-time).

<details>
<summary><b>Detailed Installation Instructions</b></summary>

### Prerequisites

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ffmpeg
brew install ffmpeg

# Python 3.11+ (usually pre-installed on modern macOS)
python3 --version
```

### Manual Setup

1. **Install UV** (fast dependency manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync --all-groups
   ```

3. **Create config file**:
   ```bash
   cp credentials.yml.dist credentials.yml
   ```

4. **Optional: Pre-download the AI model**:
   ```bash
   uv run python download_model.py
   ```

</details>

---

## Usage

### Basic Operation

1. **Launch SuperKeet** (if not already running):
   ```bash
   ./startkeet.command
   ```

2. **Look for the microphone icon** in your menu bar

3. **Press and hold** `Cmd + Shift + Space`

4. **Speak** your text clearly

5. **Release** the hotkey when done

6. **Text appears** automatically in your active application!

### Customization

Edit `config.yml` to customize:

```yaml
# Change the hotkey
hotkey:
  combination: "ctrl+space"  # Options: ctrl+space, cmd+space, etc.

# Adjust audio settings
audio:
  sample_rate: 16000
  gain: 2.0  # Increase if microphone is quiet

# Enable transcript logging
transcripts:
  enabled: true
  directory: "transcripts"
```

### Menu Bar Options

Right-click the menu bar icon for:
- Recent transcriptions (quick re-use)
- Settings
- Quit

---

## Configuration

SuperKeet is highly configurable through `config.yml`:

| Setting | Description | Default |
|---------|-------------|---------|
| `hotkey.combination` | Global hotkey | `ctrl+space` |
| `audio.device` | Specific microphone | Auto-detect |
| `audio.gain` | Microphone volume boost | `2.0` |
| `transcripts.enabled` | Save transcripts to disk | `true` |
| `text.auto_paste` | Auto-paste after transcription | `true` |
| `logging.level` | Log verbosity | `DEBUG` |

See [config.yml](config.yml) for all available options.

---

## FAQ

<details>
<summary><b>Why do I need Accessibility permissions?</b></summary>

SuperKeet needs Accessibility permissions to automatically paste transcribed text into your active application. Without it, text will be copied to your clipboard but not pasted.

**To grant access:**
1. Go to System Settings > Privacy & Security > Accessibility
2. Click the lock to make changes
3. Add Terminal (or your terminal app) to the list
4. Restart SuperKeet

</details>

<details>
<summary><b>Does SuperKeet work without an internet connection?</b></summary>

Yes! After the initial model download (~600MB, one-time), SuperKeet works 100% offline. Your voice never leaves your computer.

</details>

<details>
<summary><b>Can I use a different hotkey?</b></summary>

Absolutely! Edit `config.yml` and change the `hotkey.combination` value. Supported combinations:
- `ctrl+space`
- `cmd+space`
- `cmd+shift+space`
- And more...

</details>

<details>
<summary><b>What languages are supported?</b></summary>

Currently, SuperKeet supports **English only**. The Parakeet model is optimized for English and provides the best accuracy and speed for this language.

</details>

<details>
<summary><b>Will this work on Intel Macs?</b></summary>

No, SuperKeet requires Apple Silicon (M1/M2/M3/M4). The MLX framework used for AI acceleration is Apple Silicon-only.

</details>

<details>
<summary><b>How accurate is the transcription?</b></summary>

SuperKeet uses NVIDIA's Parakeet ASR model, which achieves state-of-the-art accuracy on English benchmarks. It includes smart punctuation and capitalization. Accuracy depends on:
- Clear speech
- Good microphone quality
- Minimal background noise

</details>

---

## Troubleshooting

### Text Not Being Pasted Automatically

**Solution:** Grant Accessibility permissions (see FAQ above)

### "ImportError: Numba needs NumPy 2.2 or less"

**Solution:** Reset your environment:
```bash
rm -rf .venv
./setup.sh
```

### Model Download Fails

**Solution:** Try manual download:
```bash
uv run python -c "from parakeet_mlx import from_pretrained; from_pretrained('mlx-community/parakeet-tdt-0.6b-v3')"
```

### Microphone Not Working

**Solution:** List available devices and configure in `config.yml`:
```bash
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

## Development

### Running Tests
```bash
uv run pytest
```

### Quality Checks
```bash
./checkpython.sh
```

---

## Contributing

SuperKeet is open source and contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run quality checks (`./checkpython.sh`)
5. Commit your changes
6. Push to the branch
7. Open a Pull Request

See [AGENTS.md](AGENTS.md) for AI-assisted development guidelines.

---

## License

SuperKeet is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **NVIDIA** for the [Parakeet ASR model](https://github.com/NVIDIA/NeMo)
- **MLX Community** for the Apple Silicon optimization
- **Parakeet-MLX** project for the Python bindings

---

<div align="center">

**Built with care for the macOS community**

[Report Bug](https://github.com/cspenn/SuperKeet/issues) | [Request Feature](https://github.com/cspenn/SuperKeet/issues)

</div>
