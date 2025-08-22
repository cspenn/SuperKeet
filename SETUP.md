# SuperKeet Setup Guide

## System Requirements

- **macOS**: Apple Silicon (M1/M2/M3) Mac required
- **Python**: Version 3.11 or higher
- **Homebrew**: Package manager for macOS

## Step-by-Step Setup

### 1. Install System Dependencies

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ffmpeg (required for audio processing)
brew install ffmpeg

# Install Python 3.11 if needed
brew install python@3.11
```

### 2. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your PATH (add to ~/.zshrc or ~/.bash_profile):
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 3. Clone and Setup Project

```bash
# Clone the repository (or navigate to your project directory)
cd /Users/cspenn/Documents/github/superkeet

# Install Python dependencies
poetry install

# Copy credentials template
cp credentials.yml.dist credentials.yml

# Make check script executable
chmod +x checkpython.sh
```

### 4. Download ASR Model (Optional)

The model will auto-download on first use, but you can pre-download it:
```bash
poetry run python download_model.py
```

### 5. Test Components

Run the component test to verify everything is installed correctly:
```bash
poetry run python test_components.py
```

### 6. Grant Permissions

When you first run SuperKeet, macOS will ask for:
1. **Microphone Access**: Required for audio recording
2. **Accessibility Access**: Required for text injection

Grant these permissions in System Preferences > Security & Privacy.

### 7. Run SuperKeet

```bash
poetry run python -m src.main
```

## Troubleshooting

### Import Errors
If you get import errors, ensure you're running with Poetry:
```bash
poetry shell  # Enter Poetry environment
python -m src.main
```

### Audio Device Issues
List available audio devices:
```bash
poetry run python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Model Download Issues
If the model fails to download:
1. Check your internet connection
2. Try manual download:
   ```bash
   poetry run python -c "from parakeet_mlx import from_pretrained; from_pretrained('mlx-community/parakeet-tdt-0.6b-v2')"
   ```

### Permission Issues
If text injection doesn't work:
1. Go to System Preferences > Security & Privacy > Privacy > Accessibility
2. Add Terminal (or your IDE) to the allowed apps
3. Restart SuperKeet

## Verification Checklist

- [ ] ffmpeg installed (`ffmpeg -version`)
- [ ] Python 3.11+ installed (`python3 --version`)
- [ ] Poetry installed (`poetry --version`)
- [ ] Dependencies installed (`poetry show`)
- [ ] Component test passes (`poetry run python test_components.py`)
- [ ] Microphone permission granted
- [ ] Accessibility permission granted