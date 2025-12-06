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

### 2. Install Dependency Manager

SuperKeet supports both **Poetry** (stable) and **UV** (fast). Choose one:

**Option A: Poetry (Recommended for most users)**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your PATH (add to ~/.zshrc or ~/.bash_profile):
```bash
export PATH="$HOME/.local/bin:$PATH"
```

**Option B: UV (Recommended for speed)**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Clone and Setup Project

```bash
# Clone the repository (or navigate to your project directory)
cd /Users/cspenn/Documents/github/superkeet

# Quick setup (automatically detects UV or Poetry)
./setup.sh

# Or manually with Poetry:
poetry install

# Or manually with UV:
./install-uv.sh

# Copy credentials template (if not done automatically)
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

**Using the launcher (easiest)**:
```bash
./startkeet.command
```

**Manual launch with Poetry**:
```bash
poetry run python -m superkeet.main
```

**Manual launch with UV**:
```bash
source .venv/bin/activate
python -m superkeet.main
```

## Troubleshooting

### Dependency Errors

If you encounter errors like `ImportError: Numba needs NumPy 2.2 or less`:

```bash
# Quick fix - run the setup script
./setup.sh

# Or reset your environment:
# For Poetry:
poetry env remove --all
poetry install

# For UV:
rm -rf .venv
./install-uv.sh
```

For comprehensive troubleshooting and dependency management information, see:
📖 **[Dependency Management Guide](docs/DEPENDENCY_MANAGEMENT.md)**

### Import Errors
If you get import errors, ensure you're running in a virtual environment:
```bash
# With Poetry:
poetry shell  # Enter Poetry environment
python -m superkeet.main

# With UV:
source .venv/bin/activate
python -m superkeet.main

# Or just use the launcher:
./startkeet.command
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