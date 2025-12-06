#!/bin/bash
# SuperKeet Launcher - Auto-detects and activates virtual environment

# Get the directory where this script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to validate Python environment
validate_env() {
    python -c "import numpy, numba; assert numpy.__version__.startswith('2.2'), 'NumPy version mismatch'; assert numba.__version__.startswith('0.62'), 'Numba version mismatch'" 2>/dev/null
    return $?
}

echo ""
echo "════════════════════════════════════════"
echo "       SuperKeet Launcher"
echo "════════════════════════════════════════"
echo ""

# Strategy 1: Try UV virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "✓ Found UV virtual environment (.venv)"
    source .venv/bin/activate
    if validate_env; then
        echo "✓ Environment validated (NumPy 2.2.x, Numba 0.62.x)"
        echo ""
        python -m superkeet.main
        exit $?
    else
        echo "⚠ Warning: UV environment has dependency issues"
        deactivate 2>/dev/null
    fi
fi

# Strategy 2: Try Poetry virtual environment
if command_exists poetry; then
    echo "✓ Attempting to use Poetry environment..."
    POETRY_VENV=$(poetry env info --path 2>/dev/null)
    if [ -n "$POETRY_VENV" ] && [ -d "$POETRY_VENV" ]; then
        source "$POETRY_VENV/bin/activate"
        if validate_env; then
            echo "✓ Environment validated (NumPy 2.2.x, Numba 0.62.x)"
            echo ""
            poetry run python -m superkeet.main
            exit $?
        else
            echo "⚠ Warning: Poetry environment has dependency issues"
            deactivate 2>/dev/null
        fi
    fi
fi

# Strategy 3: Try running with Poetry directly (creates env if needed)
if command_exists poetry; then
    echo "✓ Running with Poetry (may install dependencies)..."
    echo ""
    poetry install
    poetry run python -m superkeet.main
    exit $?
fi

# Strategy 4: System Python (last resort - not recommended)
echo "⚠ Warning: No virtual environment found, using system Python"
echo "  This is not recommended and may cause conflicts"
echo ""
if validate_env; then
    python -m superkeet.main
else
    echo ""
    echo "════════════════════════════════════════"
    echo "❌ ERROR: Dependency Issues Detected"
    echo "════════════════════════════════════════"
    echo ""
    echo "Your Python environment has incompatible dependencies."
    echo ""
    echo "To fix this, run the setup script:"
    echo "  ./setup.sh"
    echo ""
    echo "Or install dependencies manually:"
    echo ""
    echo "  Option 1 (Poetry - Recommended):"
    echo "    poetry install"
    echo ""
    echo "  Option 2 (UV - Faster):"
    echo "    ./install-uv.sh"
    echo ""
    exit 1
fi
