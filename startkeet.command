#!/bin/bash
# SuperKeet Launcher - Uses UV for dependency management

# Get the directory where this script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

echo ""
echo "════════════════════════════════════════"
echo "       SuperKeet Launcher"
echo "════════════════════════════════════════"
echo ""

# Check for UV and uv.lock
if command_exists uv && [ -f "uv.lock" ]; then
    echo "✓ Found UV environment (uv.lock)"
    echo "✓ Running with UV..."
    echo ""
    uv run python -m superkeet.main
    exit $?
fi

# UV not found or uv.lock missing
echo ""
echo "════════════════════════════════════════"
echo "❌ ERROR: UV Environment Not Found"
echo "════════════════════════════════════════"
echo ""

if ! command_exists uv; then
    echo "UV is not installed."
    echo ""
    echo "To install UV, run:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
else
    echo "UV is installed but uv.lock is missing."
    echo ""
fi

echo "To set up SuperKeet, run:"
echo "  ./setup.sh"
echo ""
exit 1
