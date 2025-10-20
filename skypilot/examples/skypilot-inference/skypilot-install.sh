#!/bin/bash
# skypilot-install.sh: Idempotent installer for SkyPilot with Nebius support
set -euo pipefail

VENV_DIR="$HOME/venvs/skypilot"
PYTHON_BIN="python3"

# Create venv directory if it doesn't exist
mkdir -p "$HOME/venvs"

# Create virtual environment if not present
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating Python virtual environment at $VENV_DIR..."
  $PYTHON_BIN -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists at $VENV_DIR."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"


# Upgrade pip first
pip install --upgrade pip

# Install/upgrade SkyPilot with Nebius and Kubernetes support
pip install --upgrade "skypilot[nebius,kubernetes]"

# Ensure OS-level prerequisites for Kubernetes backend
OS_NAME="$(uname -s)"
case "$OS_NAME" in
  Darwin)
    echo "Detected macOS. Verifying Homebrew and required binaries (socat, netcat)..."
    if ! command -v brew >/dev/null 2>&1; then
      cat <<'EOS'
Homebrew not found. Please install Homebrew first:
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
Then re-run this script.
EOS
    else
      # Install socat if missing
      if ! command -v socat >/dev/null 2>&1; then
        echo "Installing socat via Homebrew..."
        brew install socat || true
      else
        echo "socat already installed."
      fi
      # Install GNU netcat if missing (SkyPilot requires GNU variant for portforward mode)
      if ! command -v nc >/dev/null 2>&1; then
        echo "Installing netcat via Homebrew..."
        brew install netcat || true
      else
        # macOS ships BSD nc; Homebrew's netcat provides GNU variant. Install if Brew netcat is absent.
        if ! brew list --versions netcat >/dev/null 2>&1; then
          echo "Installing netcat (GNU) via Homebrew..."
          brew install netcat || true
        else
          echo "netcat already installed via Homebrew."
        fi
      fi
    fi
    ;;
  Linux)
    echo "Detected Linux. If you're on Debian/Ubuntu, ensure 'socat' and 'netcat' are installed (may require sudo):"
    echo "  sudo apt update && sudo apt install -y socat netcat"
    ;;
  *)
    echo "Unrecognized OS ($OS_NAME). Please ensure 'socat' and 'netcat' are installed for Kubernetes portforward support."
    ;;
esac

echo "SkyPilot installation complete. To activate the environment in the future, run:"
echo "  source $VENV_DIR/bin/activate"
echo

echo "To verify your setup, run:"
echo "  sky check nebius"
echo "  sky check kubernetes"
