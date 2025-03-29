#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$(pwd)" =~ " " ]]; then echo This script relies on Miniconda which can not be silently installed under a path with spaces. && exit; fi

# Check for special characters in installation path
if [[ "$(pwd)" =~ [!#\$%&\(\)*+,;<=>?@\[\]\^`{|}~] ]]; then
    echo "WARNING: Special characters were detected in the installation path!"
    echo "This can cause the installation to fail!"
fi

# deactivate existing conda envs as needed to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null

# Determine system architecture
OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*)    OS_ARCH="x86_64";;
    arm64*)     OS_ARCH="aarch64";;
    aarch64*)   OS_ARCH="aarch64";;
    *)          echo "Unknown system architecture: $OS_ARCH! This script runs only on x86_64 or arm64" && exit
esac

# Define paths
HOME_DIR="$(pwd)"
PARENT_DIR="$(realpath "$HOME_DIR/..")"
INSTALL_DIR="$HOME_DIR/installer_files"
CONDA_HOME="$INSTALL_DIR/conda"
ENV_HOME="$INSTALL_DIR/env"
CONDA_PARENT="$PARENT_DIR/installer_files/conda"
ENV_PARENT="$PARENT_DIR/installer_files/env"

# Read user_env.txt into ENV_FLAG
ENV_FLAG=""
if [[ -f "$INSTALL_DIR/user_env.txt" ]]; then
    ENV_FLAG=$(< "$INSTALL_DIR/user_env.txt")
fi

# Assign paths based on ENV_FLAG
if [[ "$ENV_FLAG" == "$ENV_HOME" ]]; then
    CONDA_ROOT_PREFIX="$CONDA_HOME"
    INSTALL_ENV_DIR="$ENV_HOME"
    goto activate_conda
fi
if [[ "$ENV_FLAG" == "$ENV_PARENT" ]]; then
    CONDA_ROOT_PREFIX="$CONDA_PARENT"
    INSTALL_ENV_DIR="$ENV_PARENT"
    goto activate_conda
fi

# First-run setup
echo "Welcome to ad_discordbot"

echo "Checking for existing Conda installation..."
if [[ -x "$CONDA_PARENT/bin/conda" ]]; then
    echo "The bot can be integrated with your existing text-generation-webui environment."
    echo "[A] Integrate with TGWUI *Recommended*"
    echo "[B] Create and use own environment"
    echo "[N] Nothing, exit script"
    read -p "Enter A, B, or N: " USER_CHOICE
    USER_CHOICE=${USER_CHOICE,,} # Convert to lowercase

    if [[ "$USER_CHOICE" == "a" ]]; then
        CONDA_ROOT_PREFIX="$CONDA_PARENT"
        INSTALL_ENV_DIR="$ENV_PARENT"
        goto activate_conda
    elif [[ "$USER_CHOICE" == "b" ]]; then
        setup_conda
        goto activate_conda
    elif [[ "$USER_CHOICE" == "n" ]]; then
        echo "Exiting script."
        exit
    else
        echo "Invalid input. Please enter A, B, or N."
        exec "$0"
    fi
else
    echo "No existing Conda installation detected. Install standalone?"
    echo "[Y] Yes, install standalone"
    echo "[N] No, exit"
    read -p "Enter Y or N: " USER_CHOICE
    USER_CHOICE=${USER_CHOICE,,}
    
    if [[ "$USER_CHOICE" == "y" ]]; then
        setup_conda
        goto activate_conda
    elif [[ "$USER_CHOICE" == "n" ]]; then
        echo "Exiting script."
        exit
    else
        echo "Invalid input. Please enter Y or N."
        exec "$0"
    fi
fi

# Function to install Conda and setup environment
setup_conda() {
    mkdir -p "$INSTALL_DIR"
    MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/Miniconda3-py311_24.11.1-0-Linux-${OS_ARCH}.sh"
    echo "Downloading Miniconda..."
    curl -L "$MINICONDA_DOWNLOAD_URL" -o "$INSTALL_DIR/miniconda_installer.sh"
    
    echo "Installing Miniconda..."
    bash "$INSTALL_DIR/miniconda_installer.sh" -b -p "$CONDA_HOME"
    rm "$INSTALL_DIR/miniconda_installer.sh"
    
    if [[ ! -x "$CONDA_HOME/bin/conda" ]]; then
        echo "Miniconda installation failed."
        exit 1
    fi
    
    echo "Creating Conda environment..."
    "$CONDA_HOME/bin/conda" create -y -k --prefix "$ENV_HOME" python=3.11
    
    if [[ ! -x "$ENV_HOME/bin/python" ]]; then
        echo "Conda environment creation failed."
        exit 1
    fi
    echo "Conda environment created!"
}

# Function to activate Conda and run the script
activate_conda() {
    echo "Trying to activate Conda from: $CONDA_ROOT_PREFIX/bin/conda"
    if [[ ! -x "$CONDA_ROOT_PREFIX/bin/conda" ]]; then
        echo "Conda activation script not found! Please check your environment and try running the script again."
        rm -f "$ENV_FLAG"
        exit 1
    fi
    
    source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh"
    conda activate "$INSTALL_ENV_DIR"
    
    if [[ $? -ne 0 ]]; then
        echo "Failed to activate the Conda environment. Exiting..."
        exit 1
    fi
    
    echo "Conda activated successfully."
    python "$HOME_DIR/one_click.py" --conda-env-path "$INSTALL_ENV_DIR" "$@"
}

activate_conda "$@"
