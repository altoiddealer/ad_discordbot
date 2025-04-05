#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$(pwd)" =~ " " ]]; then echo This script relies on Miniconda which can not be silently installed under a path with spaces. && exit; fi

# Check for special characters in installation path
if [[ "$(pwd)" =~ [!#\$%&\(\)\*+,;<=>?@\[\]\^`{|}~] ]]; then
    echo "WARNING: Special characters were detected in the installation path!"
    echo "         This can cause the installation to fail!"
fi

# deactivate existing conda envs as needed to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2>/dev/null

# Determine architecture (M Series or Intel)
OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*)    OS_ARCH="x86_64";;
    arm64*)     OS_ARCH="arm64";;
    *)          echo "Unknown system architecture: $OS_ARCH! This script runs only on x86_64 or arm64" && exit
esac

# Define root directories
HOME_DIR="$(pwd)"
PARENT_DIR="$(cd "$HOME_DIR/.." && pwd)"

# Define configuration paths
CONDA_HOME="$HOME_DIR/installer_files/conda"
ENV_HOME="$HOME_DIR/installer_files/env"
CONDA_PARENT="$PARENT_DIR/installer_files/conda"
ENV_PARENT="$PARENT_DIR/installer_files/env"

# Read user_env.txt into ENV_FLAG
ENV_FLAG=""
if [[ -f "$HOME_DIR/installer_files/user_env.txt" ]]; then
    ENV_FLAG=$(cat "$HOME_DIR/installer_files/user_env.txt")
fi

# Assign paths and activate if env flag exists
if [[ "$ENV_FLAG" == "$ENV_HOME" ]]; then
    CONDA_ROOT_PREFIX="$CONDA_HOME"
    INSTALL_ENV_DIR="$ENV_HOME"
    goto activate_conda
elif [[ "$ENV_FLAG" == "$ENV_PARENT" ]]; then
    CONDA_ROOT_PREFIX="$CONDA_PARENT"
    INSTALL_ENV_DIR="$ENV_PARENT"
    goto activate_conda
fi

# Welcome message
echo "Welcome to ad_discordbot"
echo

# Check if conda environment exists
if [[ -d "$CONDA_PARENT/bin" ]]; then
    echo "The bot can be integrated with your existing text-generation-webui environment."
    echo "[A] Integrate with TGWUI *Recommended*"
    echo "[B] Create and use own environment"
    echo "[N] Nothing, exit script"
    read -p "Enter A, B, or N: " USER_CHOICE
    USER_CHOICE=${USER_CHOICE:0:1}

    case "$USER_CHOICE" in
        [Aa])
            CONDA_ROOT_PREFIX="$CONDA_PARENT"
            INSTALL_ENV_DIR="$ENV_PARENT"
            goto activate_conda
            ;;
        [Bb])
            setup_conda
            goto activate_conda
            ;;
        [Nn])
            echo "Exiting script."
            exit
            ;;
        *)
            echo "Invalid input. Please enter A, B, or N."
            exec "$0"
            ;;
    esac
else
    echo "This bot can be integrated with text-generation-webui, but it was not detected."
    echo "Install the bot as standalone? This option can be changed later via the update-wizard script."
    echo "[Y] Yes, install standalone"
    echo "[N] No, exit"
    read -p "Enter Y or N: " USER_CHOICE
    USER_CHOICE=${USER_CHOICE:0:1}

    case "$USER_CHOICE" in
        [Yy])
            setup_conda
            goto activate_conda
            ;;
        [Nn])
            echo "Exiting script."
            exit
            ;;
        *)
            echo "Invalid input. Please enter Y or N."
            exec "$0"
            ;;
    esac
fi

# Function to install conda and setup environment
setup_conda() {
    INSTALL_DIR="$HOME_DIR/installer_files"
    CONDA_ROOT_PREFIX="$CONDA_HOME"
    INSTALL_ENV_DIR="$ENV_HOME"
    MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/Miniconda3-py311_24.11.1-0-MacOSX-${OS_ARCH}.sh"

    mkdir -p "$INSTALL_DIR"
    echo "Downloading Miniconda..."
    curl -L "$MINICONDA_DOWNLOAD_URL" -o "$INSTALL_DIR/miniconda_installer.sh"

    echo "Installing Miniconda..."
    bash "$INSTALL_DIR/miniconda_installer.sh" -b -p "$CONDA_ROOT_PREFIX"
    rm "$INSTALL_DIR/miniconda_installer.sh"

    if [[ ! -f "$CONDA_ROOT_PREFIX/bin/conda" ]]; then
        echo "Miniconda installation failed."
        exit
    fi

    echo "Creating conda environment..."
    "$CONDA_ROOT_PREFIX/bin/conda" create -y -k --prefix "$INSTALL_ENV_DIR" python=3.11

    if [[ ! -f "$INSTALL_ENV_DIR/bin/python" ]]; then
        echo "Conda environment creation failed."
        exit
    fi

    echo "Conda environment created!"
}

# Function to activate conda and run script
activate_conda() {
    echo "Trying to activate Conda from: \"$CONDA_ROOT_PREFIX/bin/conda\""
    if [[ ! -f "$CONDA_ROOT_PREFIX/bin/conda" ]]; then
        echo "Conda activation script not found! Please check your environment and try running the script again."
        rm "$ENV_FLAG"
        exit
    fi

    source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh"
    conda activate "$INSTALL_ENV_DIR"
    if [[ $? -ne 0 ]]; then
        echo "Failed to activate the conda environment. Exiting..."
        exit
    fi

    echo "Conda activated successfully."
    python "$HOME_DIR/one_click.py" --conda-env-path "$INSTALL_ENV_DIR" "$@"
}

activate_conda
