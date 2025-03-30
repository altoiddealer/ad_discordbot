#!/bin/bash

# detect if build-essential is missing or broken
if ! dpkg-query -W -f'${Status}' "build-essential" 2>/dev/null | grep -q "ok installed"; then
echo "build-essential not found or broken!

A C++ compiler is required to build needed Python packages!
To install one, run cmd_wsl.bat and enter these commands:

sudo apt-get update
sudo apt-get install build-essential
"
read -n1 -p "Continue the installer anyway? [y,n]" EXIT_PROMPT
# only continue if user inputs 'y' else exit
if ! [[ $EXIT_PROMPT == "Y" || $EXIT_PROMPT == "y" ]]; then exit; fi
fi

# deactivate existing conda envs as needed to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null

# Get the current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# root directories
HOME_DIR="$SCRIPT_DIR"
PARENT_DIR="$(dirname "$HOME_DIR")"

# configs
CONDA_HOME="$HOME_DIR/installer_files/conda"
ENV_HOME="$HOME_DIR/installer_files/env"
CONDA_PARENT="$PARENT_DIR/installer_files/conda"
ENV_PARENT="$PARENT_DIR/installer_files/env"

MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/Miniconda3-py311_24.11.1-0-Linux-x86_64.sh"
conda_exists="F"

# Read user_env.txt into ENV_FLAG
ENV_FLAG=""
if [ -f "$HOME_DIR/internal/user_env.txt" ]; then
    ENV_FLAG=$(cat "$HOME_DIR/internal/user_env.txt")
fi

# If env flag exists, assign paths and activate
if [ "$ENV_FLAG" == "$ENV_HOME" ]; then
    CONDA_ROOT_PREFIX="$CONDA_HOME"
    INSTALL_ENV_DIR="$ENV_HOME"
    source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh"
    conda activate "$INSTALL_ENV_DIR"
    exit
fi
if [ "$ENV_FLAG" == "$ENV_PARENT" ]; then
    CONDA_ROOT_PREFIX="$CONDA_PARENT"
    INSTALL_ENV_DIR="$ENV_PARENT"
    source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh"
    conda activate "$INSTALL_ENV_DIR"
    exit
fi

# Check if conda exists
if [[ -x "$CONDA_ROOT_PREFIX/bin/conda" ]]; then
    conda_exists="T"
fi

# (if necessary) install git and conda into a contained environment
# download miniconda
if [ "$conda_exists" == "F" ]; then
    echo "Downloading Miniconda from $MINICONDA_DOWNLOAD_URL to $HOME_DIR/installer_files/miniconda_installer.sh"
    curl -L "$MINICONDA_DOWNLOAD_URL" > "$HOME_DIR/installer_files/miniconda_installer.sh"
    chmod u+x "$HOME_DIR/installer_files/miniconda_installer.sh"
    bash "$HOME_DIR/installer_files/miniconda_installer.sh" -b -p "$CONDA_ROOT_PREFIX"
    # test the conda binary
    echo "Miniconda version:"
    "$CONDA_ROOT_PREFIX/bin/conda" --version
    # delete the Miniconda installer
    rm "$HOME_DIR/installer_files/miniconda_installer.sh"
fi

# Create the installer environment if it doesn't exist
if [ ! -e "$INSTALL_ENV_DIR" ]; then
    "$CONDA_ROOT_PREFIX/bin/conda" create -y -k --prefix "$INSTALL_ENV_DIR" python=3.11 git
fi

# Check if the conda environment was actually created
if [ ! -e "$INSTALL_ENV_DIR/bin/python" ]; then
    echo "Conda environment is empty."
    exit
fi

# Activate the installer environment
source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh"
conda activate "$INSTALL_ENV_DIR"

# Navigate to the installation directory
pushd "$INSTALL_DIR" 1> /dev/null || exit

# Copy CMD_FLAGS.txt to the install dir to allow edits within Windows
if [[ ! "$INSTALL_INPLACE" == "1" ]]; then
    if [ ! -f "./wsl.sh" ]; then
        git pull || exit
        [ -f "../adbot.py" ] && mv "../adbot.py" "../adbot-old.py"
    fi
    if [ -f "$(dirs +1)/CMD_FLAGS.txt" ] && [ -f "./CMD_FLAGS.txt" ]; then
        cp -u "$(dirs +1)/CMD_FLAGS.txt" "$INSTALL_DIR"
    fi
fi

# Setup installer environment and update if necessary
case "$1" in
("update-wizard") python one_click.py --update-wizard-wsl;;
(*) python one_click.py "$@";;
esac
