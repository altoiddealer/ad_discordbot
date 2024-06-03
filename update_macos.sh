#!/bin/bash

# Step 1: Perform git pull
git pull
if [ $? -ne 0 ]; then
    echo "Failed to pull from the repository."
    exit 1
fi

# Step 2: Go up one directory level
cd ..
if [ $? -ne 0 ]; then
    echo "Failed to navigate to the parent directory."
    exit 1
fi

# Set the working directory
WORKING_DIR=$(pwd)

# Step 3: Execute the existing Miniconda activation script

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$(pwd)" =~ " " ]]; then
    echo "This script relies on Miniconda which can not be silently installed under a path with spaces."
    exit 1
fi

# Deactivate existing conda envs as needed to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null

# Config
CONDA_ROOT_PREFIX="$WORKING_DIR/installer_files/conda"
INSTALL_ENV_DIR="$WORKING_DIR/installer_files/env"

# Environment isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export CUDA_PATH="$INSTALL_ENV_DIR"
export CUDA_HOME="$CUDA_PATH"

# Check if conda.sh exists
if [ ! -f "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh" ]; then
    echo "Conda.sh not found at $CONDA_ROOT_PREFIX/etc/profile.d/conda.sh"
    exit 1
fi

# Activate installer env
source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh"
conda activate "$INSTALL_ENV_DIR"
if [ $? -ne 0 ]; then
    echo "Failed to activate the conda environment."
    exit 1
fi

# Step 4: Install dependencies
pip install -r ad_discordbot/requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
fi

echo "ad_discordbot has been updated."
