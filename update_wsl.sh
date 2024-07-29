#!/bin/bash

# Step 1: Perform git pull
git pull
if [ $? -ne 0 ]; then
    echo "Failed to pull from the repository."
    read -p "Press any key to continue..."
    exit $?
fi

# Step 2: Go up one directory level
cd ..
if [ $? -ne 0 ]; then
    echo "Failed to navigate to the parent directory."
    read -p "Press any key to continue..."
    exit $?
fi

# Set the working directory explicitly to the parent directory
WORKING_DIR=$(pwd)

# Change directory to where the script should execute
cd "$WORKING_DIR"
if [ $? -ne 0 ]; then
    echo "Failed to change directory to $WORKING_DIR."
    read -p "Press any key to continue..."
    exit $?
fi

echo "$WORKING_DIR" | grep -q " "
if [ $? -eq 0 ]; then
    echo "This script relies on Miniconda which cannot be silently installed under a path with spaces."
    read -p "Press any key to continue..."
    exit 1
fi

# Ensure conda is initialized
conda init bash
if [ $? -ne 0 ]; then
    echo "Failed to initialize conda."
    read -p "Press any key to continue..."
    exit $?
fi

# Deactivate existing conda envs as needed to avoid conflicts
conda deactivate 2>/dev/null
conda deactivate 2>/dev/null
conda deactivate 2>/dev/null

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
    read -p "Press any key to continue..."
    exit 1
fi

# Activate installer env
. "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh" && conda activate "$INSTALL_ENV_DIR"
if [ $? -ne 0 ]; then
    echo "Failed to activate the conda environment."
    read -p "Press any key to continue..."
    exit 1
fi

# Step 4: Install dependencies
pip install -r ad_discordbot/requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    read -p "Press any key to continue..."
    exit 1
fi

echo "ad_discordbot has been updated."
read -p "Press any key to continue..."
