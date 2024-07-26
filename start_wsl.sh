#!/bin/bash

# Go up one directory level
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

# Check if bot.py is in the root directory
if [ -f "bot.py" ]; then
    echo "bot.py found in the root directory."
    echo "bot.py is now expected to be in the ad_discordbot directory."
    echo "Please move bot.py to the ad_discordbot directory and try again."
    read -p "Press any key to continue..."
    exit 1
fi

# Read command flags from CMD_FLAGS.txt
CMD_FLAGS=""
if [ ! -f "ad_discordbot/CMD_FLAGS.txt" ]; then
    echo "CMD_FLAGS.txt is not found."
else
    # Read each line from CMD_FLAGS.txt, skipping comments
    while IFS= read -r line; do
        if [[ ! "$line" =~ ^\s*# ]]; then
            CMD_FLAGS="$line"
            break
        fi
    done < "ad_discordbot/CMD_FLAGS.txt"
    
    if [ -z "$CMD_FLAGS" ]; then
        echo "CMD_FLAGS.txt is empty."
    fi
fi

# Launch ad_discordbot with flags from CMD_FLAGS.txt
python ad_discordbot/bot.py $CMD_FLAGS
if [ $? -ne 0 ]; then
    echo "bot.py execution failed"
    read -p "Press any key to continue..."
    exit 1
fi

read -p "Press any key to continue..."
