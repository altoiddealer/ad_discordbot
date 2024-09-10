#!/bin/bash

# Go up one directory level
cd ..
if [ $? -ne 0 ]; then
    echo "Failed to navigate to the parent directory."
    exit 1
fi

# Set the working directory
WORKING_DIR=$(pwd)

# Execute the existing Miniconda activation script

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

# Check if bot.py is in the root directory
if [ -f "bot.py" ]; then
    echo "bot.py found in the root directory."
    echo "bot.py is now expected to be in the ad_discordbot directory."
    echo "Please move bot.py to the ad_discordbot directory and try again."
    exit 1
fi

# Read command flags from CMD_FLAGS.txt
CMD_FLAGS_FILE="$WORKING_DIR/ad_discordbot/CMD_FLAGS.txt"
CMD_FLAGS=""  # Initialize CMD_FLAGS as an empty string

if [ ! -f "$CMD_FLAGS_FILE" ]; then
    echo "CMD_FLAGS.txt is not found."
else
    # Read each line from CMD_FLAGS.txt, skipping comments
    CMD_FLAGS=$(grep -v '^#' "$CMD_FLAGS_FILE" | xargs)  # xargs trims any extra whitespace
    if [ -z "$CMD_FLAGS" ]; then
        echo "CMD_FLAGS.txt is empty or only contains comments."
    fi
fi

# Launch ad_discordbot with flags from CMD_FLAGS.txt (even if CMD_FLAGS is empty)
BOT_SCRIPT="$WORKING_DIR/ad_discordbot/bot.py"
if [ ! -f "$BOT_SCRIPT" ]; then
    echo "bot.py not found in the ad_discordbot directory."
    exit 1
fi

# Run bot.py with or without CMD_FLAGS (if CMD_FLAGS is empty, it won't pass any extra arguments)
python "$BOT_SCRIPT" $CMD_FLAGS
if [ $? -ne 0 ]; then
    echo "bot.py execution failed"
    exit 1
fi