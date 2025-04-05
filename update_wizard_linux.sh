#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$(pwd)" =~ " " ]]; then echo "This script relies on Miniconda which can not be silently installed under a path with spaces." && exit; fi

# deactivate existing conda envs as needed to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null

# root directories
HOME_DIR="$(pwd)"
PARENT_DIR="$(realpath "$HOME_DIR/..")"

# configs
CONDA_HOME="$HOME_DIR/installer_files/conda"
ENV_HOME="$HOME_DIR/installer_files/env"
CONDA_PARENT="$PARENT_DIR/installer_files/conda"
ENV_PARENT="$PARENT_DIR/installer_files/env"
CONDA_ROOT_PREFIX="$CONDA_HOME"
INSTALL_ENV_DIR="$ENV_HOME"

# Read user_env.txt into ENV_FLAG
ENV_FLAG=""
if [[ -f "$HOME_DIR/internal/user_env.txt" ]]; then
    ENV_FLAG=$(<"$HOME_DIR/internal/user_env.txt")
fi

# if TGWUI integration flag, run from its env
if [[ "$ENV_FLAG" == "$ENV_PARENT" ]]; then
    CONDA_ROOT_PREFIX="$CONDA_PARENT"
    INSTALL_ENV_DIR="$ENV_PARENT"
fi

# environment isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

# activate installer env
source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized'
conda activate "$INSTALL_ENV_DIR"

# update installer env
python one_click.py --update-wizard-linux --conda-env-path "$INSTALL_ENV_DIR" && echo -e "\nHave a great day!"
