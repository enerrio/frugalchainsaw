#!/usr/bin/env bash

# Usage: ./setup.sh <repo_url> <branch>

REPO_URL=${1:-"git@github.com:enerrio/frugalchainsaw.git"}
BRANCH=${2:-"main"}
ENV_NAME="chainsaw"
PYTHON_VERSION="3.12"

# 1. Clone the repo
git clone -b $BRANCH $REPO_URL
cd frugalchainsaw

# 2. Create and activate conda environment
conda create -y -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# 3. Install dependencies
conda env update --file environment.yml --name $ENV_NAME

# 4. Run your data prep script
python scripts/prep_data.py