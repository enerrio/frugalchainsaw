#!/usr/bin/env bash

# Usage: ./setup.sh <repo_url> <branch>

REPO_URL=${1:-"git@github.com:enerrio/frugalchainsaw.git"}
BRANCH=${2:-"main"}
VENV_PATH=".venv"

# 0. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone the repo
git clone -b $BRANCH $REPO_URL
cd frugalchainsaw

# 2. Create virtual environment and install dependencies (exclude dev dependencies)
uv sync --no-dev

# 3. Run your data prep script
uv run scripts/prep_data.py --normalization_mode global