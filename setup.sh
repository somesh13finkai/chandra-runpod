#!/bin/bash
set -e

# Update apt-get and install required system packages
apt-get update
apt-get install -y tmux python3-pip

# Upgrade pip
python3 -m pip install --upgrade pip

# Install the Python requirements
python3 -m pip install -r requirements.txt

# Prompt for Hugging Face token
hf auth login