#!/bin/bash

# Name of the virtual environment
VENV_NAME="venv"

# Remove the existing virtual environment if it exists
if [ -d "$VENV_NAME" ]; then
    echo "Removing existing virtual environment $VENV_NAME..."
    rm -rf $VENV_NAME
fi

# Check if python3-venv is installed
if ! dpkg -s python3-venv &> /dev/null; then
    echo "python3-venv package is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install python3-venv
fi

# Create the virtual environment
python3 -m venv $VENV_NAME
echo "Virtual environment $VENV_NAME created."

# Activate the virtual environment
source $VENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Clear the pip cache to ensure fresh downloads
pip cache purge

# Install TensorFlow
pip install tensorflow

# Install NumPy
pip install numpy

# Install Pandas
pip install pandas

# Install google-api-python-client
pip install google-api-python-client

# Install google-auth-oauthlib
pip install google-auth-oauthlib

# Install google-auth-httplib2
pip install google-auth-httplib2

echo "Dependencies installed in virtual environment $VENV_NAME."

# Deactivate the virtual environment
deactivate