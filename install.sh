#!/bin/bash

# Navigate to the frontend folder
echo "Installing React dependencies..."
cd frontend || exit 1

# Ensure Node.js and npm are installed (only for Linux)
if ! command -v node &>/dev/null; then
    echo "Node.js is not installed. Installing..."
    sudo apt update && sudo apt install -y nodejs
fi

if ! command -v npm &>/dev/null; then
    echo "npm is not installed. Installing..."
    sudo apt install -y npm
fi

# Install React dependencies
echo "Installing React dependencies..."
npm install
echo "---FrontEnd installed---"

# Navigate back to the project root
cd ..

# Navigate to the backend folder
cd backend || exit 1

echo "Installing Python..."
# Ensure Python and pip are installed (only for Linux)
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Installing..."
    sudo apt update && sudo apt install -y python3
fi

echo "Installing PiP..."
if ! command -v pip &>/dev/null; then
    echo "pip is not installed. Installing..."
    sudo apt install -y python3-pip
fi

echo "Installing Python Requirements..."
# Install Python dependencies
pip install --no-cache-dir -r requirements.txt


# Run model construction if model.safetensors is missing
MODEL_PATH="ml_model/complete_model/model.safetensors"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file not found. Running construction script..."
    python ml_model/construction.py
else
    echo "Model already exists. Skipping construction."
fi

echo "---BackEnd installed---"
echo "---Installation is fully complete!---"
