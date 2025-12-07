#!/bin/bash

# RAG Application Startup Script
# This script sets the correct PYTHONPATH and starts the Streamlit app

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to include the project root
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Load environment variables from .env file
if [ -f "${SCRIPT_DIR}/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' "${SCRIPT_DIR}/.env" | xargs)
else
    echo "Warning: .env file not found. Please create one with your OPENAI_API_KEY."
fi

# Start the Streamlit application
streamlit run src/presentation/ui/app.py
