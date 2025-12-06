#!/bin/bash
# Quick start script for HVSR Progressive Layer Stripping GUI

echo "=========================================="
echo "HVSR Progressive Layer Stripping GUI"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import PySide6" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: PySide6 not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application
echo "Starting GUI application..."
python3 app.py

echo ""
echo "Application closed"
