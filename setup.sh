#!/bin/bash
# Setup script for FSD Web Demo

set -e

echo "=== FSD Web Demo Setup ==="

# Create virtual environment for backend
echo "Setting up backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Backend dependencies installed"

# Install frontend dependencies
echo "Setting up frontend..."
cd ../frontend
npm install
echo "Frontend dependencies installed"

# Convert models if torch is available
echo ""
echo "Checking for model conversion..."
cd ..
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch found. Converting models..."
    python3 scripts/convert_models.py
else
    echo "PyTorch not installed. Skipping model conversion."
    echo "To convert models later: pip install torch onnxruntime && python3 scripts/convert_models.py"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the demo:"
echo "  Terminal 1: cd backend && source venv/bin/activate && uvicorn main:app --reload"
echo "  Terminal 2: cd frontend && npm run dev"
echo ""
echo "Open http://localhost:5173 in your browser"
