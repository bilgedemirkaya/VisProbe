#!/bin/bash
# Setup script for VisProbe documentation

echo "🚀 Setting up VisProbe documentation environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install documentation dependencies
echo "📚 Installing documentation dependencies..."
pip install -r requirements-docs.txt

echo ""
echo "✨ Setup complete! Your documentation environment is ready."
echo ""
echo "To serve the documentation locally:"
echo "  1. source venv/bin/activate"
echo "  2. mkdocs serve"
echo ""
echo "To deploy to GitHub Pages:"
echo "  1. source venv/bin/activate"
echo "  2. mkdocs gh-deploy"
echo ""
