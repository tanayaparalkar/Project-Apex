#!/usr/bin/env bash
# build.sh – Render Build Script for EventOracle
# Installs Python deps, then builds the React frontend into frontend/dist/

set -o errexit  # exit on error

echo "════════════════════════════════════════════"
echo "  EventOracle – Render Build"
echo "════════════════════════════════════════════"

# 1. Install Python dependencies
echo "→ Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Install Node.js dependencies and build React frontend
echo "→ Installing Node.js dependencies..."
cd frontend
npm install

echo "→ Building React frontend..."
npm run build

echo "→ Frontend built to frontend/dist/"
cd ..

echo "════════════════════════════════════════════"
echo "  Build Complete!"
echo "════════════════════════════════════════════"
