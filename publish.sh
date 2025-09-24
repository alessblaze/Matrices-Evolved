#!/bin/bash
set -e

echo "Building and publishing matrices_evolved..."

# Clean previous builds
rm -rf dist/ build/ wheelhouse/

# Build wheel
python3 -m build --wheel

# Install auditwheel if not present
pip install auditwheel twine

# Repair wheel for manylinux compatibility
auditwheel repair dist/*.whl --plat manylinux_2_34_x86_64

# Upload to PyPI
python3 -m twine upload wheelhouse/*

echo "Successfully published to PyPI!"