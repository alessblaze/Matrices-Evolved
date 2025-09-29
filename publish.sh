#!/bin/bash
set -e

echo "Building and publishing matrices_evolved..."

# Clean previous builds
rm -rf dist/ build/ wheelhouse/

# Build wheel
python3.11 -m build --wheel

# Install auditwheel if not present
python3.11 -m pip install auditwheel twine

# Repair wheel for manylinux compatibility
python3.11 -m auditwheel repair dist/*.whl --plat manylinux_2_34_x86_64

# Upload to PyPI
python3.11 -m twine upload wheelhouse/*

echo "Successfully published to PyPI!"
