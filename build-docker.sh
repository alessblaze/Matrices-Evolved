#!/bin/bash
set -e

echo "Building matrices_evolved wheels for multiple architectures..."

mkdir -p wheelhouse
rm -rf wheelhouse/*
# Build for amd64
echo "Building for amd64..."
docker buildx build -t matrices-evolved-builder-amd64 --platform linux/amd64 --load .
docker run --platform linux/amd64 --rm -v $(pwd):/host -w /src matrices-evolved-builder-amd64 bash -c "
    rsync -av --exclude=build --exclude=dist /host/ . && 
    python3.11 -m build --wheel && 
    auditwheel repair dist/*.whl -w wheelhouse/ && 
    cp wheelhouse/* /host/wheelhouse/
"

# Build for arm64
echo "Building for arm64..."
docker buildx build -t matrices-evolved-builder-arm64 --platform linux/arm64 --load .
docker run --platform linux/arm64 --rm -v $(pwd):/host -w /src matrices-evolved-builder-arm64 bash -c "
    rsync -av --exclude=build --exclude=dist /host/ . && 
    python3.11 -m build --wheel && 
    auditwheel repair dist/*.whl -w wheelhouse/ && 
    cp wheelhouse/* /host/wheelhouse/
"

echo "All wheels built:"
ls -la wheelhouse/
python3 -m twine upload wheelhouse/*
