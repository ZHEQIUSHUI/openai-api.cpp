#!/bin/bash
# Local CI test script

set -e

echo "=== Local CI Test ==="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 passed${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

# Create build directory
mkdir -p build
cd build

# Configure
echo ">>> Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENAI_API_BUILD_TESTS=ON \
    -DOPENAI_API_BUILD_EXAMPLES=ON
print_status "Configure"

# Build
echo ">>> Building..."
cmake --build . -j$(nproc)
print_status "Build"

# Test
echo ">>> Running tests..."
ctest --output-on-failure --timeout 300
print_status "Test"

cd ..

echo ""
echo "=== All CI checks passed! ==="
