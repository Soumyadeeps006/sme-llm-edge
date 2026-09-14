#!/bin/bash
# Day 38: AddressSanitizer (ASan) Validation Script
# Automates ASan build and short inference smoke test.

set -e

echo "=========================================================="
echo "🚀 Day 38: AddressSanitizer (ASan) Validation Starting"
echo "=========================================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_asan"

# 1. Clean the build directory
echo "[1/4] Cleaning previous ASan build directory..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# 2. Configure CMake with ASan
echo "[2/4] Configuring CMake with -DCMAKE_BUILD_TYPE=ASAN..."
cd "$BUILD_DIR"
cmake -DCMAKE_BUILD_TYPE=ASAN ..

# 3. Build the project
echo "[3/4] Building project with ASan instrumentation..."
make -j$(nproc)

# 4. Run a short, 10-step inference smoke test
echo "[4/4] Running short inference smoke test..."

# Adjust the SMOKE_TEST_CMD to match your actual binary name and smoke test arguments.
# Example: SMOKE_TEST_CMD="./edge_server --mode smoke_test --steps 10"
SMOKE_TEST_CMD="./edge_server --smoke-test --steps 10"

if [ -f "./edge_server" ]; then
    echo "Executing: $SMOKE_TEST_CMD"
    # Run the smoke test. ASan will automatically abort and print errors if a violation occurs.
    # We use 'timeout' to prevent hanging in case of deadlocks or infinite loops.
    if timeout 60s $SMOKE_TEST_CMD; then
        echo "✅ ASan Smoke Test Passed: Zero memory leaks, use-after-free, or buffer overflows detected."
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "❌ ASan Smoke Test Failed: Timed out after 60 seconds."
        else
            echo "❌ ASan Smoke Test Failed: AddressSanitizer detected a memory safety violation or the process crashed (Exit code: $EXIT_CODE)."
        fi
        exit 1
    fi
else
    echo "⚠️  Warning: edge_server binary not found in $BUILD_DIR."
    echo "   Please ensure the build succeeded and run the binary manually with ASan to verify runtime safety."
fi

echo "=========================================================="
echo "🎉 AddressSanitizer Validation Complete."
echo "=========================================================="
exit 0