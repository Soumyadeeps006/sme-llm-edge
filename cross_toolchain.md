# Cross-Compilation Toolchain Setup

To compile this project for an ARM Neoverse V3 development board with SME/SVE2 capabilities from an x86_64 host (or container), use the cross-compilation toolchain setup detailed below.

## Prerequisites

On your Ubuntu 22.04 LTS host, install the cross-compilation gcc/g++ toolchain:

```bash
sudo apt-get update
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu cmake ninja-build git
```

Install Rust and configure the target architecture:

```bash
rustup target add aarch64-unknown-linux-gnu
```

## Compilation using CMake Toolchain File

Create a CMake toolchain file, e.g., `aarch64-toolchain.cmake`:

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

Build the project:

```bash
mkdir build_arm64 && cd build_arm64
cmake -DCMAKE_TOOLCHAIN_FILE=../aarch64-toolchain.cmake ..
make -j$(nproc)
```

This will produce the binary `server/llm_server` compiled for `aarch64` with SME/SVE2 instructions enabled.
