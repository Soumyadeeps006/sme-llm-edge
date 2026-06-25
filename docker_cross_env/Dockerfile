FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and cross-compilation packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    cmake \
    ninja-build \
    curl \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Rustup and configure the aarch64 cross-compile target
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup target add aarch64-unknown-linux-gnu

# Configure cargo for cross compilation
RUN mkdir -p /root/.cargo && \
    echo '[target.aarch64-unknown-linux-gnu]' > /root/.cargo/config.toml && \
    echo 'linker = "aarch64-linux-gnu-gcc"' >> /root/.cargo/config.toml

# Write CMake toolchain file inside the container
RUN echo 'set(CMAKE_SYSTEM_NAME Linux)' > /toolchain.cmake && \
    echo 'set(CMAKE_SYSTEM_PROCESSOR aarch64)' >> /toolchain.cmake && \
    echo 'set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)' >> /toolchain.cmake && \
    echo 'set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)' >> /toolchain.cmake && \
    echo 'set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)' >> /toolchain.cmake && \
    echo 'set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)' >> /toolchain.cmake && \
    echo 'set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)' >> /toolchain.cmake && \
    echo 'set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)' >> /toolchain.cmake

WORKDIR /workspace

CMD ["/bin/bash"]
