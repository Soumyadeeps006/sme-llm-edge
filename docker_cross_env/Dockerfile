# ==============================================================================
# Stage 1: Builder Environment
# ==============================================================================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies and cross-compilation packages
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
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rustup and configure the aarch64 cross-compile target
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup target add aarch64-unknown-linux-gnu

# Configure cargo for cross compilation
RUN mkdir -p /root/.cargo && \
    echo '[target.aarch64-unknown-linux-gnu]' > /root/.cargo/config.toml && \
    echo 'linker = "aarch64-linux-gnu-gcc"' >> /root/.cargo/config.toml

WORKDIR /workspace

# Copy source code
COPY . .

# Build Rust kernels
RUN cd src/sme_kernel/rust_sme && \
    cargo build --release --target aarch64-unknown-linux-gnu

# Build C++ server
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_TOOLCHAIN_FILE=../aarch64-toolchain.cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -G Ninja .. && \
    ninja -j$(nproc)

# ==============================================================================
# Stage 2: Minimal Runtime Environment
# ==============================================================================
# Using arm64v8 Ubuntu for native execution on the edge device. 
# If running via qemu on x86, standard ubuntu:22.04 works but arm64v8 is optimal for edge.
FROM arm64v8/ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install ONLY minimal runtime dependencies required by the binary
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libstdc++6 \
    libomp5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy ONLY the compiled binary from the builder stage
COPY --from=builder /workspace/build/edge_server /app/edge_server

# Day 39: Strip debug symbols to drastically reduce binary size for edge deployment
RUN strip --strip-all /app/edge_server

# Expose the HTTP streaming port
EXPOSE 8080

# Default command to run the server
CMD ["/app/edge_server"]