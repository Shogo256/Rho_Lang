# Formula for ρ (RHO) Language Compiler Development Environment
FROM ubuntu:24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install core development tools and dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    python3-venv \
    clang \
    llvm \
    lld \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set up working directory
WORKDIR /workspace

# Copy the repository source into the container
COPY . /workspace

# Build RHO compiler in release mode
RUN cargo build --release

# Expose compiled binary to system PATH
ENV PATH="/workspace/target/release:${PATH}"

# Default shell
CMD ["/bin/bash"]
