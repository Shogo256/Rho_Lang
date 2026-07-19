FROM ubuntu:24.04

LABEL maintainer="Escoffier1156 <escoffier.office1156@gmail.com>"
LABEL description="Unified build and runtime environment for ρ (RHO) Language Compiler v1.0"

# Install build dependencies, LLVM, Clang, Rust, and Python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    clang \
    llvm \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Workspace setup
WORKDIR /app
COPY . /app

# Build compiler
RUN cargo build --release

CMD ["cargo", "test"]
