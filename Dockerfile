FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    python3.11 \
    python3-pip \
    python3.11-dev \
    python3.11-venv \
    cmake \
    build-essential \
    git \
    curl \
    rsync \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Install Clang 20
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y clang-20 lld-20 && \
    rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    /root/.cargo/bin/rustup default stable
ENV PATH="/root/.cargo/bin:$PATH"

#RUN ln -s /usr/bin/python3.11 /usr/bin/python3 && \
#    python3 -m pip install --upgrade pip setuptools
# Install Python build tools
RUN python3.11 -m pip install build auditwheel

# Set environment
ENV CC=clang-20
ENV CXX=clang++-20
ENV CARGO_EXECUTABLE=/root/.cargo/bin/cargo

WORKDIR /src

# Default command
CMD ["/bin/bash"]