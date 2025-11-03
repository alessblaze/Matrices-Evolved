FROM debian:bookworm
# Install dependencies
ARG TARGETARCH

RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    cmake \
    build-essential \
    git \
    curl \
    rsync \
    patchelf \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Clang 20
RUN wget -O /usr/share/keyrings/llvm-snapshot.gpg.key https://apt.llvm.org/llvm-snapshot.gpg.key && \
    echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg.key] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-20 main" | tee /etc/apt/sources.list.d/llvm.list && \
    apt-get update && \
    apt-get install -y clang-20 lld-20

RUN case "$TARGETARCH" in \
      amd64) CONDA_ARCH="x86_64" ;; \
      arm64) CONDA_ARCH="aarch64" ;; \
      386) CONDA_ARCH="i686" ;; \
      ppc64le) CONDA_ARCH="ppc64le" ;; \
      *) echo "Unsupported architecture: $TARGETARCH"; exit 1 ;; \
    esac && \
    curl -fsSL -o /tmp/miniconda.sh \
    "https://repo.anaconda.com/miniconda/Miniconda3-py313_25.3.1-1-Linux-${CONDA_ARCH}.sh" && \
    bash /tmp/miniconda.sh -b -f -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"
# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    /root/.cargo/bin/rustup default stable
ENV PATH="/root/.cargo/bin:$PATH"

#RUN ln -s /usr/bin/python3.11 /usr/bin/python3 && \
#    python3 -m pip install --upgrade pip setuptools
# Install Python build tools
RUN python3 -m venv /opt/build-env && \
    /opt/build-env/bin/pip install build auditwheel
# Set environment
ENV CC=clang-20
ENV CXX=clang++-20
ENV CARGO_EXECUTABLE=/root/.cargo/bin/cargo

WORKDIR /src

# Default command
CMD ["/bin/bash"]