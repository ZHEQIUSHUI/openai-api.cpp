# Multi-stage build for OpenAI API Server

# Build stage
FROM ubuntu:24.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
WORKDIR /src
COPY . .

# Build
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENAI_API_BUILD_TESTS=OFF \
    -DOPENAI_API_BUILD_EXAMPLES=OFF \
    && cmake --build build --parallel

# Runtime stage
FROM ubuntu:24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /src/build/openai_api_server /usr/local/bin/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/v1/models || exit 1

# Entry point
ENTRYPOINT ["openai_api_server"]
CMD ["8080"]
