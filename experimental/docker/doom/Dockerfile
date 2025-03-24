# Build stage
FROM ubuntu:22.04 AS builder

# Install build dependencies only
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    automake \
    autoconf \
    autoconf-archive \
    autotools-dev \
    libtool \
    git \
    pkg-config \
    libsdl2-dev \
    libsdl2-net-dev \
    libsdl2-mixer-dev \
    libmicrohttpd-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up build working directory
WORKDIR /build

# Copy only source code and build files
COPY src/ /build/src/
COPY data/ /build/data/
COPY opl/ /build/opl/
COPY pcsound/ /build/pcsound/
COPY textscreen/ /build/textscreen/
COPY autogen.sh /build/
COPY configure.ac /build/
COPY Makefile.am /build/

# Build RESTful DOOM
RUN ./autogen.sh && \
    ./configure && \
    make

# Runtime stage
FROM ubuntu:22.04

# Install only runtime dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libsdl2-2.0-0 \
    libsdl2-net-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libmicrohttpd12 \
    x11-utils \
    curl \
    net-tools \
    psmisc \
    socat \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for running DOOM
RUN useradd -m -s /bin/bash doomuser && \
    mkdir -p /tmp/.X11-unix && \
    chmod 1777 /tmp/.X11-unix

# Set up runtime working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/wad /app/www && \
    chown -R doomuser:doomuser /app

# Copy the built executable and data from builder stage
RUN mkdir -p /app/src
COPY --from=builder /build/src/restful-doom /app/src/
COPY --from=builder /build/data /app/data/

# Create a placeholder WAD file (will be overwritten by volume mount)
RUN touch /app/wad/placeholder.wad && \
    chown doomuser:doomuser /app/wad/placeholder.wad

# Set environment variables
ENV DISPLAY=:1
ENV SDL_VIDEODRIVER=x11
ENV SDL_AUDIODRIVER=dummy
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV API_PORT=8000
ENV HOME=/home/doomuser
ENV USER=doomuser

# Expose the API port
EXPOSE 8000

# Copy web UI files and entrypoint script
COPY experimental/docker/web-ui/ /app/www/
COPY experimental/docker/doom/entrypoint.sh /app/

# Set permissions and ownership
RUN chmod +x /app/entrypoint.sh && \
    chown -R doomuser:doomuser /app

# Switch to non-root user
USER doomuser

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
