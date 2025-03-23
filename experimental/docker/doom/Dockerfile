FROM ubuntu:20.04

# Install required packages for building and running DOOM with API
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
    curl \
    libmicrohttpd-dev \
    x11-utils \
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

# Set up working directory
WORKDIR /app

# Create directories
RUN mkdir -p /app/wad

# Copy source code and build files from project root
COPY src/ /app/src/
COPY data/ /app/data/
COPY opl/ /app/opl/
COPY pcsound/ /app/pcsound/
COPY textscreen/ /app/textscreen/
COPY autogen.sh /app/
COPY configure.ac /app/
COPY Makefile.am /app/

# Create WAD and www directories
RUN mkdir -p /app/wad /app/www

# Create a placeholder WAD file (will be overwritten by volume mount)
RUN touch /app/wad/placeholder.wad

# Build RESTful DOOM
RUN ./autogen.sh && \
    ./configure && \
    make

# Set environment variables
ENV DISPLAY=:1
ENV SDL_VIDEODRIVER=x11
ENV SDL_AUDIODRIVER=dummy
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV API_PORT=8000

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
