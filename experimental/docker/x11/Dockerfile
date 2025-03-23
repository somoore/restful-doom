FROM ubuntu:20.04

# Install X11 server and window manager
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    xvfb \
    fluxbox \
    x11-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create directory for X11 socket
RUN mkdir -p /tmp/.X11-unix && \
    chmod 1777 /tmp/.X11-unix

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set display environment variable
ENV DISPLAY=:1

# Run as root (needed for X11 socket permissions)
# No need for USER directive as root is default

# Expose ports if needed for X11 forwarding
EXPOSE 6000

# Start Xvfb and window manager
ENTRYPOINT ["/entrypoint.sh"]
