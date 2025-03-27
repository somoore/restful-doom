# Multi-stage build for X11 server
FROM ubuntu:22.04 AS builder

# Install X11 server and window manager (build dependencies if any)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    xvfb \
    fluxbox \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Final stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    xvfb \
    fluxbox \
    x11-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    sudo \
    && rm -rf /var/lib/apt/lists/* && \
    # Create x11user but note we still need to run as root for some operations
    groupadd -r x11group && \
    useradd -r -g x11group -d /home/x11user -m x11user && \
    # Add x11user to video and input groups if they exist
    groupadd -r video 2>/dev/null || true && \
    groupadd -r input 2>/dev/null || true && \
    usermod -a -G video,input x11user 2>/dev/null || true && \
    # Give x11user sudo access to specific X11 commands
    echo "x11user ALL=(root) NOPASSWD: /usr/bin/Xvfb, /usr/bin/fluxbox" > /etc/sudoers.d/x11user && \
    chmod 0440 /etc/sudoers.d/x11user

# Configure Fluxbox to remove window decorations and disable toolbar
RUN mkdir -p /home/x11user/.fluxbox && \
    echo '[app] (name=doom)' > /home/x11user/.fluxbox/apps && \
    echo '  [Deco] {NONE}' >> /home/x11user/.fluxbox/apps && \
    echo '  [Layer] {Above}' >> /home/x11user/.fluxbox/apps && \
    echo '  [Fullscreen] {yes}' >> /home/x11user/.fluxbox/apps && \
    echo '[end]' >> /home/x11user/.fluxbox/apps && \
    echo '[app] (title=The Ultimate DOOM)' >> /home/x11user/.fluxbox/apps && \
    echo '  [Deco] {NONE}' >> /home/x11user/.fluxbox/apps && \
    echo '  [Layer] {Above}' >> /home/x11user/.fluxbox/apps && \
    echo '  [Fullscreen] {yes}' >> /home/x11user/.fluxbox/apps && \
    echo '[end]' >> /home/x11user/.fluxbox/apps && \
    echo 'session.screen0.toolbar.visible: false' > /home/x11user/.fluxbox/init && \
    chown -R x11user:x11group /home/x11user/.fluxbox

# Create directory for X11 socket with proper permissions
RUN mkdir -p /tmp/.X11-unix && \
    chmod 1777 /tmp/.X11-unix && \
    chown root:root /tmp/.X11-unix

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    chown root:x11group /entrypoint.sh

# Set display environment variable
ENV DISPLAY=:1
ENV HOME=/home/x11user
ENV USER=x11user

# Create runtime directory for X11user
RUN mkdir -p /run/user/$(id -u x11user) && \
    chown x11user:x11group /run/user/$(id -u x11user)

# Expose ports if needed for X11 forwarding
EXPOSE 6000

# Note: X11 server still needs to run as root for hardware access
# However, the entrypoint can drop privileges where possible
# USER x11user - Uncomment if entrypoint.sh is modified to handle privilege escalation

# Start Xvfb and window manager
ENTRYPOINT ["/entrypoint.sh"]
