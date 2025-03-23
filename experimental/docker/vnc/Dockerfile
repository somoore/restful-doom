FROM ubuntu:20.04

# Install VNC server and dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    x11vnc \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up VNC password
RUN mkdir -p /root/.vnc && \
    x11vnc -storepasswd password /root/.vnc/passwd

# Set environment variables
ENV DISPLAY=:1
ENV VNC_PORT=5900

# Expose VNC port
EXPOSE 5900

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
