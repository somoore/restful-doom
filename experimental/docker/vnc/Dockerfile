# Multi-stage build for VNC server
FROM ubuntu:22.04 AS builder

# Create a .vnc directory and set up VNC password in the builder stage
RUN mkdir -p /.vnc && \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y x11vnc && \
    x11vnc -storepasswd password /.vnc/passwd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Final stage with minimal dependencies
FROM ubuntu:22.04

# Create non-root user
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    x11vnc \
    x11-utils \
    && rm -rf /var/lib/apt/lists/* && \
    groupadd -r vncuser && useradd -r -g vncuser -d /home/vncuser -m vncuser && \
    mkdir -p /home/vncuser/.vnc && \
    mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Copy VNC password from builder stage
COPY --from=builder /.vnc/passwd /home/vncuser/.vnc/passwd

# Set proper permissions
RUN chown -R vncuser:vncuser /home/vncuser

# Set environment variables
ENV DISPLAY=:1
ENV VNC_PORT=5900
ENV HOME=/home/vncuser
ENV USER=vncuser

# Expose VNC port
EXPOSE 5900

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    chown vncuser:vncuser /entrypoint.sh

# Switch to non-root user
USER vncuser
WORKDIR /home/vncuser

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
