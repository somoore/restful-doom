# Build stage for noVNC setup
FROM ubuntu:22.04 AS builder

# Install noVNC for configuration setup only
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    novnc \
    && rm -rf /var/lib/apt/lists/*

# Create directory for custom noVNC files
RUN mkdir -p /usr/share/novnc/custom

# Copy custom CSS for fixing display aspect ratio
COPY custom/app.css /usr/share/novnc/custom/app.css

# Add custom CSS reference to noVNC index page
RUN sed -i '/<\/head>/i <link rel="stylesheet" href="custom/app.css">' /usr/share/novnc/vnc.html

# Runtime stage with minimal components
FROM ubuntu:22.04

# Install only runtime dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    novnc \
    websockify \
    python3 \
    python3-pip \
    net-tools \
    netcat \
    && rm -rf /var/lib/apt/lists/* \
    # Create non-root user for improved security
    && groupadd -r novncuser && useradd -r -g novncuser -d /home/novncuser -s /bin/bash novncuser \
    && mkdir -p /home/novncuser \
    && mkdir -p /usr/share/novnc/custom

# Copy the modified noVNC files from the builder
COPY --from=builder /usr/share/novnc/vnc.html /usr/share/novnc/
COPY --from=builder /usr/share/novnc/custom/app.css /usr/share/novnc/custom/

# Set environment variables
ENV VNC_HOST=vnc-server
ENV VNC_PORT=5900
ENV NOVNC_PORT=6080
ENV HOME=/home/novncuser
ENV USER=novncuser

# Expose noVNC port
EXPOSE 6080

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    chown -R novncuser:novncuser /home/novncuser && \
    chown -R novncuser:novncuser /usr/share/novnc && \
    chown novncuser:novncuser /entrypoint.sh

# Switch to non-root user for better security
USER novncuser
WORKDIR /home/novncuser

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
