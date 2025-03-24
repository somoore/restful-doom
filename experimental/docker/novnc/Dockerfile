FROM ubuntu:20.04

# Install noVNC and dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    novnc \
    websockify \
    python3 \
    python3-pip \
    net-tools \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV VNC_HOST=vnc-server
ENV VNC_PORT=5900
ENV NOVNC_PORT=6080

# Expose noVNC port
EXPOSE 6080

# Copy entrypoint script and custom files
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create directory for custom noVNC files
RUN mkdir -p /usr/share/novnc/custom

# Copy custom CSS for fixing display aspect ratio
COPY custom/app.css /usr/share/novnc/custom/app.css

# Create non-root user for improved security
RUN groupadd -r novncuser && useradd -r -g novncuser -d /home/novncuser -s /bin/bash novncuser \
    && mkdir -p /home/novncuser \
    && chown -R novncuser:novncuser /home/novncuser

# Add custom CSS reference to noVNC index page
RUN sed -i '/<\/head>/i <link rel="stylesheet" href="custom/app.css">' /usr/share/novnc/vnc.html \
    && chown -R novncuser:novncuser /usr/share/novnc

# Switch to non-root user for better security
USER novncuser

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
