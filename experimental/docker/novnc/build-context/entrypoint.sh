#!/bin/bash
set -e

# Get environment variables
VNC_HOST=${VNC_HOST:-vnc-server}
VNC_PORT=${VNC_PORT:-5900}
NOVNC_PORT=${NOVNC_PORT:-6080}

echo "Starting noVNC web server..."
echo "Connecting to VNC server at $VNC_HOST:$VNC_PORT"
echo "Serving noVNC on port $NOVNC_PORT"

# Start noVNC with explicit options
/usr/share/novnc/utils/launch.sh --vnc $VNC_HOST:$VNC_PORT --listen $NOVNC_PORT --web /usr/share/novnc

# If the above fails, try the backup method
if [ $? -ne 0 ]; then
    echo "Primary noVNC launch method failed, trying backup method..."
    websockify --web=/usr/share/novnc $NOVNC_PORT $VNC_HOST:$VNC_PORT
fi
