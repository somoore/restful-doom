#!/bin/bash
set -e

# Get environment variables
VNC_HOST=${VNC_HOST:-vnc-server}
VNC_PORT=${VNC_PORT:-5900}
NOVNC_PORT=${NOVNC_PORT:-6080}

echo "Waiting for VNC server to be ready..."
until nc -z $VNC_HOST $VNC_PORT; do
    echo "VNC server not ready yet, waiting..."
    sleep 2
done
echo "VNC server is ready"

echo "Starting noVNC web server..."
echo "Connecting to VNC server at $VNC_HOST:$VNC_PORT"
echo "Serving noVNC on port $NOVNC_PORT"

# Start websockify directly with explicit websocket path
websockify --web=/usr/share/novnc --heartbeat=30 $NOVNC_PORT $VNC_HOST:$VNC_PORT & 

# Keep the container running
tail -f /dev/null
