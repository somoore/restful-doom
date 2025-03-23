#!/bin/bash
set -e

# Wait for X11 server to be ready
echo "Waiting for X11 server..."
until xdpyinfo -display :1 >/dev/null 2>&1; do
    echo "X11 server not ready yet, waiting..."
    sleep 2
done
echo "X11 server is ready"

# Get port information
VNC_PORT=${VNC_PORT:-5900}

echo "Starting VNC server..."
x11vnc -forever -passwd password -display :1 -noshm -noxdamage -noxfixes -noxrecord -shared \
  -ncache 10 -ncache_cr -ping 30 -nolookup -noipv6 -nowf -nowcr \
  -timeout 60 -rfbport ${VNC_PORT} -noprimary -nosel -nosetclipboard -noclipboard

# Keep the container running
tail -f /dev/null
