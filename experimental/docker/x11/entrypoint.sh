#!/bin/bash
set -ex

echo "Starting X virtual framebuffer..."

# Switch to root user for privileged operations
# no sudo needed as we're running as root in the x11-server container
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix

# Clean up any existing X11 lock files
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1

# Debug: Show directory permissions
echo "X11 socket directory permissions:"
ls -la /tmp/.X11-unix/

# Start Xvfb with proper permissions
echo "Starting Xvfb..."
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset -nolisten tcp 2>&1 &
XVFB_PID=$!

# Wait for X11 socket to be created
while [ ! -e /tmp/.X11-unix/X1 ]; do
    echo "Waiting for X11 socket..."
    sleep 1
done

# Ensure socket has correct permissions
chmod 1777 /tmp/.X11-unix/X1

# Debug: Show Xvfb process
echo "Xvfb process:"
ps aux | grep Xvfb

# Wait longer for Xvfb to start and check multiple times
for i in {1..15}; do
    if ps -p $XVFB_PID > /dev/null; then
        echo "Waiting for Xvfb to initialize (attempt $i)..."
        if DISPLAY=:1 xdpyinfo >/dev/null 2>&1; then
            echo "Xvfb started successfully!"
            echo "X11 socket directory contents:"
            ls -la /tmp/.X11-unix/
            break
        fi
    else
        echo "ERROR: Xvfb process died!"
        echo "Last lines of Xvfb log:"
        tail -n 20 /var/log/Xvfb.1.log || true
        exit 1
    fi
    sleep 2
done

# Final check
if ! DISPLAY=:1 xdpyinfo >/dev/null 2>&1; then
    echo "ERROR: Xvfb failed to initialize properly!"
    echo "X11 socket directory contents:"
    ls -la /tmp/.X11-unix/
    echo "Last lines of Xvfb log:"
    tail -n 20 /var/log/Xvfb.1.log || true
    exit 1
fi

# Test that X server is working
echo "Testing X server..."
DISPLAY=:1 xdpyinfo || echo "X server not responding properly"

echo "Starting window manager..."
fluxbox -display :1 &
FLUXBOX_PID=$!
sleep 1

echo "X11 server ready"

# Keep the container running
trap "echo 'Shutting down X11 server'; kill $XVFB_PID $FLUXBOX_PID" SIGTERM SIGINT
wait
