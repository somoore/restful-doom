#!/bin/bash
set -e

echo "Starting X virtual framebuffer..."
Xvfb :1 -screen 0 1024x768x24 &
XVFB_PID=$!
sleep 2

# Verify Xvfb is running
if ! ps -p $XVFB_PID > /dev/null; then
    echo "ERROR: Xvfb failed to start!"
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
