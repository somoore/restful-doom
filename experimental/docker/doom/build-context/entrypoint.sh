#!/bin/bash
set -e

# Wait for X11 server to be ready
echo "Waiting for X11 server..."
until xdpyinfo -display :1 >/dev/null 2>&1; do
    echo "X11 server not ready yet, waiting..."
    sleep 2
done
echo "X11 server is ready"

# Check for the specified WAD file
if [ ! -f "/app/wad/$WAD_FILENAME" ]; then
  echo "Warning: Specified WAD file '$WAD_FILENAME' not found in /app/wad/"
  
  # Check if DOOM.WAD exists as a fallback
  if [ -f "/app/wad/DOOM.WAD" ]; then
    echo "Using DOOM.WAD as fallback"
    WAD_FILENAME="DOOM.WAD"
  else
    echo "Error: No WAD files found. DOOM cannot start."
    exit 1
  fi
fi

# Get port information
API_PORT=${API_PORT:-8000}

echo "Starting DOOM with RESTful API..."
echo "Command: ./src/restful-doom -iwad $WAD_FILENAME -width 1024 -height 768 -apihost 0.0.0.0 -apiport ${API_PORT} -ticrate 35 -nosleep -window -v"

# Function to check if DOOM is already running
check_doom_running() {
  pgrep -f "./src/restful-doom" > /dev/null
  return $?
}

# Function to handle DOOM crashes and restarts
restart_doom() {
  while true; do
    if ! check_doom_running; then
      echo "[$(date)] Starting DOOM process..."
      cd /app && ./src/restful-doom -iwad /app/wad/${WAD_FILENAME} -width 1024 -height 768 -window -apihost 0.0.0.0 -apiport ${API_PORT} -ticrate 35 -nosleep -v > /app/doom.log 2>&1 &
      DOOM_PID=$!
      echo "DOOM started with PID $DOOM_PID"
    fi
    sleep 5
  done
}

# Start DOOM in the background with auto-restart
restart_doom &
RESTART_PID=$!

# Make a test request to the API after a delay to ensure it's working
(sleep 10; curl -s http://localhost:${API_PORT}/api/player) &

# Keep the container running
wait $RESTART_PID
