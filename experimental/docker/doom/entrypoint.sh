#!/bin/bash
set -e

# NOTE: We're using the X11 server from the x11-server container
# So we don't start our own Xvfb or VNC server here

# Wait for X11 server to be ready
echo "Waiting for X11 server (DISPLAY=:1)..."
until xdpyinfo -display :1 >/dev/null 2>&1; do
    echo "X11 server not ready yet, waiting..."
    sleep 1
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

# Function to check if DOOM is running and healthy
check_doom_health() {
  # Check if process exists and is not zombie
  local pid=$(pgrep -f "./src/restful-doom")
  if [ -z "$pid" ]; then
    return 1
  fi
  
  # Check process state
  if [ "$(ps -p $pid -o state= 2>/dev/null)" = "Z" ]; then
    return 1
  fi

  # Check if API is responding with timeout
  if ! timeout 5 curl -s -f "http://localhost:${API_PORT}/api/player" > /dev/null; then
    return 1
  fi

  return 0
}

# Function to kill any existing DOOM processes and cleanup
kill_doom() {
  echo "Cleaning up DOOM processes and resources..."
  
  # Kill any existing DOOM processes more aggressively
  pkill -9 -f "./src/restful-doom" || true
  
  # Force close any process using our port
  fuser -k -9 ${API_PORT}/tcp || true
  
  # Clean up any zombie processes
  wait $(jobs -p) 2>/dev/null || true
  
  # Wait for port to be free with timeout
  local timeout=10
  while netstat -tln | grep -q ":${API_PORT} "; do
    echo "Waiting for port ${API_PORT} to be released..."
    timeout=$((timeout-1))
    if [ $timeout -le 0 ]; then
      echo "Force killing any remaining processes on port ${API_PORT}"
      fuser -k -9 ${API_PORT}/tcp || true
      break
    fi
    sleep 1
  done
  
  # DO NOT remove X11 socket files - they're managed by x11-server container
  # Simply check if X11 is still available
  if ! xdpyinfo -display :1 >/dev/null 2>&1; then
    echo "X11 display appears to be disconnected"
  fi
  
  # Clear any shared memory segments
  for id in $(ipcs -m | grep "^0x" | awk '{print $2}'); do
    ipcrm -m $id 2>/dev/null || true
  done
  
  sleep 3
}

# Function to check if port is in use
check_port() {
  if netstat -tln | grep -q ":${API_PORT} "; then
    return 0
  else
    return 1
  fi
}

# Function to wait for port to be free
wait_for_port() {
  local retries=10
  while check_port && [ $retries -gt 0 ]; do
    echo "Waiting for port ${API_PORT} to be free..."
    sleep 2
    retries=$((retries - 1))
  done
}

# Function to handle DOOM crashes and restarts
restart_doom() {
  while true; do
    if ! check_doom_health; then
      # Only kill and restart if the process is unhealthy
      echo "[$(date)] DOOM process unhealthy, restarting..."
      kill_doom
      
      cd /app
      echo "Current directory: $(pwd)"
      echo "WAD file: /app/wad/${WAD_FILENAME}"
      ls -l /app/wad/
      
      echo "Starting DOOM with debug output..."
      # Ensure the environment is properly set for X11 rendering
      export DISPLAY=:1
      export SDL_VIDEODRIVER=x11
      export LIBGL_ALWAYS_SOFTWARE=1
      
      # Print environment for debugging
      echo "Environment variables:"
      env | grep -E "(DISPLAY|SDL|GL)"
      
      # Verify X11 access
      echo "Testing X11 connection:"
      xdpyinfo -display :1 || echo "X11 connection failed"
      
      # Check if DOOM WAD file exists and is readable
      echo "Checking WAD file: /app/wad/${WAD_FILENAME}"
      ls -la /app/wad/${WAD_FILENAME}
      
      # Start DOOM with a more compatible set of options
      cd /app && ./src/restful-doom -iwad /app/wad/${WAD_FILENAME} -width 1024 -height 768 -window -apihost 0.0.0.0 -apiport ${API_PORT} -ticrate 35 -mb 128 -heapsize 128 -zone 16384 -nosleep -v > /app/doom.log 2> /app/doom.error.log &
      DOOM_PID=$!
      echo "DOOM started with PID $DOOM_PID"
      
      # Give DOOM time to start up and initialize
      sleep 5
      
      # Check if DOOM is still running
      if ps -p $DOOM_PID > /dev/null; then
        echo "DOOM is running with PID $DOOM_PID"
      else
        echo "DOOM process failed to start or crashed"
        cat /app/doom.error.log
        cat /app/doom.log
        exit 1
      fi
      
      # Wait for process to start and check for errors
      sleep 5
      if ! check_doom_health; then
        echo "DOOM process failed to start!"
        if [ -s /app/doom.error.log ]; then
          echo "DOOM Error Log:"
          cat /app/doom.error.log
        fi
        if [ -s /app/doom.log ]; then
          echo "DOOM Output Log:"
          cat /app/doom.log
        fi
      fi
    fi
    
    # Wait before checking again
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
