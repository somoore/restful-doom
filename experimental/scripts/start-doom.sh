#!/bin/bash
set -e

# RESTful DOOM Microservices starter script
echo "RESTful DOOM Microservices"
echo "=========================="
echo

# Set default values for environment variables
export HOST_API_PORT=${HOST_API_PORT:-8000}
export HOST_VNC_PORT=${HOST_VNC_PORT:-5900}
export HOST_NOVNC_PORT=${HOST_NOVNC_PORT:-6080}
export HOST_WEB_PORT=${HOST_WEB_PORT:-8080}

# Ensure WAD_PATH is absolute
if [ -z "$WAD_PATH" ]; then
    # Check if we're in the experimental or scripts directory
    if [[ "$(pwd)" == */experimental/scripts ]]; then
        # We're in the scripts directory
        export WAD_PATH="$(dirname "$(pwd)")/wad"
    elif [[ "$(pwd)" == */experimental ]]; then
        # We're in the experimental directory
        export WAD_PATH="$(pwd)/wad"
    else
        # We're in the project root or elsewhere
        export WAD_PATH="$(pwd)/experimental/wad"
    fi
else
    # Convert relative path to absolute if needed
    if [[ "$WAD_PATH" != /* ]]; then
        export WAD_PATH="$(pwd)/$WAD_PATH"
    fi
fi

# Auto-detect WAD file if not specified
if [ -z "$WAD_FILENAME" ]; then
    # First check for DOOM.WAD
    if [ -f "$WAD_PATH/DOOM.WAD" ]; then
        export WAD_FILENAME="DOOM.WAD"
    else
        # Find first .wad file (case insensitive)
        WAD_FILE=$(find "$WAD_PATH" -maxdepth 1 -type f -iname "*.wad" | head -n 1)
        if [ -n "$WAD_FILE" ]; then
            export WAD_FILENAME=$(basename "$WAD_FILE")
        fi
    fi
fi

# If still no WAD file found or specified, exit
if [ -z "$WAD_FILENAME" ] || [ ! -f "$WAD_PATH/$WAD_FILENAME" ]; then
    echo "Error: No WAD files found in $WAD_PATH"
    echo "Please place a DOOM WAD file (*.wad) in $WAD_PATH"
    echo "The preferred filename is DOOM.WAD"
    exit 1
fi

# Verify WAD file is readable
if [ ! -r "$WAD_PATH/$WAD_FILENAME" ]; then
    echo "Error: WAD file $WAD_PATH/$WAD_FILENAME exists but is not readable"
    echo "Please check file permissions"
    exit 1
fi

cd "$(dirname "$0")/../.."

# Make sure we're in the right directory
if [ ! -f "experimental/docker/docker-compose.yml" ]; then
  echo "Error: Could not find experimental/docker/docker-compose.yml in $(pwd)"
  exit 1
fi

# Check for Docker and Docker Compose
if ! command -v docker &> /dev/null; then
  echo "Error: Docker is not installed or not in PATH. Please install Docker first."
  exit 1
fi

if ! docker compose version &> /dev/null; then
  echo "Error: Docker Compose is not installed or not in PATH. Please install Docker Compose first."
  exit 1
fi

# Display WAD file information
echo "Using WAD file: $WAD_FILENAME"
echo "WAD location: $WAD_PATH/$WAD_FILENAME"
echo

# Print configuration
echo "Configuration:"
echo "- API Port: $HOST_API_PORT"
echo "- VNC Port: $HOST_VNC_PORT"
echo "- noVNC Port: $HOST_NOVNC_PORT"
echo "- Web UI Port: $HOST_WEB_PORT"
echo "- WAD File: $WAD_FILENAME"
echo "- WAD Path: $WAD_PATH"
echo

# Function to check if all services are healthy
check_services_health() {
    local max_attempts=30
    local attempt=1
    local all_healthy=false

    echo "Waiting for services to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if docker compose -f experimental/docker/docker-compose.yml ps | grep -q "(unhealthy)"; then
            echo "Some services are still unhealthy (attempt $attempt/$max_attempts)"
            sleep 2
            ((attempt++))
        else
            if docker compose -f experimental/docker/docker-compose.yml ps | grep -q "(healthy)"; then
                all_healthy=true
                break
            fi
            echo "Waiting for health checks (attempt $attempt/$max_attempts)"
            sleep 2
            ((attempt++))
        fi
    done

    if [ "$all_healthy" = true ]; then
        return 0
    else
        return 1
    fi
}

# Build and run containers
echo "Building and starting microservices..."

# Stop any existing containers first
docker compose -f experimental/docker/docker-compose.yml down 2>/dev/null || true

# Start fresh
docker compose -f experimental/docker/docker-compose.yml up -d --build --force-recreate

# Handle cleanup on exit
trap 'echo "Stopping containers..."; docker compose -f experimental/docker/docker-compose.yml down' EXIT

# Wait for services to be healthy and open port info page
if check_services_health; then
    echo "All services are healthy!"
    echo "Opening port information page..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://127.0.0.1:${HOST_WEB_PORT}/port-info.html"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "http://127.0.0.1:${HOST_WEB_PORT}/port-info.html"
    else
        echo "Please open http://127.0.0.1:${HOST_WEB_PORT}/port-info.html in your browser"
    fi
    # Keep containers running
    docker compose -f experimental/docker/docker-compose.yml logs -f
else
    echo "Error: Services did not become healthy within the timeout period"
    docker compose -f experimental/docker/docker-compose.yml logs
    exit 1
fi
