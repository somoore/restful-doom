#!/bin/bash
set -e

# RESTful DOOM Microservices test script
echo "RESTful DOOM Microservices Tester"
echo "=================================="
echo

# Set default values for environment variables
export HOST_API_PORT=${HOST_API_PORT:-8000}
export HOST_VNC_PORT=${HOST_VNC_PORT:-5900}
export HOST_NOVNC_PORT=${HOST_NOVNC_PORT:-6080}
export HOST_WEB_PORT=${HOST_WEB_PORT:-8080}

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to check container health
check_container() {
  local container_name=$1
  local service_name=$2
  
  echo -n "Checking ${service_name} container... "
  if [ "$(docker ps -q -f name=${container_name})" ]; then
    if [ "$(docker inspect --format='{{.State.Health.Status}}' ${container_name} 2>/dev/null)" == "healthy" ]; then
      echo -e "${GREEN}HEALTHY${NC}"
      return 0
    else
      echo -e "${YELLOW}RUNNING (health status unavailable or not healthy yet)${NC}"
      return 1
    fi
  else
    echo -e "${RED}NOT RUNNING${NC}"
    return 2
  fi
}

# Function to test API endpoints
test_api_endpoint() {
  local endpoint=$1
  local method=${2:-GET}
  local expected_status=${3:-200}
  
  echo -n "Testing API endpoint (${method} ${endpoint})... "
  local status=$(curl -s -o /dev/null -w "%{http_code}" -X ${method} http://localhost:${HOST_API_PORT}${endpoint})
  
  if [ "$status" == "$expected_status" ]; then
    echo -e "${GREEN}OK (${status})${NC}"
    return 0
  else
    echo -e "${RED}FAILED (Expected: ${expected_status}, Got: ${status})${NC}"
    return 1
  fi
}

# Function to test service connectivity
test_service_connectivity() {
  local host=$1
  local port=$2
  local service_name=$3
  
  echo -n "Testing connectivity to ${service_name} (${host}:${port})... "
  if nc -z -w 5 ${host} ${port} > /dev/null 2>&1; then
    echo -e "${GREEN}CONNECTED${NC}"
    return 0
  else
    echo -e "${RED}FAILED${NC}"
    return 1
  fi
}

echo "Starting microservices health check..."
echo "-------------------------------------"

# Check if microservices are running
echo "Checking if Docker Compose services are running..."
cd "$(dirname "$0")/.."
if [ ! -f "docker/docker-compose.yml" ]; then
  echo -e "${RED}Error: Could not find docker/docker-compose.yml in $(pwd)${NC}"
  exit 1
fi

# Function to check if a file exists in a container
check_file_in_container() {
  local container_name=$1
  local file_path=$2
  local description=$3
  
  echo -n "Checking ${description} in ${container_name}... "
  if docker exec ${container_name} test -f "${file_path}"; then
    echo -e "${GREEN}FOUND${NC}"
    return 0
  else
    echo -e "${RED}NOT FOUND${NC}"
    return 1
  fi
}

# Function to check if a process is running in a container
check_process_in_container() {
  local container_name=$1
  local process_name=$2
  local description=$3
  
  echo -n "Checking ${description} process in ${container_name}... "
  if docker exec ${container_name} pgrep -f "${process_name}" > /dev/null; then
    echo -e "${GREEN}RUNNING${NC}"
    return 0
  else
    echo -e "${RED}NOT RUNNING${NC}"
    return 1
  fi
}

# Check individual container health
# Extract project name from directory path to find container names
PROJECT_NAME=$(basename $(pwd) | tr '[:upper:]' '[:lower:]')

check_container "${PROJECT_NAME}-x11-server-1" "X11 Display"
check_container "${PROJECT_NAME}-doom-game-1" "DOOM Game"
check_container "${PROJECT_NAME}-vnc-server-1" "VNC Server"
check_container "${PROJECT_NAME}-novnc-1" "noVNC"
check_container "${PROJECT_NAME}-web-ui-1" "Web UI"

echo
echo "Checking critical files and processes..."
echo "-----------------------------------"
# Check WAD file in DOOM container
check_file_in_container "${PROJECT_NAME}-doom-game-1" "/app/wad/${WAD_FILENAME:-DOOM.WAD}" "WAD file"

# Check X11 socket in containers that need it
check_file_in_container "${PROJECT_NAME}-doom-game-1" "/tmp/.X11-unix/X1" "X11 socket"
check_file_in_container "${PROJECT_NAME}-vnc-server-1" "/tmp/.X11-unix/X1" "X11 socket"

# Check critical processes
check_process_in_container "${PROJECT_NAME}-x11-server-1" "Xvfb" "X11 server"
check_process_in_container "${PROJECT_NAME}-doom-game-1" "restful-doom" "DOOM game"
check_process_in_container "${PROJECT_NAME}-vnc-server-1" "x11vnc" "VNC server"
check_process_in_container "${PROJECT_NAME}-novnc-1" "websockify" "noVNC websocket"

echo
echo "Testing connectivity to services..."
echo "----------------------------------"
test_service_connectivity "localhost" ${HOST_API_PORT} "DOOM API"
test_service_connectivity "localhost" ${HOST_VNC_PORT} "VNC Server"
test_service_connectivity "localhost" ${HOST_NOVNC_PORT} "noVNC Web Server"
test_service_connectivity "localhost" ${HOST_WEB_PORT} "Web UI"

echo
echo "Testing API endpoints..."
echo "----------------------"
test_api_endpoint "/api/things"
test_api_endpoint "/api/players"
test_api_endpoint "/api/sectors"
test_api_endpoint "/api/doors"
test_api_endpoint "/api/players/0/health"

echo
echo "Summary:"
echo "--------"
echo "1. To view the game: http://localhost:${HOST_NOVNC_PORT}/vnc.html"
echo "2. To access the Web UI: http://localhost:${HOST_WEB_PORT}"
echo "3. Direct API access: http://localhost:${HOST_API_PORT}/api"
echo
echo "Testing completed. Check the results above for any issues."
