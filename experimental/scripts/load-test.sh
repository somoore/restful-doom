#!/bin/bash
set -e

# RESTful DOOM Microservices Load Testing Script
echo "RESTful DOOM API Load Tester"
echo "==========================="
echo

# Default settings
HOST_API_PORT=${HOST_API_PORT:-8000}
CONCURRENCY=${CONCURRENCY:-10}
REQUESTS=${REQUESTS:-1000}
ENDPOINT=${ENDPOINT:-"/api/things"}
TIMEOUT=${TIMEOUT:-10}
METHOD=${METHOD:-"GET"}
DATA=${DATA:-""}

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display help
function show_help {
  echo "Usage: $0 [OPTIONS]"
  echo
  echo "Options:"
  echo "  -c, --concurrency NUMBER   Number of concurrent connections (default: $CONCURRENCY)"
  echo "  -n, --requests NUMBER      Total number of requests (default: $REQUESTS)"
  echo "  -e, --endpoint URL_PATH    API endpoint to test (default: $ENDPOINT)"
  echo "  -t, --timeout SECONDS      Request timeout in seconds (default: $TIMEOUT)"
  echo "  -m, --method HTTP_METHOD   HTTP method to use (default: $METHOD)"
  echo "  -d, --data JSON_DATA       Request data for POST/PUT (default: empty)"
  echo "  -h, --help                 Display this help message"
  echo
  echo "Examples:"
  echo "  $0 -c 20 -n 2000 -e /api/players"
  echo "  $0 --method POST --endpoint /api/players/0/position --data '{\"x\":100,\"y\":100}'"
  echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    -n|--requests)
      REQUESTS="$2"
      shift 2
      ;;
    -e|--endpoint)
      ENDPOINT="$2"
      shift 2
      ;;
    -t|--timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    -m|--method)
      METHOD="$2"
      shift 2
      ;;
    -d|--data)
      DATA="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if required tools are installed
if ! command -v curl &> /dev/null; then
  echo -e "${RED}Error: curl is not installed. Please install curl first.${NC}"
  exit 1
fi

if ! command -v bc &> /dev/null; then
  echo -e "${YELLOW}Warning: bc is not installed. Some calculations may not work correctly.${NC}"
fi

# Check if the API is running
echo -n "Checking if API is available... "
if curl -s -o /dev/null -w "%{http_code}" http://localhost:${HOST_API_PORT}/api/health 2>/dev/null | grep -q "200"; then
  echo -e "${GREEN}OK${NC}"
else
  echo -e "${RED}FAILED${NC}"
  echo -e "${RED}Error: API is not available at http://localhost:${HOST_API_PORT}${NC}"
  echo "Make sure the microservices are running with ./start-doom.sh"
  exit 1
fi

# Prepare test
URL="http://localhost:${HOST_API_PORT}${ENDPOINT}"
echo
echo -e "${BLUE}Load Test Configuration:${NC}"
echo "- API URL: $URL"
echo "- Method: $METHOD"
echo "- Concurrency: $CONCURRENCY"
echo "- Total Requests: $REQUESTS"
echo "- Timeout: ${TIMEOUT}s"
if [[ -n "$DATA" ]]; then
  echo "- Request Data: $DATA"
fi
echo

# Start time
START_TIME=$(date +%s.%N)

# Create a temporary file for storing results
TEMP_FILE=$(mktemp)
ERRORS_FILE=$(mktemp)

# Function to execute a batch of requests
function execute_batch {
  local batch_size=$1
  local completed=0
  local success=0
  local errors=0
  
  echo -n "Executing batch of $batch_size requests... "
  
  for ((i=1; i<=$batch_size; i++)); do
    if [[ "$METHOD" == "GET" ]]; then
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X $METHOD -m $TIMEOUT "$URL" 2>/dev/null)
    else
      HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X $METHOD -H "Content-Type: application/json" -d "$DATA" -m $TIMEOUT "$URL" 2>/dev/null)
    fi
    
    if [[ "$HTTP_CODE" -ge 200 && "$HTTP_CODE" -lt 300 ]]; then
      ((success++))
    else
      ((errors++))
      echo "Error $HTTP_CODE on request $i" >> "$ERRORS_FILE"
    fi
    
    ((completed++))
    
    # Display progress every 10% of the batch
    if ((i % (batch_size / 10) == 0)) || ((i == batch_size)); then
      PCT=$((completed * 100 / batch_size))
      echo -ne "\rExecuting batch of $batch_size requests... ${PCT}% complete"
    fi
  done
  
  echo -e "\rExecuting batch of $batch_size requests... ${GREEN}DONE${NC}"
  echo "$success $errors" >> "$TEMP_FILE"
}

# Execute requests in batches based on concurrency
BATCH_SIZE=$((REQUESTS / CONCURRENCY))
if [[ $BATCH_SIZE -lt 1 ]]; then
  BATCH_SIZE=1
fi

echo "Starting load test with $CONCURRENCY concurrent users ($BATCH_SIZE requests per user)"
echo

for ((i=1; i<=$CONCURRENCY; i++)); do
  execute_batch $BATCH_SIZE &
done

# Wait for all background jobs to complete
wait

# Calculate results
TOTAL_SUCCESS=0
TOTAL_ERRORS=0

while read line; do
  SUCCESS=$(echo $line | cut -d' ' -f1)
  ERRORS=$(echo $line | cut -d' ' -f2)
  TOTAL_SUCCESS=$((TOTAL_SUCCESS + SUCCESS))
  TOTAL_ERRORS=$((TOTAL_ERRORS + ERRORS))
done < "$TEMP_FILE"

# End time
END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)
RPS=$(echo "scale=2; $TOTAL_SUCCESS / $DURATION" | bc)

# Display results
echo
echo -e "${BLUE}Load Test Results:${NC}"
echo "- Total Requests: $REQUESTS"
echo "- Successful Requests: $TOTAL_SUCCESS"
echo "- Failed Requests: $TOTAL_ERRORS"
echo "- Duration: ${DURATION}s"
echo "- Requests per second: ${RPS}/sec"
echo

if [[ $TOTAL_ERRORS -gt 0 ]]; then
  echo -e "${RED}Some errors occurred during the test. See details below:${NC}"
  cat "$ERRORS_FILE"
else
  echo -e "${GREEN}All requests completed successfully!${NC}"
fi

# Clean up
rm -f "$TEMP_FILE" "$ERRORS_FILE"

# Check for segmentation faults in logs (requires sudo access)
echo
echo "Checking for segmentation faults in container logs..."
if docker logs restful-doom-doom-game-1 2>&1 | grep -q "segmentation fault"; then
  echo -e "${RED}Segmentation faults detected in the DOOM container logs!${NC}"
  echo "This indicates that the microservices approach might not have fully resolved the stability issues."
else
  echo -e "${GREEN}No segmentation faults detected in the DOOM container logs.${NC}"
  echo "The microservices approach appears to have improved stability!"
fi

echo
echo "Load test completed."
