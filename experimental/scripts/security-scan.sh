#!/bin/bash
# RESTful DOOM Security Scanner
# Analyzes Docker images for security vulnerabilities
# -----------------------------------------------------

# ANSI color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Banner
echo -e "${BOLD}${BLUE}🔒 RESTful DOOM Security Scanner${NC}"
echo -e "${BLUE}=================================${NC}"
echo -e "${YELLOW}⚠️  FOR DEVELOPMENT USE ONLY${NC}\n"

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker to continue.${NC}"
    exit 1
fi

# Check if Trivy is installed
if ! command -v trivy &> /dev/null; then
    echo -e "${YELLOW}⚙️  Trivy not found, attempting to install...${NC}"
    
    # Try to install Trivy based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Installing via Homebrew..."
        brew tap aquasecurity/trivy
        brew install trivy
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Installing via apt for Debian/Ubuntu..."
        sudo apt-get install wget apt-transport-https gnupg lsb-release
        wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
        echo deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main | sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install trivy
    else
        echo -e "${RED}❌ Unsupported OS. Please install Trivy manually:${NC}"
        echo -e "${YELLOW}https://aquasecurity.github.io/trivy/latest/getting-started/installation/${NC}"
        exit 1
    fi
    
    # Verify installation
    if ! command -v trivy &> /dev/null; then
        echo -e "${RED}❌ Failed to install Trivy. Please install manually.${NC}"
        exit 1
    fi
fi

# List of Docker images to scan
DOCKER_IMAGES=(
    "docker-doom-game:1.0"
    "docker-novnc:1.0"
    "docker-vnc-server:1.0"
    "docker-web-ui:1.0"
    "docker-x11-server:1.0"
)

echo -e "${CYAN}🔍 Scanning project Docker images for vulnerabilities...${NC}\n"

# Initialize counters
total_critical=0
total_high=0
total_medium=0
total_low=0

# Prepare report file
report_file="/tmp/restful-doom-security-report-$(date +%Y%m%d-%H%M%S).txt"
echo "# RESTful DOOM Security Report - $(date)" > "$report_file"
echo "===============================================" >> "$report_file"
echo "" >> "$report_file"

# Scan each image
for image in "${DOCKER_IMAGES[@]}"; do
    echo -e "${BOLD}${CYAN}📋 Scanning ${image}${NC}"
    echo -e "${CYAN}-----------------${NC}"
    
    # Add to report
    echo "## $image" >> "$report_file"
    echo "" >> "$report_file"
    
    # Run the Trivy scan and capture detailed results
    trivy_results=$(trivy image --no-progress "$image")
    trivy_exit_code=$?
    
    # Extract vulnerability counts
    critical_count=$(echo "$trivy_results" | grep -c 'CRITICAL')
    high_count=$(echo "$trivy_results" | grep -c 'HIGH')
    medium_count=$(echo "$trivy_results" | grep -c 'MEDIUM')
    low_count=$(echo "$trivy_results" | grep -c 'LOW')
    
    # Update totals
    total_critical=$((total_critical + critical_count))
    total_high=$((total_high + high_count))
    total_medium=$((total_medium + medium_count))
    total_low=$((total_low + low_count))
    
    # Save detailed results to the report
    echo "$trivy_results" >> "$report_file"
    echo "" >> "$report_file"
    
    # Print summary for this image
    echo -e "${MAGENTA}Summary for ${image}:${NC}"
    echo -e "  ${RED}Critical: ${critical_count}${NC}"
    echo -e "  ${YELLOW}High:     ${high_count}${NC}"
    echo -e "  ${YELLOW}Medium:   ${medium_count}${NC}"
    echo -e "  ${GREEN}Low:      ${low_count}${NC}"
    
    # Add spacer between images
    echo ""
done

# Print overall summary
echo -e "${BOLD}${BLUE}==== Overall Security Summary ====${NC}"
echo -e "${RED}Critical Vulnerabilities: ${total_critical}${NC}"
echo -e "${YELLOW}High Vulnerabilities:     ${total_high}${NC}"
echo -e "${YELLOW}Medium Vulnerabilities:   ${total_medium}${NC}"
echo -e "${GREEN}Low Vulnerabilities:      ${total_low}${NC}"
echo ""

# Calculate risk level
if [ $total_critical -gt 0 ]; then
    risk_level="${RED}CRITICAL${NC}"
    risk_message="Critical vulnerabilities found. Immediate action is required!"
elif [ $total_high -gt 5 ]; then
    risk_level="${YELLOW}HIGH${NC}"
    risk_message="Multiple high vulnerabilities found. Action recommended."
elif [ $total_high -gt 0 ]; then
    risk_level="${YELLOW}MODERATE${NC}"
    risk_message="Some high vulnerabilities found. Consider addressing them."
elif [ $total_medium -gt 10 ]; then
    risk_level="${YELLOW}LOW-MODERATE${NC}"
    risk_message="Several medium vulnerabilities found."
else
    risk_level="${GREEN}LOW${NC}"
    risk_message="No critical or high vulnerabilities found."
fi

echo -e "${BOLD}Risk Assessment: ${risk_level}${NC}"
echo -e "${risk_message}"
echo ""

# Report location
echo -e "${BLUE}📄 Detailed report saved to: ${CYAN}${report_file}${NC}"
echo -e "${BLUE}📋 View with: ${CYAN}cat ${report_file}${NC}"
echo ""

# Security recommendations
echo -e "${BOLD}${BLUE}==== Security Recommendations ====${NC}"
echo -e "${YELLOW}1. Remember that RESTful DOOM is for development use only.${NC}"
echo -e "${YELLOW}2. Keep all base images updated to their latest versions.${NC}"
echo -e "${YELLOW}3. Consider implementing vulnerability scanning in CI/CD pipelines.${NC}"
echo -e "${YELLOW}4. Review the detailed report for specific remediation steps.${NC}"
echo -e "${YELLOW}5. Use the security-recommendations.md file as a reference.${NC}"
echo ""

exit 0
