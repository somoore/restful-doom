# RESTful DOOM - Experimental Features

This directory contains experimental features and enhancements for RESTful DOOM, including Docker containerization, web-based interfaces, and a new microservices architecture designed to improve stability and maintainability.

## Directory Structure

- `docker/` - Docker-related files and configurations for both monolithic and microservices approaches
  - `doom/` - DOOM game service container files
  - `x11/` - X11 display service container files
  - `vnc/` - VNC server service container files
  - `novnc/` - noVNC web service container files
  - `web-ui/` - Web UI service container files
- `scripts/` - Helper scripts for running and managing the containerized environment

## Quick Start

### Monolithic Approach (Original)

1. Build and start the monolithic Docker container:
   ```bash
   ./scripts/start-doom.sh
   ```

2. Access the various interfaces:
   - Launch Page: http://127.0.0.1:8000/play-doom.html
   - Dashboard: http://127.0.0.1:8000/doom-dashboard.html
   - API Documentation: http://127.0.0.1:8000/docs.html
   - Direct VNC: connect to 127.0.0.1:5900 (password: password)
   - Web VNC: http://127.0.0.1:6080/vnc.html?autoconnect=true&view_only=0

### Microservices Approach (New)

1. Place your DOOM WAD file in the `experimental/wad/` directory:
   - The system will automatically detect and use any `.wad` file in this directory
   - Only one WAD file should be present to avoid ambiguity
   - You can swap WAD files without rebuilding containers
   - Alternatively, set the `WAD_FILENAME` environment variable to specify a particular WAD file

2. Build and start the microservices architecture:
   ```bash
   ./scripts/start-doom.sh
   ```

2. Test the services are working properly:
   ```bash
   ./scripts/test-microservices.sh
   ```

3. Access the various interfaces (same ports as the monolithic version):
   - Web UI: http://127.0.0.1:8080
   - API Access: http://127.0.0.1:8000/api
   - Direct VNC: connect to 127.0.0.1:5900 (password: password)
   - Web VNC: http://127.0.0.1:6080/vnc.html

4. For load testing the API:
   ```bash
   ./scripts/load-test.sh --endpoint /api/things --concurrency 10 --requests 100
   ```

## Features

### Docker Support
- Containerized environment for consistent deployment
- Debug configuration available
- VNC server integration

### Web Interfaces
- Interactive DOOM dashboard
- Real-time API interaction
- VNC web client for remote gameplay

### Microservices Architecture
- Improved stability through service isolation
- Fault-tolerant design with independent containers
- Better resource allocation and scaling
- Simplified maintenance and debugging
- Container health checks and monitoring

## Development

### Monolithic Architecture
To modify or enhance the original monolithic container:

1. The Docker configuration is in `docker/Dockerfile.debug`
2. Web interfaces are in the `docker/web-ui/` directory
3. Helper scripts are in the `scripts/` directory

### Microservices Architecture
To modify or enhance the microservices architecture:

1. The Docker Compose configuration is in `docker/docker-compose.yml`
2. Each service has its own Dockerfile and entrypoint.sh script in its respective directory
3. Shared volumes handle inter-service communication
4. See `architecture.md` for a detailed explanation of the microservices design
5. Refer to `DEPLOYMENT.md` for deployment instructions and configuration options
6. Check `stability-improvements.md` for details on stability enhancements

## Contributing

These experimental features are intended to be contributed back to the main RESTful DOOM project. When submitting PRs:

1. Keep changes isolated to the experimental directory
2. Maintain backward compatibility
3. Document any new dependencies or requirements
4. Test thoroughly in a containerized environment

## Notes

### Monolithic Container
- VNC password is set to: "password"
- The container exposes ports 8000, 5900, and 6080
- All web interfaces are served from the container

### Microservices Environment
- Default ports: 8000 (API), 8080 (Web UI), 5900 (VNC), 6080 (noVNC)
- Ports are configurable via environment variables
- Services communicate via shared Docker volumes
- Container health checks ensure service availability
- Each service can be scaled or restarted independently
