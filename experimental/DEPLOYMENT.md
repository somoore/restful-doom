# RESTful DOOM Microservices: Deployment Guide

## Overview

This document provides step-by-step instructions for deploying the RESTful DOOM microservices architecture. The system is containerized using Docker and orchestrated with Docker Compose.

## Prerequisites

- Docker Engine (20.10.0 or later)
- Docker Compose (v2.0.0 or later)
- A DOOM WAD file (DOOM.WAD, DOOM2.WAD, etc.)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/jeff-1amstudios/restful-doom.git
   cd restful-doom
   ```

2. Place your WAD file(s) in the `experimental/docker/wad/` directory:
   ```bash
   cp /path/to/your/DOOM.WAD experimental/docker/wad/
   ```

3. Run the start script:
   ```bash
   chmod +x experimental/scripts/start-doom.sh
   ./experimental/scripts/start-doom.sh
   ```

4. Access the services:
   - Web UI: http://localhost:8080
   - noVNC: http://localhost:6080/vnc.html
   - REST API: http://localhost:8000/api

## Configuration

You can customize the deployment by setting the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| HOST_API_PORT | Port for the REST API | 8000 |
| HOST_VNC_PORT | Port for the VNC server | 5900 |
| HOST_NOVNC_PORT | Port for the noVNC web client | 6080 |
| HOST_WEB_PORT | Port for the Web UI | 8080 |
| WAD_FILENAME | Name of the WAD file to use | DOOM.WAD |
| WAD_PATH | Path to the directory containing WAD files | ./experimental/docker/wad |

Example:
```bash
HOST_API_PORT=9000 WAD_FILENAME=DOOM2.WAD ./experimental/scripts/start-doom.sh
```

## Microservices Architecture

The system consists of the following containers:

1. **X11 Server**: Provides a virtual display for DOOM
2. **DOOM Game**: Runs the DOOM engine and REST API
3. **VNC Server**: Captures the X11 display for remote viewing
4. **noVNC**: Provides a web interface to the VNC server
5. **Web UI**: Serves the web interface for controlling DOOM

## Validation and Testing

After deployment, verify that all services are running correctly:

```bash
chmod +x experimental/scripts/test-microservices.sh
./experimental/scripts/test-microservices.sh
```

To perform stress testing on the API:

```bash
chmod +x experimental/scripts/load-test.sh
./experimental/scripts/load-test.sh --endpoint /api/things --concurrency 10 --requests 1000
```

## Troubleshooting

### Common Issues

1. **Docker Compose errors**:
   - Ensure Docker and Docker Compose are installed and running
   - Check that you have sufficient permissions to run Docker commands

2. **WAD file issues**:
   - Verify that your WAD file is correctly placed in the `experimental/docker/wad/` directory
   - Check that the WAD filename matches the WAD_FILENAME environment variable

3. **Port conflicts**:
   - If any of the ports are already in use, change them using the environment variables

### Container-Specific Issues

1. **X11 Server**:
   - Check logs: `docker logs experimental-x11-server-1`
   - Verify that the X11 socket is properly mounted

2. **DOOM Game**:
   - Check logs: `docker logs experimental-doom-game-1`
   - Verify that the game can connect to the X11 display

3. **VNC Server**:
   - Check logs: `docker logs experimental-vnc-server-1`
   - Verify VNC password is correctly set

4. **noVNC**:
   - Check logs: `docker logs experimental-novnc-1`
   - Verify it can connect to the VNC server

5. **Web UI**:
   - Check logs: `docker logs experimental-web-ui-1`
   - Verify API connectivity from the browser console

## Advanced Configuration

For advanced users, you can modify the Docker Compose file directly:

```bash
nano experimental/docker/docker-compose.yml
```

Changes to the Dockerfiles or entrypoint scripts should be made in their respective directories:

```
experimental/docker/
├── doom/
├── novnc/
├── vnc/
├── web-ui/
└── x11/
```

## Maintenance

To stop all containers:

```bash
cd experimental
docker compose down
```

To rebuild and restart a specific service:

```bash
cd experimental
docker compose up -d --build doom-game
```

## Security Considerations

- The VNC server is protected with a password
- The API has no authentication by default
- For production use, consider adding a reverse proxy with proper authentication
