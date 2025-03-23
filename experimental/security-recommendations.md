# Security Recommendations for RESTful DOOM

⚠️ **IMPORTANT: This project is designed for LOCAL DEVELOPMENT AND TESTING ONLY** ⚠️

This document outlines security considerations and recommendations for the RESTful DOOM project. While the project is intended for local development only, following these guidelines helps prevent accidental misuse and ensures safe development practices.

## Current Security Profile

### Container Security

#### Elevated Privileges
- **x11-server** runs with `privileged: true`
  - Required for X11 display functionality
  - Grants full access to host system
  - Cannot be removed due to X11 requirements
  
- **doom-game** uses `ipc: host`
  - Shares host's IPC namespace
  - Required for game engine functionality
  - Consider documenting specific IPC requirements

#### Resource Management
- No current resource limits on containers
- Potential for resource exhaustion
- Recommended limits:
  ```yaml
  deploy:
    resources:
      limits:
        cpus: '1.0'
        memory: 512M
      reservations:
        cpus: '0.25'
        memory: 256M
  ```

### Network Security

#### Port Exposure
- Following ports are exposed to host:
  - 8000 (API)
  - 5900 (VNC)
  - 6080 (noVNC)
  - 8080 (Web UI)
- No authentication required
- HTTP only (no HTTPS)

#### CORS Configuration
- Current nginx configuration allows all origins (`*`)
- Necessary for local development
- Should never be used in production

### File System Security

#### Volume Mounts
- WAD file mounted read-only ✅
- Shared memory (`/dev/shm`) mounted in multiple containers
  - Required for X11 and game functionality
  - Potential attack surface if compromised

#### File Permissions
- Container processes run as root
- No user namespace restrictions
- Consider adding non-root users where possible

## Recommended Improvements

### 1. Environment Validation
Add to `start-doom.sh`:
```bash
if [ -n "$PRODUCTION" ] || [ -n "$PROD" ]; then
    echo "ERROR: This project is not meant for production use!"
    echo "It contains several security issues that make it unsuitable for production deployment."
    exit 1
fi
```

### 2. Port Conflict Detection
Add to `start-doom.sh`:
```bash
for port in ${HOST_API_PORT:-8000} ${HOST_VNC_PORT:-5900} ${HOST_NOVNC_PORT:-6080} ${HOST_WEB_PORT:-8080}; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "ERROR: Port $port is already in use!"
        echo "Please make sure no other services are running on the required ports."
        exit 1
    fi
done
```

### 3. Network Isolation
Add to `docker-compose.yml`:
```yaml
networks:
  doom-network:
    driver: bridge
    internal: false  # Set to true for complete isolation
    name: doom-network
```

### 4. Resource Limits
Add to each service in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '1.0'
      memory: 512M
    reservations:
      cpus: '0.25'
      memory: 256M
```

## Development Guidelines

1. **Never Deploy to Production**
   - This project contains intentional security compromises
   - Not suitable for internet-facing deployment
   - Keep all ports local to development machine

2. **Safe Development Practices**
   - Always run in a development environment
   - Don't expose ports to external networks
   - Keep Docker and dependencies updated
   - Don't store sensitive data in the project directory

3. **Testing Considerations**
   - Use load-test.sh responsibly
   - Monitor system resources during testing
   - Clean up containers and volumes after testing

## Why These Security Compromises Exist

1. **X11 Requirements**
   - X11 needs privileged container access
   - Shared memory access required
   - No easy way to sandbox X11 display server

2. **Game Engine Requirements**
   - DOOM engine needs direct hardware access
   - IPC required for game functionality
   - Shared memory needed for performance

3. **Development Convenience**
   - CORS allows easy local development
   - Open ports enable quick testing
   - No HTTPS simplifies local setup

## Conclusion

While these security compromises are acceptable for local development, they make the project unsuitable for production use. Always follow the development guidelines and never deploy this project in a production environment or expose it to the internet.

Remember: The goal of these security recommendations is not to make the project production-ready, but to prevent accidental misuse and ensure safe development practices.
