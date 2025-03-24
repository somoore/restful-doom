# Security Recommendations for RESTful DOOM

⚠️ **IMPORTANT: This project is designed for LOCAL DEVELOPMENT AND TESTING ONLY** ⚠️

This document outlines security considerations and recommendations for the RESTful DOOM project. While the project is intended for local development only, following these guidelines helps prevent accidental misuse and ensures safe development practices.

## Security Checklist

### Container Security

#### Privilege and Capability Controls
- [x] Implement `no-new-privileges:true` for all non-X11 containers
- [x] Drop `NET_RAW` capability from all containers
- [x] Drop `NET_ADMIN` capability from all containers except web-ui
- [x] Configure minimal network permissions by default

#### Elevated Privileges Management
- [x] Document and justify `privileged: true` for x11-server
- [x] Document and justify `ipc: host` for doom-game
- [ ] Document specific IPC requirements for doom-game
- [ ] Investigate potential alternatives to privileged mode for X11

#### Resource Controls
- [x] Set CPU limits for all containers
  - [x] doom-game: 1.0 CPU, 512MB memory
  - [x] Other services: 0.5 CPU, 256MB memory
- [x] Set memory limits for all containers
- [x] Implement fair resource distribution
- [x] Prevent resource exhaustion attacks

### Network Security

#### Network Architecture
- [x] Implement dual-network architecture
  - [x] doom-internal network (172.20.0.0/16)
    - [x] Set `internal: true` to block outbound access
    - [x] Enable container-to-container communication
    - [x] Verify internal communication (API, VNC)
  - [x] doom-external network (172.21.0.0/16)
    - [x] Set `internal: false` for browser access
    - [x] Disable IP masquerading
    - [x] Verify localhost access

#### Internet Access Control
- [x] Implement iptables-based blocking
  - [x] Allow established connections
  - [x] Allow loopback traffic
  - [x] Allow Docker DNS access
  - [x] Allow internal network traffic
  - [x] Block all other outbound traffic
- [x] Configure web-ui container
  - [x] Add block-internet.sh to entrypoint
  - [x] Grant NET_ADMIN capability
  - [x] Verify blocking with ping tests

#### Network Access Controls
- [x] Configure service-specific network access
  - [x] x11-server: internal only
  - [x] doom-game: internal + external (API)
  - [x] vnc-server: internal + external (VNC)
  - [x] novnc: internal + external (browser)
  - [x] web-ui: internal + external (browser)

#### Port Security
- [x] Bind all ports to localhost (127.0.0.1)
  - [x] API: 8000
  - [x] VNC: 5900
  - [x] noVNC: 6080
  - [x] Web UI: 8080
- [x] Verify no external network access
- [x] Verify localhost-only HTTP is safe

#### CORS Configuration
- [x] Allow all origins for local development
- [ ] Add warning about production use
- [ ] Document CORS security implications

### File System Security

#### Volume Management
- [x] Mount WAD file as read-only
- [x] Document shared memory mounts
  - [x] Identify X11 requirements
  - [x] Identify game requirements
  - [x] Assess attack surface risks
    - X11 shared memory: Required for graphics but potential attack vector
    - VNC server: Exposes display but limited to localhost
    - API endpoints: Accessible but restricted to localhost
    - Container privileges: Some containers require elevated access
  - [x] Document mitigation strategies
    - Strict port binding to 127.0.0.1 only
    - Network isolation using dual network architecture
    - Resource limits to prevent DoS
    - Automatic cleanup of stale processes
    - No authentication needed due to localhost-only access

#### Permission Controls
- [ ] Implement non-root users where possible
  - ⚠️ Requires X11 socket permissions
  - ⚠️ Affects nginx port binding
  - ⚠️ Impacts shared memory access
- [ ] Configure user namespace restrictions
  - ⚠️ May break X11 functionality
  - ⚠️ Affects container communication
- [ ] Review file permissions
  - ⚠️ WAD file must be readable
  - ⚠️ X11 socket permissions required
- [x] Document root usage requirements
  - x11-server: Required for display
  - web-ui: Required for port 80
  - doom-game: Required for shared memory
  - vnc-server: Required for display access

## Future Improvements

### Environment Safety
- [x] Add production environment check
  ```bash
  if [ -n "$PRODUCTION" ] || [ -n "$PROD" ]; then
      echo "ERROR: This project is not meant for production use!"
      exit 1
  fi
  ```

### Startup Safety
- [x] Add port conflict detection
  ```bash
  for port in 8000 5900 6080 8080; do
      if lsof -i :$port > /dev/null 2>&1; then
          echo "ERROR: Port $port is already in use!"
          exit 1
      fi
  done
  ```
- [x] Add automatic port cleanup
- [x] Add graceful shutdown handling

### Network Architecture
- [x] Implement dual networks
  - [x] doom-internal: Isolated internal network
  - [x] doom-external: Browser-accessible network
- [x] Configure network security
  - [x] Disable IP masquerading
  - [x] Bind to localhost only
  - [x] Block outbound internet

### Resource Management
- [x] Configure resource limits
  - [x] CPU limits
  - [x] Memory limits
  - [x] Custom limits per service

### Container Management
- [x] Process isolation
- [x] Version pinning
- [x] Log rotation
- [x] Restart policies

### File Handling
- [x] Add cleanup routines
  ```bash
  # Exit cleanup
  trap 'docker compose down -v' EXIT
  
  # Port and process cleanup
  cleanup() {
      # Kill processes holding ports
      pkill nc || true
      # Stop containers
      docker compose down -v
      # Force kill port processes
      for port in $PORTS; do
          fuser -k $port/tcp || true
      done
  }
  ```
- [x] Add port conflict resolution
- [x] Add container cleanup
- [x] Add process cleanup

### Verification Status
- [x] Network isolation verified
- [x] Internet blocking verified
- [x] Internal communication verified
- [x] Browser access verified
- [x] Resource limits verified
- [x] Container security verified
- [x] Port conflict handling verified
- [x] Cleanup routines verified
- [x] Graceful shutdown verified

### 7. Docker Image Security
- Use specific version tags for base images
- Implement multi-stage builds to minimize image size
- Scan images for vulnerabilities:
```bash
docker scan doom-game:latest
```

### 8. Health Monitoring
- Add comprehensive health checks:
```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:8000/api/health || exit 1"]
  interval: 10s
  timeout: 5s
  retries: 3
  start_period: 10s
```

### 9. Logging Security
- Implement log rotation
- Avoid logging sensitive information
- Add to docker-compose.yml:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
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

### 10. Environment Variable Handling
- Use .env file for development
- Never commit .env files
- Add to .gitignore:
```
.env
*.env
```

### 11. API Rate Limiting
Add to nginx.conf:
```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
location /api/ {
    limit_req zone=api_limit burst=20 nodelay;
}
```

### 12. Container User Management
- Create dedicated users for services:
```dockerfile
RUN groupadd -r doom && useradd -r -g doom -s /sbin/nologin doom
USER doom
```

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
