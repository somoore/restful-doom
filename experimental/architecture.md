# RESTful DOOM Microservices Architecture

## Overview

The RESTful DOOM project is being restructured from a monolithic container into a microservices architecture to improve reliability, maintainability, and scalability. This document outlines the proposed architecture, the components, and how they interact with each other.

## Current Architecture Issues

The monolithic container has been experiencing segmentation faults due to:

- Multiple tightly coupled processes sharing resources
- Complex restart mechanisms
- Interdependent components failing together
- Difficulty in isolating and debugging issues
- Resource contention between processes

## Proposed Microservices Architecture

### Container Structure

![RESTful DOOM Microservices Architecture](https://mermaid.ink/img/pako:eNplkU1vgzAMhv-KlRN8aNW2Aw5TpW3atKnVtHXSdgk0tFXJB0qCqir_fSFsGtvw-fWTF9vxxF0sEbnkvqjKCLQP0EGJGlRZBw2_baDEkYzWJYnuCKDEFIFB1-CuR6CnE4z7-xMRw-M-GhXJLNrHSbyeR7v5YRRN53P2D0SjuF9OWRwn8Z5FLImnaZQ8xZv5GWUIJ0bIBhSFrfYayEthKl4qpnStOGMNPJtqAA2b-BuKDKUtyj79mZa8JlGZ0Sd0YlCZWZdSOXXBMmUNBaXM0TqaK0oGjpfeFkFCl1Bl0zYM-hVrRu_vKO-EKg0OU1C14aJBvymEqbfVJWuuVIkUZqzRThvM4pZV16aWwBR_KMxfRe9E-dtA06XRjfpCWyuQu9lA6EsLJQpwlx5KqzzoVrJlVhM3dV8r72Hgd2A4B-2z8oK0e2Xtut4Ug1bVunSu_wBUn6AV?type=png)

1. **DOOM Game Container**
   - Core game engine with RESTful API
   - Responsibilities:
     - Game state management
     - API endpoint handling
     - Game logic

2. **X11 Display Container**
   - Virtual display server
   - Responsibilities:
     - Running Xvfb virtual framebuffer
     - Window management with Fluxbox
     - Providing X11 socket for other containers

3. **VNC Server Container**
   - Screen sharing service
   - Responsibilities:
     - Connecting to X11 display
     - Providing VNC access to the game screen
     - Handling VNC authentication

4. **Web UI Container**
   - Static file web server
   - Responsibilities:
     - Serving HTML, CSS, and JavaScript files
     - Providing user interface elements
     - Dashboard and control panels

5. **noVNC Container**
   - Web-based VNC client
   - Responsibilities:
     - Translating VNC to WebSocket protocol
     - Browser-based game viewing

### Communication Pathways

- **Shared Volumes**:
  - X11 socket volume: Shared between X11, DOOM, and VNC containers
  - WAD files volume: Shared for game data

- **Network Communication**:
  - Internal Docker network for container-to-container communication
  - Exposed ports for external access

### Port Allocation

- Each container uses fixed internal ports for simplicity
- Host port mapping can still be dynamic
- Port conflicts are eliminated between services

## Advantages of this Architecture

1. **Improved Stability**:
   - When one component crashes, others continue to function
   - Segmentation faults in DOOM don't affect web interfaces

2. **Better Resource Isolation**:
   - Each component has dedicated resources
   - CPU, memory, and I/O contention is reduced

3. **Simplified Debugging**:
   - Container-specific logs
   - Easier to identify problematic components
   - Can restart individual services

4. **Scalability**:
   - Can run multiple game instances with shared web UI
   - Ability to scale components independently

5. **Maintainability**:
   - Smaller, focused codebases per container
   - Easier to update individual components
   - Separation of concerns

## Deployment

The system will be orchestrated using Docker Compose, allowing for simple local deployment and testing while maintaining the benefits of the microservices architecture.
