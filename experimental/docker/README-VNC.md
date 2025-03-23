# RESTful DOOM with VNC and Web Browser Support

This extension to the [RESTful DOOM](https://github.com/jeff-1amstudios/restful-doom) project adds VNC and browser-based visualization capabilities, allowing you to both control DOOM through the REST API and view the game rendering in real-time - either in a web browser or using a VNC client.

## Features

- **RESTful API**: Control DOOM through HTTP endpoints (from the original project)
- **VNC Server**: View the game through any VNC client
- **Browser-Based Viewing**: Play DOOM directly in your web browser without installing any client
- **Custom WAD Support**: Use your own WAD files
- **Dockerized**: Everything packaged in a container for easy deployment
- **API + Visual Dashboard**: Combined interface to see both the API and game rendering

## Quick Start

### Option 1: Using the Automated Startup Script (Recommended)

1. Clone this repository:
   ```
   git clone https://github.com/jeff-1amstudios/restful-doom
   cd restful-doom
   ```

2. Place your DOOM WAD file in the repository folder (named `DOOM.WAD`)

3. Run the startup script:
   ```
   ./start-doom.sh
   ```
   This script will automatically:
   - Check for DOOM.WAD
   - Build the Docker image if needed
   - Start the container
   - Open the game in your browser

### Option 2: Manual Setup

1. Clone this repository:
   ```
   git clone https://github.com/jeff-1amstudios/restful-doom
   cd restful-doom
   ```

2. Place your DOOM WAD file in the repository folder (named `DOOM.WAD`)

3. Build the Docker image:
   ```
   docker build -t restful-doom:combined -f Dockerfile.combined .
   ```

4. Run the container:
   ```
   docker run -d --rm -p 8000:8000 -p 5900:5900 -p 6080:6080 --name restful-doom-container restful-doom:combined
   ```

5. Start the local file server to serve HTML files and proxy API requests:
   ```
   node local-server.js
   ```

6. Open a web browser and navigate to:
   ```
   http://localhost:8080/play-doom.html  # Main launch page with options
   ```
   From there you can access:
   - VNC for playing DOOM with keyboard controls
   - Dashboard for controlling DOOM with API buttons
   - API documentation

## Connecting to DOOM

### Browser-Based Viewing (Recommended)

1. Open `play-doom.html` in your browser
2. Click inside the game display area to focus it
3. Press F8 to access the noVNC menu and ensure "Enable keyboard" is checked
4. Use the keyboard controls:
   - Arrow keys to move
   - Ctrl to fire
   - Alt to strafe
   - Spacebar to open doors/use

### VNC Client (Alternative)

1. Connect to `localhost:5900` using any VNC client
2. Use password: `password`

## Custom WAD File Support

This container supports using custom WAD files, allowing you to play different DOOM games, mods, or custom levels.

### How to Use Custom WADs:

1. Rename your custom WAD file to `DOOM.WAD` and place it in the project directory

2. Build the Docker image:
   ```
   docker build -t restful-doom:combined -f Dockerfile.combined .
   ```

3. Run the container as normal:
   ```
   docker run -d --rm -p 8000:8000 -p 5900:5900 -p 6080:6080 --name restful-doom-container restful-doom:combined
   ```

4. Access the game through any of the methods described above

### Compatibility

The container should work with most DOOM-engine WAD files, including:
- DOOM shareware/registered/ultimate (doom.wad, doom1.wad)
- DOOM II (doom2.wad)
- Final DOOM (plutonia.wad, tnt.wad)
- Many community-created WADs and mods

## Using the RESTful API

The RESTful API runs on port 8000. Here are the main endpoints:

- `GET /api/world` - Get information about the current game world
- `GET /api/player` - Get player status
- `GET /api/things` - List all things in the game world
- `POST /api/player/actions` - Perform player actions

Example API usage:
```
# Move forward
curl -X POST http://localhost:8000/api/player/actions -d '{"action":"move", "direction":"forward"}'

# Fire weapon
curl -X POST http://localhost:8000/api/player/actions -d '{"action":"shoot"}'
```

## Dashboard

The `doom-dashboard.html` provides a combined interface with:
- The game rendering on the left
- API visualization on the right

This is particularly useful for developers who want to see how API calls affect the game in real-time.

## Technical Details

### Modifications from Original Project

This fork enhances the original [RESTful DOOM](https://github.com/jeff-1amstudios/restful-doom) project with the following changes:

1. **Visual Output**:
   - Added X11 rendering within the container
   - Integrated VNC server for remote viewing
   - Added noVNC for browser-based access

2. **Dockerfile Changes**:
   - Modified binding to use all interfaces (`0.0.0.0`) instead of just localhost
   - Added X11, VNC, and noVNC dependencies
   - Configured TigerVNC and websockify
   - Created startup scripts for all services
   - Added custom WAD file support

3. **Code Modifications**:
   - Updated network binding in `api.c`
   - Modified IWAD handling in `d_iwad.c` to be more flexible
   - Added debugging output

4. **HTML Interfaces**:
   - Created documentation and guide pages
   - Built browser integration for game viewing
   - Developed a combined API + visual dashboard

### Port Mapping

- `8000`: RESTful API
- `5900`: VNC Server
- `6080`: Web-based VNC (noVNC)

### Dependencies

- Docker
- A DOOM WAD file (shareware or full version)
- Web browser or VNC client

## Recent Fixes and Improvements

The following improvements have been made to enhance stability and fix issues:

1. **Improved Container Stability**:
   - Added screen package to run DOOM in a detached session
   - VNC server now stays running even if DOOM crashes
   - Removed the `-window` flag to run DOOM in fullscreen mode

2. **Fixed VNC Connection Issues**:
   - Simplified VNC parameters for better browser compatibility
   - Used `-forever -shared -rfbport 5900` for reliable connections
   - Set explicit password for consistent authentication

3. **Resolved Static File Serving**:
   - Created a Node.js local server (`local-server.js`) to serve HTML files
   - Server proxies API requests to the RESTful DOOM API
   - Updated HTML links to point to the local server (port 8080)

4. **Fixed Dashboard Controls**:

## Troubleshooting

### Black Screen in VNC
If you see a black screen in VNC, ensure:
1. The container is running with proper X11 configuration
2. The DOOM process has rendering enabled (no `-nodraw` flag)
3. The environment variables are set correctly:
   ```bash
   export DISPLAY=:1
   export SDL_VIDEODRIVER=x11
   export SDL_AUDIODRIVER=dummy
   export LIBGL_ALWAYS_SOFTWARE=1
   export SDL_RENDER_DRIVER=software
   ```

### API Connection Issues
If you can't connect to the API:
1. Check if the container ports are mapped correctly (8000, 5900, 6080)
2. Verify the API is listening on 0.0.0.0 instead of localhost
3. Check the container logs for any binding errors

### Game Crashes
If DOOM crashes frequently:
1. Use the memory safety environment variables:
   ```bash
   export MALLOC_CHECK_=2
   export MALLOC_PERTURB_=0
   export GLIBCXX_FORCE_NEW=1
   ```
2. Start with minimal game parameters:
   ```bash
   ./restful-doom -iwad DOOM.WAD -skill 1 -episode 1 -warp 1
   ```
   - Corrected API payload format in all HTML files
   - Updated all button handlers to send the proper JSON structure
   - Enhanced error handling and feedback in UI

These fixes ensure that both the VNC viewing and API control functionalities work reliably together.

## Troubleshooting

### Common Issues and Solutions

- **Keyboard Input Issues**: 
  - Make sure to click inside the game area to focus it
  - Press F8 to open the noVNC menu and verify 'Enable keyboard' is checked
  - For best results, use play-doom.html in a dedicated browser tab instead of embedded iframes
  - Some browser security settings may block keyboard input in iframes

- **API Controls Not Working**:
  - The API expects `type` parameter (not `action` and `direction`):
    - Use `{type:'forward', amount:10}` instead of `{action:'move', direction:'forward'}`
    - Use `{type:'turn-left', amount:10}` instead of `{action:'turn', direction:'left'}`
    - Use `{type:'shoot'}` instead of `{action:'attack'}`
  - If you see "Error: Failed to fetch" in the dashboard, the API may not be allowing CORS
  - Network issues may prevent the dashboard from connecting to the API on port 8000
  - Check browser console for detailed error messages

- **Container Not Starting**: 
  - Ensure you have a valid DOOM.WAD file in the project directory
  - Verify that ports 8000, 5900, and 6080 aren't already in use
  - Check container logs with `docker logs restful-doom-container`

- **Segmentation Faults**:
  - If you see segmentation faults in logs, ensure the Docker image was built correctly
  - The startup sequence needs time between starting X server, window manager, and DOOM
  - This is already handled in the Dockerfile.combined

- **VNC Connection Issues**: 
  - Verify the container is running with `docker ps`
  - Try connecting with a native VNC client to port 5900 with password 'password'
  - For browser-based VNC, access http://localhost:6080/vnc.html?autoconnect=true&password=password&view_only=0
  - If the connection still fails, we now run DOOM in a screen session so the VNC server stays running even if DOOM crashes

- **Performance Issues**: 
  - Try using a native VNC client instead of browser-based viewing
  - Lower the quality settings in the noVNC menu (press F8)

- **Black Screen in VNC**:
  - Ensure the X server has time to start up
  - Check container logs for any errors
  - Try restarting the container

- **API Controls Not Working or 404 Errors**:
  - Make sure the local server is running with `node local-server.js`
  - Access the dashboard through http://localhost:8080/doom-dashboard.html
  - The local server will proxy your API requests to the DOOM container

### Debugging Tips

To see detailed logs:
```
docker logs restful-doom-container
```

To run in interactive mode (seeing all output):
```
docker run -it --rm -p 8000:8000 -p 5900:5900 -p 6080:6080 restful-doom:combined
```

## License

This project extends the original RESTful DOOM project, maintaining its original license.

## Docker Implementation Details

### Dockerfile Details

The `Dockerfile.combined` includes key features for reliable operation:

1. Proper timing for service startup:
   ```dockerfile
   # Create startup script with proper timing
   RUN echo '#!/bin/bash' > /app/start.sh && \
       echo 'export DISPLAY=:1' >> /app/start.sh && \
       echo 'Xvfb $DISPLAY -screen 0 800x600x24 &' >> /app/start.sh && \
       echo 'sleep 2' >> /app/start.sh && \
       # More startup with appropriate sleep intervals
   ```

2. VNC configuration for reliable remote access:
   ```dockerfile
   # VNC setup with proper authentication
   RUN mkdir -p /root/.vnc && \
       echo "password" | vncpasswd -f > /root/.vnc/passwd && \
       chmod 600 /root/.vnc/passwd
   ```

3. X11VNC optimizations for compatibility:
   ```dockerfile
   # x11vnc with optimized settings
   echo 'x11vnc -display $DISPLAY -forever -shared -repeat -noxdamage -notruecolor -rfbauth /root/.vnc/passwd &' >> /app/start.sh
   ```

These settings ensure reliable operation across different systems.

## Credits

- Original RESTful DOOM by [Jeff (1AM Studios)](https://github.com/jeff-1amstudios)
- VNC and browser integration by [Your Name or Organization]
- Based on [id Software](https://www.idsoftware.com/)'s DOOM
