#!/bin/sh
set -e

# Apply internet blocking rules first
echo "Applying internet blocking rules..."
/block-internet.sh

# Replace environment variables in the static HTML files
echo "Configuring web UI with environment variables..."
echo "DOOM API: $DOOM_API_HOST:$DOOM_API_PORT"
echo "noVNC: $NOVNC_HOST:$NOVNC_PORT"

# Update the API endpoint URLs in the JavaScript files
for file in /usr/share/nginx/html/*.js /usr/share/nginx/html/*.html; do
  if [ -f "$file" ]; then
    # Replace API URLs
    sed -i "s|http://localhost:8000|http://$DOOM_API_HOST:$DOOM_API_PORT|g" "$file"
    sed -i "s|http://127.0.0.1:8000|http://$DOOM_API_HOST:$DOOM_API_PORT|g" "$file"
    # Replace noVNC URLs
    sed -i "s|http://localhost:6080|http://$NOVNC_HOST:$NOVNC_PORT|g" "$file"
    sed -i "s|http://127.0.0.1:6080|http://$NOVNC_HOST:$NOVNC_PORT|g" "$file"
  fi
done

# Generate port information page
cat > /usr/share/nginx/html/port-info.html << EOF
<!DOCTYPE html>
<html>
<head>
  <title>RESTful DOOM Microservices</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #333; color: #fff; padding: 20px; }
    .container { max-width: 800px; margin: 0 auto; background-color: #444; padding: 20px; border-radius: 5px; }
    h1 { color: #ff6600; }
    .port-info { margin-bottom: 20px; padding: 10px; background-color: #555; border-radius: 3px; }
    .port-number { font-weight: bold; color: #33ff33; }
    a { color: #33ccff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .button { display: inline-block; background-color: #ff6600; color: white; padding: 8px 16px; 
             border-radius: 4px; margin: 10px 5px 10px 0; text-decoration: none; }
    .button:hover { background-color: #ff8800; }
  </style>
</head>
<body>
  <div class="container">
    <h1>RESTful DOOM Microservices</h1>
    <div class="port-info">
      <h2>API Endpoints</h2>
      <p>API Base URL: <span class="port-number">http://127.0.0.1:$DOOM_API_PORT</span></p>
      <p>Available Endpoints:</p>
      <ul>
        <li><a href="http://127.0.0.1:$DOOM_API_PORT/api/player" target="_blank">Player Information</a></li>
        <li><a href="http://127.0.0.1:$DOOM_API_PORT/api/world/objects" target="_blank">World Objects</a></li>
        <li><a href="http://127.0.0.1:$DOOM_API_PORT/api/things" target="_blank">Things (alias for world/objects)</a></li>
        <li><a href="http://127.0.0.1:$DOOM_API_PORT/docs.html" target="_blank">API Documentation</a></li>
      </ul>
      
      <h2>User Interfaces</h2>
      <p><a href="/play-doom.html" class="button">Launch Page</a></p>
      <p><a href="/doom-dashboard.html" class="button">Dashboard</a></p>
      
      <h2>VNC Access</h2>
      <p>Direct VNC: <span class="port-number">127.0.0.1:$VNC_PORT</span> (password: password)</p>
      <p>Web VNC: <a href="http://127.0.0.1:$NOVNC_PORT/vnc.html?autoconnect=true&view_only=0" target="_blank" class="button">Launch Web VNC</a></p>
    </div>
    <p>These services are now running in separate containers for improved stability.</p>
  </div>
</body>
</html>
EOF

# Generate index.html that redirects to port-info.html
cat > /usr/share/nginx/html/index.html << EOF
<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="refresh" content="0; url=port-info.html">
  <title>Redirecting to Port Information</title>
</head>
<body>
  <p>Redirecting to <a href="port-info.html">Port Information Page</a>...</p>
</body>
</html>
EOF

# Start nginx
echo "Starting web server..."
exec nginx -g 'daemon off;'
