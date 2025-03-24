FROM nginx:alpine-slim

# Set environment variables for templating
ENV DOOM_API_HOST=doom-game
ENV DOOM_API_PORT=8000
ENV NOVNC_HOST=novnc
ENV NOVNC_PORT=6080

# Copy all web files to nginx serving directory
COPY . /usr/share/nginx/html/

# Copy our custom nginx configuration with environment variable support
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY entrypoint.sh /entrypoint.sh
COPY block-internet.sh /block-internet.sh

# Update Alpine packages to latest versions
RUN apk update && apk upgrade

# Create non-root user for improved security
RUN addgroup -S doomweb && adduser -S -G doomweb doomweb \
    && chmod +x /entrypoint.sh \
    && chmod +x /block-internet.sh \
    && chown -R doomweb:doomweb /usr/share/nginx/html \
    # We still need root for port 80, but we'll drop privileges where possible
    && chmod 755 /entrypoint.sh /block-internet.sh

# Expose web server port
EXPOSE 80

# Start nginx with our custom entrypoint
ENTRYPOINT ["/entrypoint.sh"]
