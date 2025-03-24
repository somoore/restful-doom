FROM nginx:alpine

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

RUN chmod +x /entrypoint.sh \
    && chmod +x /block-internet.sh

# Expose web server port
EXPOSE 80

# Start nginx with our custom entrypoint
ENTRYPOINT ["/entrypoint.sh"]
