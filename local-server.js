// Simple file server to host HTML files alongside the DOOM API
const http = require('http');
const fs = require('fs');
const path = require('path');

const server = http.createServer((req, res) => {
  // Parse URL
  const url = req.url === '/' ? '/play-doom.html' : req.url;
  const filePath = path.join(__dirname, url);
  const extname = String(path.extname(filePath)).toLowerCase();
  
  // Map file extensions to MIME types
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpg',
    '.gif': 'image/gif',
  };

  const contentType = mimeTypes[extname] || 'application/octet-stream';

  // Read file
  fs.readFile(filePath, (error, content) => {
    if (error) {
      if(error.code === 'ENOENT') {
        // If file doesn't exist, try to proxy to the API if it's an API request
        if (url.startsWith('/api/')) {
          // Proxy to DOOM API
          const options = {
            hostname: 'localhost',
            port: 8000,
            path: url,
            method: req.method,
            headers: req.headers
          };
          
          const apiReq = http.request(options, (apiRes) => {
            res.writeHead(apiRes.statusCode, apiRes.headers);
            apiRes.pipe(res);
          });
          
          apiReq.on('error', (e) => {
            res.writeHead(500);
            res.end(`Error proxying to API: ${e.message}`);
          });
          
          req.pipe(apiReq);
        } else {
          res.writeHead(404);
          res.end('File not found');
        }
      } else {
        res.writeHead(500);
        res.end(`Server Error: ${error.code}`);
      }
    } else {
      // If successful, set headers and send content
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content, 'utf-8');
    }
  });
});

const PORT = 8080;
server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
  console.log(`This server hosts the static HTML files and proxies API requests to the DOOM API`);
  console.log(`Open http://localhost:${PORT}/play-doom.html to get started`);
});
