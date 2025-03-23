# RESTful-DOOM

An HTTP + JSON API hosted inside the 1993 DOOM engine!

![](http://1amstudios.com/img/restful-doom/header.jpg)

RESTful-DOOM is a version of Doom which hosts a RESTful API! The API allows you to query and manipulate various game objects with standard HTTP requests as the game runs.

There were a few challenges:

- Build an HTTP+JSON RESTful API server in C.
- Run the server code inside the Doom engine, without breaking the game loop.
- Figure out what kinds of things we can manipulate in the game world, and how to interact with them in memory to achieve the desired effect!

RESTFul-DOOM is built on top of the awesome [Chocolate Doom](https://github.com/chocolate-doom/chocolate-doom) project. I like this project because it aims to stick as close to the original experience as possible, while making it easy to compile and run on modern systems. This was only possible by building on top of their hard work!

### More details in blog post:
http://1amstudios.com/2017/08/01/restful-doom/

## API Spec

[API spec in RAML 1.0 format](https://github.com/jeff-1amstudios/restful-doom/blob/master/RAML/doom.raml)

## Build

### Building dependencies (needs to be run only once)

Takes care of building and configuring dependencies like SDL. Uses [chocpkg](https://github.com/chocolate-doom/chocpkg).
```
./configure-and-build.sh
```

### Compiling

Run `make` from the src (or root) directory. `src/restful-doom` will be created if the compile succeeds.

## Run

The DOOM engine is open source, but assets (art, maps etc) are not. You'll need to download an appropriate [WAD file](https://en.wikipedia.org/wiki/Doom_WAD) separately.

### Option 1: Using Docker (Recommended)

1. Place your DOOM WAD file in the repository folder (named `DOOM.WAD`)
2. Run the startup script:
```bash
./start-doom.sh
```

This will:
- Build the Docker image if needed
- Start the container with all necessary services
- Make the game accessible via:
  - VNC web interface: http://localhost:6080/vnc.html (password: password)
  - Direct VNC: localhost:5900 (password: password)
  - Web interface: http://localhost:8000/play-doom.html
  - API: http://localhost:8000/api/

### Option 2: Running Locally

To run restful-doom directly on your machine:
```bash
src/restful-doom -iwad <path/to/doom1.wad> -apiport 8000
```

See [README-VNC.md](README-VNC.md) for more details on VNC support and troubleshooting.

## Thanks!
[chocolate-doom](https://github.com/chocolate-doom/chocolate-doom) team  
[cJSON](https://github.com/DaveGamble/cJSON) - JSON parsing / generation  
[yuarel](https://github.com/jacketizer/libyuarel/) - URL parsing  
