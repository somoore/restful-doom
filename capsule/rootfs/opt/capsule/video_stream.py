#!/usr/bin/env python3
"""Tiny MJPEG streamer for the SDL/Xvfb Doom window."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Sequence


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RESTful Doom Live Video</title>
  <style>
    html, body {
      margin: 0;
      min-height: 100%;
      background: #050506;
      color: #f4f1e8;
      font: 14px/1.4 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    body {
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 10px 14px;
      border-bottom: 1px solid #242424;
      background: #111112;
    }
    main {
      display: grid;
      place-items: center;
      min-height: calc(100vh - 43px);
      overflow: hidden;
    }
    img {
      width: min(100vw, 1280px);
      height: auto;
      image-rendering: pixelated;
      background: #000;
    }
    a {
      color: #f4f1e8;
    }
  </style>
</head>
<body>
  <header>
    <strong>RESTful Doom live video</strong>
    <span><a href="/frame">single frame</a> / <a href="/health">health</a></span>
  </header>
  <main>
    <img src="/stream" alt="Live rendered Doom framebuffer">
  </main>
</body>
</html>
"""


class VideoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        display: str,
        geometry: str,
        fps: float,
        quality: int,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.display = display
        self.geometry = geometry
        self.fps = fps
        self.quality = quality
        self.import_command = _find_import_command()


class Handler(BaseHTTPRequestHandler):
    server: VideoServer

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_bytes(HTTPStatus.OK, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/health":
            self._health()
            return
        if self.path == "/frame":
            self._frame()
            return
        if self.path == "/stream":
            self._stream()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _health(self) -> None:
        body = json.dumps(
            {
                "status": "ok",
                "display": self.server.display,
                "geometry": self.server.geometry,
                "fps": self.server.fps,
                "quality": self.server.quality,
                "capture_command": self.server.import_command[0],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_bytes(HTTPStatus.OK, body, "application/json")

    def _frame(self) -> None:
        try:
            frame = capture_frame(
                self.server.import_command,
                self.server.display,
                self.server.geometry,
                self.server.quality,
            )
        except RuntimeError as exc:
            self._send_bytes(
                HTTPStatus.SERVICE_UNAVAILABLE,
                json.dumps({"status": "starting", "error": str(exc)}).encode("utf-8"),
                "application/json",
            )
            return
        self._send_bytes(HTTPStatus.OK, frame, "image/jpeg", cache=False)

    def _stream(self) -> None:
        boundary = b"doomframe"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=doomframe")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        min_interval = 1.0 / max(self.server.fps, 0.1)
        while True:
            started = time.monotonic()
            try:
                frame = capture_frame(
                    self.server.import_command,
                    self.server.display,
                    self.server.geometry,
                    self.server.quality,
                )
                self.wfile.write(b"--" + boundary + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            except RuntimeError:
                time.sleep(0.25)
                continue

            elapsed = time.monotonic() - started
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        cache: bool = True,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def capture_frame(command: Sequence[str], display: str, geometry: str, quality: int) -> bytes:
    argv = [
        *command,
        "-display",
        display,
        "-window",
        "root",
        "-resize",
        geometry,
        "-quality",
        str(quality),
        "jpg:-",
    ]
    completed = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=2.0,
    )
    if completed.returncode != 0 or not completed.stdout:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"capture command exited with {completed.returncode}")
    return completed.stdout


def _find_import_command() -> list[str]:
    if shutil.which("import"):
        return ["import"]
    if shutil.which("magick"):
        return ["magick", "import"]
    raise RuntimeError("ImageMagick import command not found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6080)
    parser.add_argument("--display", default=":99")
    parser.add_argument("--geometry", default="640x480")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--quality", type=int, default=70)
    args = parser.parse_args()

    server = VideoServer(
        (args.host, args.port),
        Handler,
        display=args.display,
        geometry=args.geometry,
        fps=args.fps,
        quality=args.quality,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
