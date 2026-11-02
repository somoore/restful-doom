#!/usr/bin/env python3
"""End-to-end regression checks for the REST control plane.

Requires a built ``src/restful-doom`` and a Doom IWAD (``DOOM_WAD`` or
``capsule/app/*.WAD``). Exits 0 with a "skipped" message when those are absent,
so it is safe to run anywhere.

    DOOM_WAD=/path/DOOM1.WAD python3 tests/api_smoke.py
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("RESTFUL_DOOM_BIN", ROOT / "src" / "restful-doom"))


def find_wad() -> Path | None:
    override = os.environ.get("DOOM_WAD")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    for pattern in ("*.wad", "*.WAD"):
        matches = sorted((ROOT / "capsule" / "app").glob(pattern))
        if matches:
            return matches[0]
    return None


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Server:
    def __init__(self, wad: Path, *, bind: str | None = None, token: str | None = None):
        self.port = free_port()
        env = dict(
            os.environ,
            HOME=tempfile.mkdtemp(),
            SDL_VIDEODRIVER="dummy",
            SDL_AUDIODRIVER="dummy",
        )
        if bind:
            env["DOOM_API_BIND"] = bind
        if token:
            env["DOOM_API_TOKEN"] = token
        self.process = subprocess.Popen(
            [
                str(BINARY), "-iwad", str(wad), "-warp", "1", "1", "-skill", "3",
                "-nosound", "-nomusic", "-apiport", str(self.port),
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(150):
            if self.process.poll() is not None:
                raise RuntimeError("restful-doom exited during startup")
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("restful-doom did not open the API port")

    def request(self, raw: str, host: str = "127.0.0.1") -> bytes:
        connection = socket.create_connection((host, self.port), timeout=3)
        connection.sendall(raw.encode())
        data = b""
        while True:
            try:
                chunk = connection.recv(65536)
            except (socket.timeout, ConnectionResetError):
                break
            if not chunk:
                break
            data += chunk
        connection.close()
        return data

    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


failures = 0


def check(name: str, condition: bool, detail: object = b"") -> None:
    global failures
    if condition:
        print(f"PASS {name}")
    else:
        failures += 1
        print(f"FAIL {name} :: {detail!r}")


def status(response: bytes, code: int) -> bool:
    return f" {code}\r\n".encode() in response


def json_body(response: bytes) -> dict:
    return json.loads(response.split(b"\r\n\r\n", 1)[1])


def smoke(wad: Path) -> None:
    server = Server(wad)
    try:
        response = server.request("GET /api/world HTTP/1.0\r\n\r\n")
        check("GET /api/world", status(response, 200), response)
        check("no wildcard CORS", b"Access-Control-Allow-Origin: *" not in response, response)

        response = server.request("GET /" + "A" * 300 + " HTTP/1.0\r\n\r\n")
        check("over-long path rejected", status(response, 400) and server.alive(), response)
        response = server.request("A" * 30 + " / HTTP/1.0\r\n\r\n")
        check("over-long method rejected", status(response, 400) and server.alive(), response)
        response = server.request("GET /" + "A" * 1100 + " HTTP/1.0\r\n\r\n")
        check("oversized request rejected", (status(response, 400) or status(response, 413)) and server.alive(), response)

        socket.create_connection(("127.0.0.1", server.port), timeout=3).close()
        time.sleep(0.05)
        response = server.request("GET /api/world HTTP/1.0\r\n\r\n")
        check("empty connection does not poison the socket set", status(response, 200) and server.alive(), response)

        for path in ("/api/world/doors/999999", "/api/world/doors/-1"):
            response = server.request(f"GET {path} HTTP/1.0\r\n\r\n")
            check(f"door bounds {path}", status(response, 404) and server.alive(), response)

        before = json_body(server.request("GET /api/player HTTP/1.0\r\n\r\n"))
        body = '{"weapon":10}'
        response = server.request(f"PATCH /api/player HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n{body}")
        check("weapon index rejected", status(response, 400) and server.alive(), response)
        after = json_body(server.request("GET /api/player HTTP/1.0\r\n\r\n"))
        check("weapon OOB left ammo untouched", before["ammo"] == after["ammo"], (before["ammo"], after["ammo"]))

        body = '{"ammo":100000,"amount":16777216}'
        response = server.request(f"PATCH /api/player HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n{body}")
        check("ammo index rejected", status(response, 400) and server.alive(), response)
        body = '{"ammo":1,"amount":12}'
        response = server.request(f"PATCH /api/player HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n{body}")
        check("valid ammo accepted", status(response, 200) and server.alive(), response)

        for body, name in (
            ('{"skill":9}', "skill"),
            ('{"map":0}', "map"),
            ('{"episode":0}', "episode"),
        ):
            response = server.request(f"PATCH /api/world HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n{body}")
            check(f"{name} out of range rejected", status(response, 400) and server.alive(), response)

        for body, name in (
            ("{}", "missing object fields"),
            ("{", "malformed JSON"),
        ):
            response = server.request(f"POST /api/world/objects HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n{body}")
            check(f"{name} rejected", status(response, 400) and server.alive(), response)

        body = json.dumps({"text": "M" * 600})
        response = server.request(f"POST /api/message HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n{body}")
        check("oversized HUD message handled", (status(response, 201) or status(response, 400)) and server.alive(), response)
    finally:
        server.stop()


def auth(wad: Path) -> None:
    token = "regression-token"
    server = Server(wad, bind="0.0.0.0", token=token)
    try:
        check("read works off-loopback", status(server.request("GET /api/world HTTP/1.0\r\n\r\n"), 200))
        patch_body = '{"weapon":1}'
        patch = (
            "PATCH /api/player HTTP/1.0\r\n"
            f"Content-Length: {len(patch_body)}\r\n\r\n{patch_body}"
        )
        check("mutation without token refused", status(server.request(patch), 401))
        with_token = (
            "PATCH /api/player HTTP/1.0\r\n"
            f"Authorization: Bearer {token}\r\nContent-Length: {len(patch_body)}\r\n\r\n"
            f"{patch_body}"
        )
        check("mutation with token accepted", status(server.request(with_token), 200))
        bad = with_token.replace(token, "wrong")
        check("mutation with wrong token refused", status(server.request(bad), 401))
    finally:
        server.stop()


def main() -> int:
    wad = find_wad()
    if wad is None:
        print("skipped: set DOOM_WAD or add capsule/app/*.WAD")
        return 0
    if not BINARY.is_file():
        print(f"skipped: {BINARY} is not built")
        return 0

    smoke(wad)
    auth(wad)
    lan_ip = os.environ.get("DOOM_TEST_LAN_IP")
    if lan_ip:
        server = Server(wad)
        try:
            response = server.request("GET /api/world HTTP/1.0\r\n\r\n", host=lan_ip)
            check("off-host peer dropped in loopback mode", not status(response, 200), response)
        finally:
            server.stop()

    print("ALL PASS" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
