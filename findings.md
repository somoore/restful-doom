# Security remediation — restful-doom

**Repository:** `/Users/scottmoore/github/restful-doom` @ `84103e8`
**Scanned:** 2026-11-02
**Inputs:** `/Users/scottmoore/github/.security-scan/restful-doom/` (`SUMMARY.txt`, raw scanner JSON,
`01-pm.md`, `02-appsec.md`, `03-pentest.md`, `04-devil.md`, `05-qa.md`).
**Tools:** bandit, semgrep, trivy (fs + config), hadolint, cargo-audit, gitleaks, plus manual review of
the first-party C REST surface, Python agent/MCP code, Rust gRPC bridge and capsule scripts. A live
instance of the built binary was exercised on loopback with a shareware IWAD.
**Team roles:** PM, AppSec, Pentester, Devil's Advocate, QA.

The scanner baseline reported 75 rows (mostly rule noise). The review team found the severe issues
were **not** scanner findings: they are in first-party C (`src/doom/api*.c`), which none of the
configured scanners parse. The Devil's Advocate verdicts were treated as authoritative; the
Pentester's reachability evidence decided exploitability.

**Outcome:** 16 fixed, 2 mitigated, 3 accepted risk, 1 deferred, 9 false positives. No secret was
found in the working tree or history.

## Findings

| ID | Severity | Status | Location | Issue | Fix | Verification |
|----|----------|--------|----------|-------|-----|--------------|
| C-01 (NEW-01, PEN-01, PEN-08) | **HIGH** | **FIXED** | `src/doom/api.c:128-129` → `src/doom/api_request.c` | Request line parsed with unbounded `sscanf("%s%s%s", …)` into `method[10]`/`protocol[10]`/global `path[100]`; `buffer[buffer_len]=0` wrote one byte past a full 1024-byte read. Pre-auth stack/global corruption. | Replaced with a bounded tokenizer that rejects over-width tokens and trailing garbage; terminate only when `recv_len < sizeof buffer`, otherwise return **413**; `snprintf` for `full_path`/`method`. Added `-fstack-protector-strong` + `-D_FORTIFY_SOURCE=2`. | `./tests/c/run.sh` (ASan/UBSan parser cases); `tests/api_smoke.py`: 300-byte path, 30-byte method, 1100-byte request all rejected with process alive. |
| C-02 (NEW-02, PEN-02) | **HIGH** | **FIXED** | `src/doom/api_player_controller.c:303,315` | Attacker integers indexed `player->weaponowned[val-1]` and `player->ammo[val]` with no range check → controlled 4-byte OOB write relative to `players[]`. | `API_WeaponNumberInRange` (1..NUMWEAPONS) and `API_IndexInRange` (0..NUMAMMO-1); reject with 400. Ammo amount must be non-negative. | `tests/api_smoke.py`: `{"weapon":10}` → 400 and ammo unchanged; `{"ammo":100000,…}` → 400; valid values still 200. |
| C-03 (PEN-05) | **MEDIUM** | **FIXED** | `src/doom/api_door_controller.c:111,125,35,81,132` | Door id from URL indexed `lines[]` unbounded (negative and >`numlines`); one-sided lines dereferenced `sides[sidenum[1]]` before `IsLineADoor`; PATCH passed a bogus line to `EV_VerticalDoor`; non-string `state` could be `strcmp`'d. | Bounds-check id against `numlines`; new `DoorSector()` returns NULL for `sidenum[1] < 0`/`>= numsides`; skip non-doors before deref; require `state` to be a string; guard `sectorIds[512]` overflow. | `tests/api_smoke.py`: doors `999999`/`-1` → 404, process alive; `./tests/c/run.sh` bounds helpers. |
| C-04 (PEN-07) | **MEDIUM** | **FIXED** | `src/doom/api_world_controller.c:34,97` | `PATCH /api/world` accepted any integer for `skill`/`map`/`episode`; `doom_skills[gameskill]` then read an out-of-bounds pointer (info leak/crash). | Validate `skill` 0..4 (`sk_baby`..`sk_nightmare`), `map` 1..9, `episode` 1..4; apply all-or-nothing before `G_DeferedInitNew`; bounds-check the description lookup. | `tests/api_smoke.py`: skill 9, map 0, episode 0 → 400. |
| C-05 (PEN-04) | **HIGH** | **FIXED** | `src/doom/api.c:87-115` | Peer connect+close without data: `recv_len == 0` never removed/closed the socket, leaving a dangling `client_sd` in the set; next request used freed memory (SIGTRAP in `SDLNet_TCP_Accept`). Also leaked set slots. | Handle `recv_len <= 0` by DelSocket/Close and `client_sd = NULL`; close any previous client before adopting a new one; handle `AddSocket` failure; reset `client_sd` after every request. | `tests/api_smoke.py`: empty connect+close followed by `GET /api/world` returns 200 and the process stays up. |
| C-06 (NEW-04, PEN-03, SEC-10, gRPC bind) | **MEDIUM** | **MITIGATED** | `src/doom/api.c:48,352`; `doom-grpc-server/src/lib.rs:998`; `mcp/restfuldoom_mcp/server.py:78`; `capsule/rootfs/opt/capsule/start.sh` | REST + gRPC control planes listened on all interfaces with no in-process auth; `Access-Control-Allow-Origin: *` let any visited web page read game state; Docker published all ports on every host interface. | Default REST mode accepts **loopback peers only** (SDL_net 2.x cannot bind a concrete address; the peer is filtered at accept) and gRPC binds `127.0.0.1` unless `DOOM_GRPC_BIND` is set. Mutations require `Authorization: Bearer`/`X-Doom-Token` when a token is configured, and are **refused** on non-loopback binds without one. CORS is opt-in via `DOOM_API_CORS_ORIGIN`. Docker start publishes `127.0.0.1:<port>` and injects a generated `DOOM_API_TOKEN`; capsule start opts into `0.0.0.0` inside the container. | `tests/api_smoke.py` auth block: no token → 401, valid Bearer → 200, wrong token → 401; off-host peer to default server is dropped (reset). `mcp` compile + `bash -n`. |
| C-07 (NEW-03, PEN-06) | **MEDIUM** | **FIXED** | `src/doom/api.c:166`; `src/doom/api_object_controller.c:82-91,181` | `cJSON_Parse` NULL and missing/mistyped keys dereferenced (`position`, `type`), so `{}`/`{` crashed the game; `if (!obj)` guarded the wrong pointer at `:181`. | Reject non-empty unparseable bodies with 400 at the router; validate every `cJSON_GetObjectItem` with `cJSON_IsObject/IsNumber/IsString`; fixed the `obj_to_attack` guard; check `mo != NULL`. | `tests/api_smoke.py`: `{}` and `{` → 400, process alive. |
| C-08 (PEN-09) | **LOW** | **FIXED** | `src/doom/api.c:522` | `strncpy(hud_message, msg, 512)` could leave the 512-byte global unterminated; HUD read past the buffer. | `snprintf(hud_message, sizeof hud_message, "%s", msg)`. | `tests/api_smoke.py`: 600-byte message returns 201 and the process stays alive. |
| C-09 (PEN-10) | **LOW** | **FIXED** | `src/doom/api.c:94-110` | On parse failure the access log read an uninitialised `api_request_t` with `%s` (stack disclosure / over-read). | `memset` the request; only log after a successful parse; peer IP NULL-checked. | Code inspection + `tests/api_smoke.py` malformed requests. |
| C-10 (NEW-05) | **LOW** | **FIXED** | `src/doom/api_world_controller.c:136-144`; `src/v_video.c:814` | Unauthenticated GET unlinked and rewrote predictable `/tmp/DOOM00.pcx` (symlink/race, arbitrary app-writable overwrite). | Screenshots go to a per-process `mkdtemp` directory (0700); `lbmname` widened to hold the path; returned `path` points at the private file. | Live GET smoke (screenshot not asserted byte-wise); code review. |
| C-11 (PEN-10 supporting) | **LOW** | **FIXED** | `src/doom/api.c:215-330,286` | Route matching used `strstr`, so `/anything/api/players/1` matched; `sscanf` in `/api/world/los/` and `movetest` ignored the return value, leaving `id`/`id2` uninitialised. | Prefix matching with `strncmp`; initialise all route locals; require both `los` ids. | Build + smoke routes. |
| P-01 (SEC-01, PEN-11) | **LOW** | **MITIGATED** | `agent/restfuldoom_agent/snapshot_builder.py:342,365-385` | `subprocess.run(..., shell=True)` on an operator template; `{tick}`/`{record_index}` were interpolated **unquoted** from trajectory JSON (command injection with a crafted trajectory). | `{tick}`/`{record_index}` must parse as integers or rendering fails; added `{tick_sh}`/`{record_index_sh}`; simple commands now run as an argv list without a shell, shell syntax (`\|`, `>`, `;`, …) still supported for documented templates. | `agent/tests/test_security_fixes.py`: injection strings raise, `printf` template runs argv-only, quoting/argv tests pass. |
| P-02 (SEC-02, SEC-04) | **LOW** | **FIXED** | `agent/restfuldoom_agent/ppo.py:577` | `torch.load` without an explicit `weights_only`; on the `torch>=2.3` floor this allows arbitrary pickle execution from a checkpoint. | `torch.load(..., weights_only=True)`; verified every committed checkpoint still loads. | `agent/tests/test_security_fixes.py`: monkeypatched load records `weights_only=True`; committed `.pt` files load with it. |
| P-03 (NEW-06, PEN-12) | **LOW** | **FIXED** | `agent/restfuldoom_agent/brain.py:5109-5122,5568` | Archive member names were checked but symlink/hardlink members and `manifest["memory_path"]` could escape `destination`. | Reject link members; use `tarfile.extract(..., filter="data")` where available; resolve `memory_path` and require it under `dest`. | `agent/tests/test_security_fixes.py`: `../escaped.json` and a symlink member both raise. |
| P-04 (SEC-09, PEN-13) | **LOW** | **FIXED** | `mcp/restfuldoom_mcp/server.py:42-43,594` | Predictable world-writable `/tmp` PID/log files: a planted PID file made `viewer_stop` SIGKILL an arbitrary process; the log open followed symlinks. | State moved to a private per-user dir (0700); PID written atomically with 0600; log opened `O_NOFOLLOW`/0600; PID must belong to a `restfuldoom_mcp.viewer` process and be > 1 before signalling. | `python -m py_compile`; code review. |
| P-05 (PEN-14) | **LOW** | **DEFERRED** | `mcp/restfuldoom_mcp/server.py:164-175,269-286,586-591` | MCP tools accept arbitrary `host`/`endpoint` values and make outbound connections (blind SSRF/port scan from a prompt-injected agent). | Allow-listing would break documented remote Hellbox/Shrink endpoints and operator-supplied hosts. Needs a separate policy/allow-list design. Not fixed. | — |
| P-06 | **LOW** | **FIXED** | `scripts/hellbox-agent-demo.sh:275-279` | `shrink token --raw > $TOKEN_JSON` created the bearer token with the default umask (typically 0644). | Write under `umask 077` and `chmod 600`. | `bash -n`; code review. |
| P-07 | **LOW** | **FIXED** | `capsule/rootfs/opt/capsule/video_stream.py:234` | MJPEG server defaulted to `0.0.0.0`. | Default `127.0.0.1`; the capsule passes `--host 0.0.0.0` explicitly (container-only exposure). | `py_compile`; `bash -n` on `start.sh`. |
| D-01 (SEC-05) | LOW | **ACCEPTED RISK** | `agent/uv.lock` (filelock 3.29.4) | Nine filelock advisories; filelock is transitive to torch, never imported by repo code, and not installed in the capsule image. All are same-host/local lock-file issues. | Not bumped (`uv lock --upgrade-package filelock` will follow a torch bump). Compensating control: no repo import, no container install. | `grep` of imports; Devil's Advocate analysis. |
| D-02 (SEC-06) | LOW-MED | **ACCEPTED RISK** | `capsule/Dockerfile:71` | No `USER`: ready-hook, MJPEG server and supervisor run as root (the game itself drops to `app` via `runuser`). | Naive `USER app` breaks the image (`/run/capsule.ready`, `/var/log`, `runuser` need root). Requires reworking the readiness/log contract; tracked as follow-up. | Devil's Advocate fix-risk analysis. |
| D-03 (SEC-08) | LOW | **ACCEPTED RISK** | `agent/restfuldoom_agent/env.py:206` | Fixed `/tmp/restfuldoom-ppo-memory.json` path; loaded as JSON (not executed); a local squatter can cause a parse error or poison a dev run. | Moving it changes documented resume behaviour; single-user dev-host threat only. | Code review. |
| D-04 (SEC-03) | INFO | **FALSE POSITIVE** | `agent/restfuldoom_agent/ppo.py:535` | Why FP: this is `torch.save` (serialisation), not `torch.load`; writing a pickle cannot execute attacker code. | None. | — |
| D-05 (SEC-07) | INFO | **FALSE POSITIVE** | `capsule/Dockerfile:13,23,33,43,58` | Why FP: DL3041 exact dnf pins break rolling repos; DL4006 pipes are `printf … \| sha256sum -c`/`make` and the tarballs are SHA256-verified. | None (cosmetic `SHELL -o pipefail` optional). | — |
| D-06 (SEC-11) | INFO | **FALSE POSITIVE** | `RAML/doom_api.html` | Why FP: generated raml2html documentation (no auth/cookies/user data); SRI and CDN HTTP links are regenerated output. | None; regenerate upstream if desired. | `<meta name="generator">` in file. |
| D-07 (SEC-12) | INFO | **FALSE POSITIVE** | `chocpkg/chocpkg/chocpkg_functions.sh:7,17` | Why FP: scoped `SAVE_IFS=$IFS; IFS=:; …; IFS=$SAVE_IFS` is the correct pattern; vendored build tooling. | None. | Code review. |
| D-08 (SEC-13) | INFO | **FALSE POSITIVE** | `configure.ac:136` | Why FP: `case "$host"` is autoconf's target triplet, not an HTTP Host header. | None. | Code review. |
| D-09 (SEC-14) | INFO | **FALSE POSITIVE** | `env.py:3233`, `mcp/server.py:330`, `rollout_config.py:78` | Why FP: `secret_delta: 0`, `token_present: False`, and a `self.token == ""` validation are data keys, not credentials; gitleaks history + worktree are clean. | None. | `gitleaks detect` → no leaks. |
| D-10 (SEC-15) | INFO | **FALSE POSITIVE** | bandit B101/B311/B404/B603 | Why FP: asserts are dev-tool invariants, `random.Random` is used for sampling seeds, subprocess calls use argv lists with fixed binaries. | None. | Code review. |
| D-11 (SEC-16) | INFO | **FALSE POSITIVE** | `capsule/Dockerfile` | Why FP: readiness is the purpose-built `:9000` hook plus `/health` on the video server; Docker HEALTHCHECK is unused by the MicroVM runtime. | None. | `start.sh` review. |
| D-12 (NEW-07) | LOW | **FALSE POSITIVE** | `agent/requirements.txt` | Why FP: `agent/uv.lock` is committed and pins the full graph; lower bounds alone are not a vulnerability. | Install with `uv sync --frozen` (documented). | Lock review. |
| Q-01 | INFO | **FIXED** | repo root | No C tests, no fuzz target, no CI: every C fix could regress silently. | Added `tests/c/test_api_security.c`, `tests/c/fuzz_parse_request.c`, `tests/c/run.sh` (ASan/UBSan + libFuzzer when present), `tests/api_smoke.py`, and `.github/workflows/ci.yml` (clean build + C tests + pytest + gitleaks, `permissions: contents: read`). | `make clean && make -j8` → 0; `./tests/c/run.sh` → ok; `pytest` → 420 passed; gitleaks → no leaks. |

## Verification

Run from the repository root unless noted.

| Command | Result |
|---|---|
| `make clean && make -j8` | exit 0; build now uses `-fstack-protector-strong -D_FORTIFY_SOURCE=2` (confirmed in generated `Makefile`) |
| `./tests/c/run.sh` | `api security tests: ok` (parser + bounds + 50k deterministic fuzz iterations under ASan/UBSan). libFuzzer target built/run where clang ships the runtime; this Mac's Xcode clang lacks `libclang_rt.fuzzer_osx.a`, so it reports `skipped` locally (Ubuntu CI has it). |
| `cd agent && .venv/bin/python -m pytest -q` | `420 passed` (411 baseline + 9 new) |
| `cd doom-grpc-server && cargo check` | exit 0 |
| `gitleaks detect --source . --redact --no-banner` | `242 commits scanned`, `no leaks found` |
| `DOOM_WAD=… DOOM_TEST_LAN_IP=… python3 tests/api_smoke.py` | `ALL PASS` (23 checks: parsing bounds, socket lifecycle, player/ammo/door/skill/map bounds, JSON handling, token auth, off-host peer rejection) |
| `bash -n capsule/rootfs/opt/capsule/start.sh scripts/hellbox-agent-demo.sh` | exit 0 |
| `python3 -m py_compile mcp/…/server.py capsule/…/video_stream.py agent/…/{snapshot_builder,brain,ppo}.py` | exit 0 |

## Residual risk

1. **The REST listener still binds `INADDR_ANY`.** SDL_net 2.x's `SDLNet_TCP_Open` treats a concrete host
   as a *client* connect, so a true `bind(127.0.0.1)` requires replacing the SDL_net accept path with raw
   sockets. Instead, non-loopback peers are dropped at accept (`API_PeerAllowed`). The port remains
   visible to a network port scan, but no request bytes are read or served. `-apibind`/`DOOM_API_BIND`
   with a non-loopback value disables the peer filter (opt-in).
2. **Loopback clients are unauthenticated by default.** Any process running as the same user can drive a
   locally launched game. Set `DOOM_API_TOKEN` to require a bearer token for mutations even locally.
3. **CAPTURE templates still use a shell when they contain shell syntax.** Substituted `{tick}` /
   `{record_index}` are restricted to integers and the `_sh` variants are quoted, but an
   operator-supplied template (e.g. with pipes/redirection) is trusted.
4. **The capsule image still runs its readiness hook, supervisor and MJPEG server as root.** The game
   process drops to `app`. A correct `USER app` change needs to move `/run/capsule.ready`, logs and the
   hook out of root-only paths first.
5. **gRPC has no in-protocol authentication.** It is loopback-bound by default and Docker publishes only
   `127.0.0.1`, but an operator who sets `DOOM_GRPC_BIND=0.0.0.0` or publishes the port exposes it.
6. **filelock CVEs (9) remain in `agent/uv.lock`.** Not reachable from repo code, absent from the
   container; no compatible security-urgent bump was made.
7. **MCP SSRF (P-05) is deferred.** The tools intentionally accept operator-supplied endpoints; an
   allow-list would break documented remote workflows.
8. **`agent/restfuldoom_agent/env.py` still uses a fixed `/tmp` memory path (D-03).** Low, local-only.
9. **Behaviour changes to note:** `GET /api/world/screenshot` now returns a path inside a per-process
   private temp directory (not `/tmp/DOOM00.pcx`); `Access-Control-Allow-Origin` is no longer sent by
   default (set `DOOM_API_CORS_ORIGIN` to restore); Docker REST mutations need `DOOM_API_TOKEN`.
10. **No secret was found** in the worktree or history, so no credential rotation is required.
