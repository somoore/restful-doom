# RESTful-DOOM

This fork adds a protobuf + gRPC Agent Mode and Hellbox capsule surface for
running DOOM as a headless, structured agent environment inside a cloud
MicroVM. The original REST API is still available, but the main new path is a
bidirectional 35 Hz observe-act stream for AI policies.

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

## Fork Delta

This fork keeps the original HTTP + JSON API and adds:

- protobuf schema for agent-focused game state and actions
- Rust `tonic` gRPC server linked into the Doom binary as a static library
- C/Rust FFI bridge that publishes state after each `G_Ticker()` and overlays queued actions into `ticcmd`
- Python async client, deterministic smoke agent, reward scoring, Bedrock policy hook, and JSONL trajectory logging
- Hellbox-style headless capsule packaging for a MicroVM-ready agent environment

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

To run restful-doom on port 6666:
```
src/restful-doom -iwad <path/to/doom1.wad> -apiport 6666 ...
```

## gRPC Agent Mode

RESTful-DOOM also includes a high-performance protobuf + gRPC bridge for AI agents. The
bridge runs inside the Doom process, publishes structured state after every simulated tic, and
accepts player actions over a bidirectional stream.

```mermaid
flowchart LR
    Agent["Python Agent"] <-->|"gRPC :50051\nGameState / PlayerAction"| Doom["restful-doom process\nRust gRPC + C FFI"]
    Doom <--> Sim["DOOM simulation\n35 Hz ticks"]
    Doom -.-> Trace["JSONL trajectories\nstate / action / reward"]
    subgraph Box["Hellbox MicroVM"]
        Doom
        Sim
    end
    Box -.->|"freeze / thaw"| Snapshot["resumable VM state"]
```

The shared schema lives at:

```
proto/restfuldoom/v1/agent.proto
```

Run Doom with the agent bridge on the default gRPC port:

```
src/restful-doom -iwad <path/to/doom1.wad> -warp 1 1 -skill 3 -agent
```

Or choose a port explicitly:

```
src/restful-doom -iwad <path/to/doom1.wad> -warp 1 1 -skill 3 -agentport 50051
```

The stream service is:

```
restfuldoom.v1.DoomAgent/GameSession
```

It returns `GameState` messages containing player state, enemy state, useful map objects,
level progress, and optional per-client `StateDelta` payloads. Actions can be high-level
semantic commands such as `ACTION_FORWARD` and `ACTION_SHOOT`, or exact raw `ticcmd` overlays
for low-level policies.

### Python Agent

Generate Python protobuf stubs and run the deterministic smoke agent:

```
cd agent
python -m pip install -r requirements.txt
PYTHONPATH=. python -m restfuldoom_agent.generate_stubs
PYTHONPATH=. python -m restfuldoom_agent.smoke_agent --endpoint 127.0.0.1:50051
```

Add `--trajectory-jsonl trajectories/run.jsonl` to the smoke agent to persist
state/action/reward records for replay or training analysis.
Use `--goal-preset survival|navigation|combat|item_collection|exit_seeking` to
switch reward shaping, and leave reconnect enabled for Hellbox demos so stderr
reports the gRPC status, backoff delay, and last observed Doom tick.

The Python package includes:

- async gRPC client helpers
- streaming rollout API with reconnect/backoff and last-seen-tick reporting
- a deterministic smoke policy
- named goal/reward presets
- optional nonblocking AWS Bedrock text-reasoning policy
- JSONL trajectory logging

### Resilient Rollout Demo

Run a long-lived rollout with a narratable goal and incremental trajectory log:

```
PYTHONPATH=agent python -m restfuldoom_agent.smoke_agent \
    --endpoint 127.0.0.1:50051 \
    --goal-preset combat \
    --trajectory-jsonl trajectories/combat.jsonl
```

If the gRPC stream drops during a Hellbox freeze/thaw or process restart, the
client prints reconnect notices to stderr:

```
{"attempt":1,"code":"UNAVAILABLE","delay_seconds":0.25,"details":"Stream removed (Socket closed)","event":"reconnect","last_seen_tick":1234}
{"attempt":2,"code":"UNAVAILABLE","delay_seconds":0.5,"details":"failed to connect to all addresses","event":"reconnect","last_seen_tick":1234}
```

Trajectory records are written one line at a time so rollouts do not retain the
full state history in memory:

```
{"index":0,"last_seen_tick":1234,"next_action":{"action":7,"amount":10,"duration_tics":1},"reconnect_attempts":0,"reward":{"done":false,"health_delta":0,"item_delta":0,"kill_delta":0,"progress_delta":0.0,"reward":0.0,"secret_delta":0},"state":{"enemy_count":6,"health":100,"tick":1234}}
```

`last_seen_tick` makes reconnect behavior debuggable. For training/eval
lineage, a future server-side run id should distinguish a thawed VM from a new
Doom process that happens to listen on the same endpoint.

### Hellbox Capsule

The `capsule/` directory contains a Hellbox-style headless capsule that builds this repo,
starts Doom in agent mode, exposes gRPC on `50051`, and uses the MicroVM ready hook on `9000`.
See `docs/hellbox/agent-capsule.md`.

The strongest demo loop is:

1. Launch the headless RESTful-DOOM agent capsule in a Hellbox MicroVM.
2. Mint short-lived access for gRPC port `50051`.
3. Run the Python smoke agent and persist JSONL trajectories.
4. Freeze the MicroVM mid-rollout.
5. Thaw it and continue from the same simulation state.

## Thanks!
[chocolate-doom](https://github.com/chocolate-doom/chocolate-doom) team  
[cJSON](https://github.com/DaveGamble/cJSON) - JSON parsing / generation  
[yuarel](https://github.com/jacketizer/libyuarel/) - URL parsing  
