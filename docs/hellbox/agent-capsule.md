# Hellbox Agent Capsule

This repo includes a Hellbox-style capsule surface for gRPC agent mode:

- `capsule/Dockerfile` builds SDL2, SDL2_mixer, SDL2_net, this RESTful Doom tree, and the Rust gRPC static library.
- `capsule/rootfs/opt/capsule/start.sh` starts `restful-doom` with `-agentport 50051`.
- `:9000` is the internal MicroVM ready hook.
- `:50051` is the external gRPC agent port to include in the MicroVM auth token.

The capsule is intentionally headless. It sets `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`
because the agent interface uses structured protobuf state rather than video frames.

Typical launch command inside the capsule:

```bash
restful-doom -iwad /home/app/app/DOOM1.WAD -warp 1 1 -skill 3 -nosound -nomusic -agentport 50051 -apiport 6666
```

Hellbox should forward or authorize port `50051` for the Python agent. The legacy REST API can
still be exposed on `6666` for debugging.
