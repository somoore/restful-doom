# Hellbox Agent Capsule

This repo includes a Hellbox-style capsule surface for gRPC agent mode. Hellbox
is the demo and brand; `shrink` is the MicroVM capsule runtime CLI;
`agent-doom` is the headless Doom capsule; `restful-doom` is the Doom agent
environment implementation. This capsule is the machine-readable companion to a
browser-streamed playable DOOM capsule:

| Demo | Interface | Audience | Hook |
| --- | --- | --- | --- |
| Playable DOOM | Browser video/audio/input | Humans | Freeze a live game mid-fight |
| Agent DOOM | gRPC structured state/actions | AI agents and evals | Freeze a policy rollout and resume the same simulation |

The agent capsule contract is:

- `capsule/Dockerfile` builds SDL2, SDL2_mixer, SDL2_net, this RESTful Doom tree, and the Rust gRPC static library.
- `capsule/rootfs/opt/capsule/start.sh` starts `restful-doom` with `-agentport 50051`.
- `capsule/agent-doom.hellbox.json` names the capsule `agent-doom` and marks gRPC port `50051` as discoverable.
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

## External Capsule Install Path

From this repo, build and launch the capsule as an external Hellbox/Shrink
capsule:

```bash
RESTFUL_DOOM_CAPSULE_DIR="$PWD/capsule" shrink build --name agent-doom
shrink up agent-doom
shrink token agent-doom --port 50051 --minutes 60 --raw > trajectories/agent-doom-token.json
```

The token JSON uses the stable `shrink.auth.v1` schema and includes the external
TLS endpoint, auth lease id, MicroVM id, port, expiry, and headers. It is
redacted by default; `--raw` is only for the demo helper or other scripts that
need the bearer credential. The Python agent sends `x-aws-proxy-auth` plus
`x-aws-proxy-port: 50051` as gRPC metadata.

Redacted docs shape:

```json
{
  "schema": "shrink.auth.v1",
  "capsule": "agent-doom",
  "endpoint": "abc.lambda-microvm.us-east-2.on.aws:443",
  "port": 50051,
  "tls": true,
  "headers": {
    "x-aws-proxy-auth": "<redacted>",
    "x-aws-proxy-port": "50051"
  }
}
```

```bash
scripts/hellbox-agent-demo.sh run
```

`scripts/hellbox-agent-demo.sh` also exposes the lifecycle commands:

```bash
scripts/hellbox-agent-demo.sh build
scripts/hellbox-agent-demo.sh up
scripts/hellbox-agent-demo.sh token
scripts/hellbox-agent-demo.sh run
scripts/hellbox-agent-demo.sh suspend   # freeze
scripts/hellbox-agent-demo.sh resume    # thaw
scripts/hellbox-agent-demo.sh production-demo
```

The rollout JSON used by the script lives at
`agent/examples/hellbox-rollout.json`. CLI flags still override that JSON, so a
fresh demo can point at a newly minted endpoint without editing the file.

## Polished Demo Loop

1. Launch the headless RESTful-DOOM capsule in a Hellbox MicroVM.
2. Mint short-lived `shrink.auth.v1` access for gRPC port `50051`.
3. Run `python -m restfuldoom_agent.smoke_agent --config agent/examples/hellbox-rollout.json --endpoint <host>:443 --token <token> --agent-port 50051 --tls`.
4. Confirm the trajectory log is receiving state/action/reward records.
5. Freeze the MicroVM mid-run.
6. Thaw the MicroVM.
7. Continue the same agent/game session from the resumed simulation state.

The punchline is that Hellbox can pause and resume an isolated agent environment,
not only a human-playable game stream.

## PPO Training Loop

For independent learning, use the PPO skill learner rather than the smoke agent:

```bash
python -m restfuldoom_agent.ppo_agent \
    --endpoint <host>:443 \
    --token <token> \
    --agent-port 50051 \
    --tls \
    --goal-preset combat \
    --updates 1 \
    --rollout-steps 512 \
    --checkpoint-dir agent_models/ppo \
    --buffer-dir trajectories/ppo \
    --memory-path agent_memory/e1m1.json
```

The PPO learner uses the same structured protobuf state stream and calls
`ResetEpisode` between episodes. It chooses one of eight high-level skills
(`engage`, `fire`, `seek_enemy`, `open_use_line`, `route_progression`,
`retreat`, `recover_stuck`, `press_exit`) while the deterministic controller
executes the low-level Doom commands. Each batch writes rollout buffers and a
`restfuldoom.ppo_checkpoint.v1` checkpoint with model weights, optimizer state,
schemas, reward config, and eval metadata.

Before promoting PPO over the deterministic brain, evaluate it on fixed run
labels and require it to beat the baseline on level completion, kills, survival,
time/states to exit, and stuck events. Current reset responses include the
requested seed but report `seed_applied=false`; do not claim deterministic
seeded replay until that field is true in a verified build.

## Production Demo Checklist

- Build the capsule from a clean checkout and confirm `/usr/local/bin/restful-doom` links the Rust gRPC static library.
- Start the MicroVM with only the ready hook (`9000`) and gRPC (`50051`) exposed.
- Mint short-lived `shrink.auth.v1` auth for `50051`; do not expose the legacy REST port unless debugging requires it.
- Wait for the capsule ready hook, then verify the gRPC port accepts connections before launching the agent.
- Run the smoke agent with reconnect enabled and JSONL logging:

  ```bash
  python -m restfuldoom_agent.smoke_agent \
      --config agent/examples/hellbox-rollout.json \
      --endpoint <host>:443 \
      --token <token> \
      --agent-port 50051 \
      --tls \
      --trajectory-jsonl trajectories/hellbox-run.jsonl
  ```

- Watch stderr for reconnect notices. Each notice includes the gRPC status, delay, and last observed Doom tick.
- Confirm the trajectory file grows with `state`, `reward`, `next_action`, `last_seen_tick`, `reconnect_attempts`, and `metadata` fields.
- For PPO demos, confirm `trajectories/ppo/*.jsonl` contains rollout records with `obs`, `action`, `reward`, `done`, `value`, and `logprob`, and confirm `agent_models/ppo/*.pt` checkpoints are written.
- For mixed-stage PPO demos, confirm rollout summaries include `curriculum_stage_counts` with more than one stage and JSONL rows preserve per-record `info.curriculum_stage.name`.
- For snapshot-backed PPO demos, build a capture plan with `scripts/hellbox-agent-demo.sh snapshot-plan`, validate the `restfuldoom.snapshot_curriculum.v1` manifest with `snapshot-validate`, run with `--snapshot-curriculum` plus a local restore command, and confirm no snapshot row calls `ResetEpisode` after restore.
- Treat a snapshot manifest with missing artifacts as planning-only. Set `SNAPSHOT_REQUIRE_ARTIFACTS=1` for promotion-quality validation so local snapshot files and `sha256:` digests are checked before training/export.
- Confirm snapshot rollout rows include `info.reset_context.source == "snapshot_restore"` and summaries include `snapshot_restore_count`, `snapshot_stage_counts`, and restore timing fields.
- Export a resume bundle after PPO training and confirm the manifest includes `ppo_checkpoints`, `observation_schema`, `action_schema`, `reward_config`, and `eval_history`.
- For snapshot-backed training, also confirm the export manifest includes `snapshot_curriculum`, `snapshot_restore_context`, and any local `snapshot_artifacts`.
- Check `metadata.reconnect_count`, `metadata.policy_errors`, `metadata.bedrock_fallback_count`, and `metadata.llm_latency_ms` when debugging a run.
- Confirm `metadata.rollout.token_present` is true and that no raw token appears in the JSONL.
- Confirm docs and screenshots use redacted token JSON; reserve `--raw` for local scripts.
- Freeze the MicroVM while the trajectory file is still receiving ticks.
- Thaw the MicroVM and confirm the next reconnect continues from a later tick instead of starting a fresh process.
- Save the trajectory, Hellbox VM id, auth lease id, freeze timestamp, and thaw timestamp with the demo notes.

`last_seen_tick` proves what the client last observed before reconnecting. For
formal eval lineage, add a server-side run id later so the client can distinguish
same MicroVM resumed, same process new game, and fresh process restart.
