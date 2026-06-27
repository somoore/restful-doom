#!/usr/bin/env bash
# Build, launch, and drive the RESTful Doom Hellbox/Shrink agent capsule.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELLBOX_CLI="${HELLBOX_CLI:-shrink}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NAME="${HELLBOX_NAME:-agent-doom}"
AGENT_PORT="${HELLBOX_AGENT_PORT:-50051}"
TOKEN_MINUTES="${HELLBOX_TOKEN_MINUTES:-60}"
CONFIG="${ROLLOUT_CONFIG:-$ROOT/agent/examples/hellbox-rollout.json}"
TRAJECTORY_JSONL="${TRAJECTORY_JSONL:-$ROOT/trajectories/hellbox-run.jsonl}"
TOKEN_JSON="${HELLBOX_TOKEN_JSON:-$ROOT/trajectories/agent-doom-token.json}"
DEMO_NOTES="${HELLBOX_DEMO_NOTES:-$ROOT/trajectories/hellbox-demo-notes.jsonl}"
FREEZE_AFTER_SECONDS="${FREEZE_AFTER_SECONDS:-8}"
THAW_AFTER_SECONDS="${THAW_AFTER_SECONDS:-3}"

usage() {
    cat >&2 <<EOF
usage: $(basename "$0") build|up|token|run|suspend|resume|validate|production-demo|loop

Environment:
  HELLBOX_CLI             CLI binary. Default: shrink
  PYTHON_BIN              Python interpreter. Default: python3
  HELLBOX_NAME            Capsule name. Default: agent-doom
  HELLBOX_AGENT_PORT      Internal gRPC port. Default: 50051
  HELLBOX_TOKEN_MINUTES   Auth token TTL. Default: 60
  HELLBOX_GRPC_ENDPOINT   Optional override endpoint, e.g. host:443.
  HELLBOX_GRPC_TOKEN      Optional override X-aws-proxy-auth token.
  HELLBOX_TOKEN_JSON      Token JSON path. Default: trajectories/agent-doom-token.json
  ROLLOUT_CONFIG          JSON rollout config. Default: agent/examples/hellbox-rollout.json
  TRAJECTORY_JSONL        Output trajectory path. Default: trajectories/hellbox-run.jsonl

The run command uses token JSON from '$HELLBOX_CLI token' unless endpoint/token
overrides are present in the environment.
EOF
}

json_value() {
    local file="$1"
    local key="$2"
    "$PYTHON_BIN" - "$file" "$key" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
value = data.get(key)
if value is None:
    sys.exit(1)
print(value)
PY
}

note() {
    mkdir -p "$(dirname "$DEMO_NOTES")"
    "$PYTHON_BIN" - "$DEMO_NOTES" "$1" <<'PY'
import json
import sys
import time

path, event = sys.argv[1], sys.argv[2]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"event": event, "timestamp": time.time()}, sort_keys=True) + "\n")
PY
}

ensure_token_json() {
    if [ ! -s "$TOKEN_JSON" ]; then
        "$0" token >/dev/null
    fi
}

run_agent() {
    ensure_token_json
    local endpoint="${HELLBOX_GRPC_ENDPOINT:-$(json_value "$TOKEN_JSON" endpoint)}"
    local token="${HELLBOX_GRPC_TOKEN:-$(json_value "$TOKEN_JSON" token)}"
    local capsule_id
    local auth_lease_id
    capsule_id="$(json_value "$TOKEN_JSON" microvm_id || true)"
    auth_lease_id="$(json_value "$TOKEN_JSON" auth_lease_id || true)"

    mkdir -p "$(dirname "$TRAJECTORY_JSONL")"
    PYTHONPATH="$ROOT/agent" "$PYTHON_BIN" -m restfuldoom_agent.smoke_agent \
        --config "$CONFIG" \
        --endpoint "$endpoint" \
        --token "$token" \
        --agent-port "$AGENT_PORT" \
        --tls \
        --trajectory-jsonl "$TRAJECTORY_JSONL" \
        --capsule-name "$NAME" \
        --capsule-id "$capsule_id" \
        --auth-lease-id "$auth_lease_id"
}

validate_trajectory() {
    "$PYTHON_BIN" - "$TRAJECTORY_JSONL" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit(f"trajectory is missing or empty: {path}")
records = []
with path.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            records.append(json.loads(line))
if not records:
    raise SystemExit(f"trajectory has no JSON records: {path}")
last = records[-1]
metadata = last.get("metadata", {})
required = {"state", "reward", "next_action", "last_seen_tick", "reconnect_attempts", "metadata"}
missing = sorted(required - set(last))
if missing:
    raise SystemExit(f"last trajectory record missing fields: {', '.join(missing)}")
if "rollout" not in metadata:
    raise SystemExit("last trajectory record missing metadata.rollout")
print(
    json.dumps(
        {
            "records": len(records),
            "last_seen_tick": last.get("last_seen_tick"),
            "reconnect_count": metadata.get("reconnect_count", 0),
            "policy_errors": metadata.get("policy_errors", 0),
            "bedrock_fallback_count": metadata.get("bedrock_fallback_count", 0),
        },
        sort_keys=True,
    )
)
PY
}

case "${1:-}" in
    build)
        RESTFUL_DOOM_CAPSULE_DIR="$ROOT/capsule" "$HELLBOX_CLI" build --name "$NAME"
        note "build"
        ;;
    up)
        "$HELLBOX_CLI" up "$NAME"
        note "up"
        ;;
    token)
        mkdir -p "$(dirname "$TOKEN_JSON")"
        "$HELLBOX_CLI" token "$NAME" --port "$AGENT_PORT" --minutes "$TOKEN_MINUTES" > "$TOKEN_JSON"
        note "token"
        cat "$TOKEN_JSON"
        ;;
    run)
        run_agent
        note "run"
        ;;
    suspend|freeze)
        "$HELLBOX_CLI" suspend --name "$NAME"
        note "freeze"
        ;;
    resume|thaw)
        "$HELLBOX_CLI" resume --name "$NAME"
        "$0" token >/dev/null
        note "thaw"
        ;;
    validate)
        validate_trajectory
        note "validate"
        ;;
    production-demo)
        "$0" build
        "$0" up
        "$0" token >/dev/null
        run_agent &
        agent_pid=$!
        sleep "$FREEZE_AFTER_SECONDS"
        "$0" suspend
        sleep "$THAW_AFTER_SECONDS"
        "$0" resume
        wait "$agent_pid"
        "$0" validate
        ;;
    loop)
        "$0" build
        "$0" up
        "$0" token >/dev/null
        "$0" run
        "$0" validate
        ;;
    *)
        usage
        exit 2
        ;;
esac
