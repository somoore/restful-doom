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
SNAPSHOT_CURRICULUM="${SNAPSHOT_CURRICULUM:-$ROOT/trajectories/snapshot-curriculum.json}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$ROOT/snapshots}"
SNAPSHOT_INDEXES="${SNAPSHOT_INDEXES:-}"
SNAPSHOT_AUTO="${SNAPSHOT_AUTO:-first-visible first-shootable first-damage}"
SNAPSHOT_SAVE_SLOT_BASE="${SNAPSHOT_SAVE_SLOT_BASE:-}"
SNAPSHOT_SLOT="${SNAPSHOT_SLOT:-0}"
SNAPSHOT_CAPTURE_COMMAND="${SNAPSHOT_CAPTURE_COMMAND:-}"
SNAPSHOT_CAPTURE_MAX_STATES="${SNAPSHOT_CAPTURE_MAX_STATES:-12000}"
SNAPSHOT_VERIFY_LOADS="${SNAPSHOT_VERIFY_LOADS:-0}"
SNAPSHOT_REQUIRE_ARTIFACTS="${SNAPSHOT_REQUIRE_ARTIFACTS:-0}"
BRAIN_MEMORY="${BRAIN_MEMORY:-$ROOT/agent_memory/e1m1.json}"
FREEZE_AFTER_SECONDS="${FREEZE_AFTER_SECONDS:-8}"
THAW_AFTER_SECONDS="${THAW_AFTER_SECONDS:-3}"

usage() {
    cat >&2 <<EOF
usage: $(basename "$0") build|up|token|run|suspend|resume|validate|snapshot-plan|snapshot-capture|snapshot-validate|snapshot-save|snapshot-load|production-demo|loop

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
  SNAPSHOT_CURRICULUM     Snapshot curriculum manifest path.
  SNAPSHOT_DIR            Snapshot artifact directory.
  SNAPSHOT_INDEXES        Comma-separated zero-based trajectory rows to snapshot.
  SNAPSHOT_AUTO           Space-separated auto selectors. Default: first-visible first-shootable first-damage
  SNAPSHOT_SAVE_SLOT_BASE Optional first Doom agent save slot assigned to generated stages.
  SNAPSHOT_SLOT           Native Doom agent save slot for snapshot-save/load. Default: 0
  SNAPSHOT_CAPTURE_COMMAND Optional command template that writes {snapshot_path_sh}.
  SNAPSHOT_CAPTURE_MAX_STATES Max brain rollout states for snapshot-capture. Default: 12000
  SNAPSHOT_VERIFY_LOADS    Set to 1 to load each captured native slot after capture.
  SNAPSHOT_REQUIRE_ARTIFACTS Set to 1 to require local snapshot files during validation.
  BRAIN_MEMORY             Structured brain memory path. Default: agent_memory/e1m1.json

The run command uses raw token JSON from '$HELLBOX_CLI token --raw' unless
endpoint/token overrides are present in the environment. The token command writes
raw JSON to HELLBOX_TOKEN_JSON and prints a redacted copy to stdout.
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
value = data
for part in key.split("."):
    if not isinstance(value, dict):
        sys.exit(1)
    value = value.get(part)
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

print_redacted_token_json() {
    "$PYTHON_BIN" - "$TOKEN_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
if "token" in data:
    data["token"] = "<redacted>"
headers = data.get("headers")
if isinstance(headers, dict) and "x-aws-proxy-auth" in headers:
    headers["x-aws-proxy-auth"] = "<redacted>"
print(json.dumps(data, indent=2, sort_keys=True))
PY
}

ensure_token_json() {
    if [ ! -s "$TOKEN_JSON" ]; then
        "$0" token >/dev/null
    fi
    local schema
    schema="$(json_value "$TOKEN_JSON" schema || true)"
    if [ "$schema" != "shrink.auth.v1" ]; then
        echo "Expected token JSON schema shrink.auth.v1, got '${schema:-missing}'." >&2
        echo "You may be using an old token file at $TOKEN_JSON." >&2
        echo "Regenerate it with: $HELLBOX_CLI token $NAME --port $AGENT_PORT --minutes $TOKEN_MINUTES --raw" >&2
        exit 2
    fi
}

run_agent() {
    ensure_token_json
    local endpoint="${HELLBOX_GRPC_ENDPOINT:-$(json_value "$TOKEN_JSON" endpoint)}"
    local token="${HELLBOX_GRPC_TOKEN:-$(json_value "$TOKEN_JSON" token)}"
    local capsule_id
    local auth_lease_id
    local capsule_name
    capsule_id="$(json_value "$TOKEN_JSON" microvm_id || true)"
    auth_lease_id="$(json_value "$TOKEN_JSON" auth_lease_id || true)"
    capsule_name="$(json_value "$TOKEN_JSON" capsule || echo "$NAME")"

    mkdir -p "$(dirname "$TRAJECTORY_JSONL")"
    PYTHONPATH="$ROOT/agent" "$PYTHON_BIN" -m restfuldoom_agent.smoke_agent \
        --config "$CONFIG" \
        --endpoint "$endpoint" \
        --token "$token" \
        --agent-port "$AGENT_PORT" \
        --tls \
        --trajectory-jsonl "$TRAJECTORY_JSONL" \
        --capsule-name "$capsule_name" \
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

snapshot_plan() {
    local args=(
        --trajectory "$TRAJECTORY_JSONL"
        --output "$SNAPSHOT_CURRICULUM"
        --name "$NAME-progressed-bottlenecks"
        --snapshot-dir "$SNAPSHOT_DIR"
        --capsule "$NAME"
    )
    if [ -n "$SNAPSHOT_INDEXES" ]; then
        args+=(--indexes "$SNAPSHOT_INDEXES")
    fi
    if [ -n "$SNAPSHOT_AUTO" ]; then
        local selector
        for selector in $SNAPSHOT_AUTO; do
            args+=(--auto "$selector")
        done
    fi
    if [ -n "$SNAPSHOT_SAVE_SLOT_BASE" ]; then
        args+=(--save-slot-base "$SNAPSHOT_SAVE_SLOT_BASE")
    fi
    if [ -n "$SNAPSHOT_CAPTURE_COMMAND" ]; then
        args+=(--capture-command "$SNAPSHOT_CAPTURE_COMMAND")
    fi
    if [ "$SNAPSHOT_REQUIRE_ARTIFACTS" = "1" ]; then
        args+=(--require-capture-artifacts)
    fi
    PYTHONPATH="$ROOT/agent" "$PYTHON_BIN" -m restfuldoom_agent.snapshot_builder "${args[@]}"
}

snapshot_validate() {
    local args=("$SNAPSHOT_CURRICULUM" --validate)
    if [ "$SNAPSHOT_REQUIRE_ARTIFACTS" = "1" ]; then
        args+=(--require-artifacts)
    fi
    PYTHONPATH="$ROOT/agent" "$PYTHON_BIN" -m restfuldoom_agent.snapshot_curriculum "${args[@]}"
}

snapshot_capture() {
    ensure_token_json
    local args=(
        --token-json "$TOKEN_JSON"
        --agent-port "$AGENT_PORT"
        --tls
        --memory-path "$BRAIN_MEMORY"
        --trajectory-jsonl "$TRAJECTORY_JSONL"
        --output "$SNAPSHOT_CURRICULUM"
        --name "$NAME-progressed-bottlenecks"
        --snapshot-dir "$SNAPSHOT_DIR"
        --capsule "$NAME"
        --max-states "$SNAPSHOT_CAPTURE_MAX_STATES"
    )
    if [ -n "$SNAPSHOT_SAVE_SLOT_BASE" ]; then
        args+=(--save-slot-base "$SNAPSHOT_SAVE_SLOT_BASE")
    fi
    if [ -n "$SNAPSHOT_AUTO" ]; then
        local selector
        for selector in $SNAPSHOT_AUTO; do
            args+=(--auto "$selector")
        done
    fi
    if [ "$SNAPSHOT_VERIFY_LOADS" = "1" ]; then
        args+=(--verify-loads)
    fi
    PYTHONPATH="$ROOT/agent" "$PYTHON_BIN" -m restfuldoom_agent.snapshot_capture "${args[@]}"
}

snapshot_slot_command() {
    ensure_token_json
    PYTHONPATH="$ROOT/agent" "$PYTHON_BIN" -m restfuldoom_agent.snapshot_slots "$1" \
        --token-json "$TOKEN_JSON" \
        --agent-port "$AGENT_PORT" \
        --tls \
        --slot "$SNAPSHOT_SLOT" \
        --run-id "$NAME-snapshot-slot"
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
        "$HELLBOX_CLI" token "$NAME" --port "$AGENT_PORT" --minutes "$TOKEN_MINUTES" --raw > "$TOKEN_JSON"
        note "token"
        print_redacted_token_json
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
    snapshot-plan)
        snapshot_plan
        note "snapshot-plan"
        ;;
    snapshot-capture)
        snapshot_capture
        note "snapshot-capture"
        ;;
    snapshot-validate)
        snapshot_validate
        note "snapshot-validate"
        ;;
    snapshot-save)
        snapshot_slot_command save
        note "snapshot-save"
        ;;
    snapshot-load)
        snapshot_slot_command load
        note "snapshot-load"
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
