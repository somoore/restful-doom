#!/usr/bin/env bash
# Install the RESTful Doom MCP server into this Codex profile.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${RESTFULDOOM_MCP_VENV:-$ROOT/.venv-mcp}"
CONFIG="${CODEX_CONFIG:-$HOME/.codex/config.toml}"

"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$ROOT/agent/requirements.txt" "mcp>=1.16,<2"
PYTHONPATH="$ROOT/agent" "$VENV/bin/python" -m restfuldoom_agent.generate_stubs >/dev/null

"$VENV/bin/python" "$ROOT/scripts/install_codex_mcp_config.py" "$CONFIG" "$ROOT" "$VENV/bin/python"

echo "installed restful_doom MCP server in $CONFIG"
echo "restart Codex or start a new thread for the new MCP tools to be discovered"

