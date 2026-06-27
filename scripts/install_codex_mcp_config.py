"""Install the RESTful Doom MCP server block into Codex config."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: install_codex_mcp_config.py <config.toml> <repo-root> <python>"
        )

    config_path = Path(sys.argv[1]).expanduser()
    repo_root = Path(sys.argv[2]).resolve()
    python = Path(sys.argv[3]).resolve()
    block = _server_block(repo_root, python)

    current = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    pattern = re.compile(
        r"\n?\[mcp_servers\.restful_doom\]\n.*?"
        r"(?=\n\[mcp_servers\.|\n\[plugins\.|\n\[projects\.|\Z)",
        re.DOTALL,
    )
    if pattern.search(current):
        updated = pattern.sub("\n" + block.rstrip() + "\n", current)
    else:
        separator = "" if current.endswith("\n\n") or not current else "\n"
        updated = current + separator + block

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(updated, encoding="utf-8")


def _server_block(repo_root: Path, python: Path) -> str:
    return f"""[mcp_servers.restful_doom]
command = "{python}"
args = ["-m", "restfuldoom_mcp.server"]
startup_timeout_sec = 60

[mcp_servers.restful_doom.env]
RESTFUL_DOOM_REPO = "{repo_root}"
PYTHONPATH = "{repo_root / 'mcp'}:{repo_root / 'agent'}"
"""


if __name__ == "__main__":
    main()

