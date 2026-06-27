"""Save or load native Doom agent snapshot slots over gRPC."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .client import DEFAULT_AGENT_PORT, DoomAgentClient


async def run_snapshot_slot_command(args: argparse.Namespace) -> dict[str, Any]:
    """Runs one native snapshot-slot command and returns JSON-safe metadata."""
    endpoint = args.endpoint
    token = args.token
    if args.token_json is not None:
        with args.token_json.open("r", encoding="utf-8") as handle:
            token_data = json.load(handle)
        endpoint = endpoint or token_data.get("endpoint")
        token = token or token_data.get("token")
    if not endpoint:
        raise ValueError("--endpoint or --token-json with endpoint is required")

    client = DoomAgentClient(
        endpoint,
        token=token,
        agent_port=args.agent_port,
        tls=args.tls,
        authority=args.authority,
    )
    try:
        if args.command == "save":
            response = await client.save_snapshot(
                slot=args.slot,
                description=args.description,
                run_id=args.run_id,
            )
            action = "save"
        elif args.command == "load":
            response = await client.load_snapshot(
                slot=args.slot,
                run_id=args.run_id,
            )
            action = "load"
        else:
            raise ValueError(f"unsupported command {args.command!r}")
    finally:
        await client.close()

    return {
        "schema": "restfuldoom.snapshot_slot_command.v1",
        "action": action,
        "accepted": response.accepted,
        "message": response.message,
        "slot": response.slot,
        "save_queued": response.save_queued,
        "load_queued": response.load_queued,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["save", "load"])
    parser.add_argument("--endpoint")
    parser.add_argument("--token")
    parser.add_argument("--token-json", type=Path)
    parser.add_argument("--agent-port", type=int, default=DEFAULT_AGENT_PORT)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--authority")
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--description", default="agent snapshot")
    parser.add_argument("--run-id", default="snapshot-slot")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs the snapshot-slot CLI."""
    args = _parser().parse_args(argv)
    result = asyncio.run(run_snapshot_slot_command(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
