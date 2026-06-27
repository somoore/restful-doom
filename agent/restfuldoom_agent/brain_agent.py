"""CLI for the structured local Doom agent brain."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .brain import (
    AgentMemory,
    BrainConfig,
    export_training_job,
    import_training_job,
    run_brain,
)


def main() -> None:
    """Run the local structured agent brain."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--token")
    parser.add_argument("--agent-port", type=int, default=50051)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--authority")
    parser.add_argument("--goal-preset", default="combat")
    parser.add_argument("--mission", default="survive, explore, and kill visible enemies")
    parser.add_argument("--max-states", type=int, default=700)
    parser.add_argument("--memory-path", type=Path, default=Path("agent_memory/e1m1.json"))
    parser.add_argument("--trajectory-jsonl", type=Path, default=Path("trajectories/brain.jsonl"))
    parser.add_argument("--evolve-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-id", default="cautious_combat_v1")
    parser.add_argument("--required-kills", type=int, default=1)
    parser.add_argument(
        "--no-require-level-complete",
        dest="require_level_complete",
        action="store_false",
    )
    parser.add_argument("--no-stop-on-death", dest="stop_on_death", action="store_false")
    parser.add_argument("--no-reconnect", dest="reconnect", action="store_false")
    parser.add_argument("--max-reconnects", type=int, default=5)
    parser.add_argument("--export-job", type=Path, help="write a portable training checkpoint and exit")
    parser.add_argument("--import-job", type=Path, help="import a portable training checkpoint and exit")
    parser.add_argument("--import-destination", type=Path, default=Path("."))
    parser.add_argument(
        "--memory-summary",
        action="store_true",
        help="print memory summary and exit without driving Doom",
    )
    args = parser.parse_args()

    if args.export_job:
        print(
            json.dumps(
                export_training_job(
                    args.export_job,
                    memory_path=args.memory_path,
                    notes_path=Path("agent-notes.md"),
                ),
                sort_keys=True,
            )
        )
        return

    if args.import_job:
        print(
            json.dumps(
                import_training_job(args.import_job, destination=args.import_destination),
                sort_keys=True,
            )
        )
        return

    if args.memory_summary:
        print(json.dumps(AgentMemory.load(args.memory_path).summary(), sort_keys=True))
        return

    config = BrainConfig(
        endpoint=args.endpoint,
        token=args.token,
        agent_port=args.agent_port,
        tls=args.tls,
        authority=args.authority,
        goal_preset=args.goal_preset,
        mission=args.mission,
        max_states=args.max_states,
        memory_path=args.memory_path,
        trajectory_jsonl=args.trajectory_jsonl,
        evolve_runs=args.evolve_runs,
        seed=args.seed,
        policy_id=args.policy_id,
        stop_on_death=args.stop_on_death,
        required_kills=args.required_kills,
        require_level_complete=args.require_level_complete,
        reconnect=args.reconnect,
        max_reconnects=args.max_reconnects,
    )
    print(json.dumps(asyncio.run(run_brain(config)), sort_keys=True))


if __name__ == "__main__":
    main()
