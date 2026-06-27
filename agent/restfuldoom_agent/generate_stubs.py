"""Generates Python gRPC stubs from the shared RESTful Doom proto."""

from __future__ import annotations

import argparse
from pathlib import Path


def repo_root() -> Path:
    """Returns the repository root."""
    return Path(__file__).resolve().parents[2]


def generate(output_dir: Path | None = None) -> Path:
    """Generates Python stubs and returns their output directory."""
    from grpc_tools import protoc

    root = repo_root()
    proto_root = root / "proto"
    proto = proto_root / "restfuldoom" / "v1" / "agent.proto"
    out = output_dir or Path(__file__).resolve().parent / "generated"
    out.mkdir(parents=True, exist_ok=True)

    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto_root}",
            f"--python_out={out}",
            f"--grpc_python_out={out}",
            str(proto),
        ]
    )
    if result != 0:
        raise RuntimeError(f"protoc failed with exit code {result}")
    return out


def main() -> None:
    """Runs the stub generator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(generate(args.output_dir))


if __name__ == "__main__":
    main()
