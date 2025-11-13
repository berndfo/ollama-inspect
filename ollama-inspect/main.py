#!/usr/bin/env python3
"""
Ollama Inspect — minimal GGUF inspector web server.

Usage:
  python main.py /path/to/your_model.gguf [--host 127.0.0.1] [--port 5000]

Starts a local web server that displays the keys extracted from the GGUF file.
"""

import sys
from pathlib import Path
from typing import Optional


def _parse_args(argv: list[str]) -> tuple[Optional[Path], str, int]:
    """Parse CLI args. Returns (model_path, host, port)."""
    model_path: Optional[Path] = None
    host = "127.0.0.1"
    port = 5000

    args = list(argv[1:])
    if args and not args[0].startswith("--"):
        model_path = Path(args.pop(0))

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif a == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                _die(f"Invalid --port value: {args[i + 1]}")
            i += 2
        else:
            _die(f"Unknown argument: {a}")
    return model_path, host, port


def _die(msg: str, code: int = 1) -> "None":
    print(f"Error: {msg}")
    sys.exit(code)


def main(argv: list[str]) -> int:
    model_path, host, port = _parse_args(argv)

    if model_path is None:
        print("Usage: python main.py /path/to/your_model.gguf [--host 127.0.0.1] [--port 5000]")
        return 2

    if not model_path.exists():
        _die(f"File not found: {model_path}")

    # Import here to keep main module lightweight for other tooling
    try:
        from webapp import create_app  # type: ignore
    except Exception as e:
        _die(f"Failed to import web application components: {e}")

    app = create_app(model_path)
    # Run development server (sufficient for local inspection use case)
    app.run(host=host, port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
