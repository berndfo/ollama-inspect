#!/usr/bin/env python3
"""
Ollama Inspect — minimal GGUF inspector web server.

Usage:
  python -m ollama_inspect /path/to/your_model.gguf [--host 127.0.0.1] [--port 13655]

Starts a local web server that displays the keys extracted from the GGUF file.
"""

import sys
from pathlib import Path
from typing import Optional


def _parse_args(argv: list[str]) -> tuple[Optional[Path], str, int]:
    """Parse CLI args. Returns (model_path, host, port)."""
    host = "127.0.0.1"
    port = 13655

    args = list(argv[1:])

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
    return host, port


def _die(msg: str, code: int = 1) -> "None":
    print(f"Error: {msg}")
    sys.exit(code)


def main(argv: list[str] = None) -> int:
    if argv is None:
        argv = sys.argv
    host, port = _parse_args(argv)

    # Import here to keep main module lightweight for other tooling
    try:
        from .webapp import create_app
    except Exception as e:
        _die(f"Failed to import web application components: {e}")

    app = create_app()
    # Run development server (sufficient for local inspection use case)
    app.run(host=host, port=port, debug=False)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
