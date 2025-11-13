from __future__ import annotations

from pathlib import Path
from typing import List

from flask import Flask, jsonify, render_template

from gguf_utils import GGUFLoadError, extract_keys


def create_app(model_path: Path) -> Flask:
    """Application factory.

    Loads GGUF keys once at startup and serves them via HTML and JSON.
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=None,
    )

    try:
        keys: List[str] = extract_keys(model_path)
    except GGUFLoadError as e:
        # Fail fast with a clear startup error
        raise RuntimeError(str(e)) from e

    app.config["GGUF_PATH"] = str(model_path)
    app.config["GGUF_KEYS"] = keys

    @app.get("/")
    def index():  # type: ignore[override]
        return render_template(
            "index.html",
            model_path=app.config["GGUF_PATH"],
            keys=app.config["GGUF_KEYS"],
            keys_count=len(app.config["GGUF_KEYS"]),
        )

    @app.get("/api/keys")
    def api_keys():  # type: ignore[override]
        return jsonify(
            {
                "model_path": app.config["GGUF_PATH"],
                "count": len(app.config["GGUF_KEYS"]),
                "keys": app.config["GGUF_KEYS"],
            }
        )

    return app
