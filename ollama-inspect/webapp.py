from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, render_template

from gguf_utils import GGUFLoadError, extract_all
import json


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
        # Load keys and values in one go to avoid double-reading the GGUF file
        key_values = extract_all(model_path)
    except GGUFLoadError as e:
        # Fail fast with a clear startup error
        raise RuntimeError(str(e)) from e

    app.config["GGUF_PATH"] = str(model_path)
    app.config["GGUF_KEY_VALUES"] = key_values

    def _make_preview(value: Any, max_chars: int = 240, max_lines: int = 6) -> tuple[str, bool]:
        """Create a compact preview and a boolean indicating if it was truncated.

        Returns (preview_text, is_truncated).
        """
        s = str(value)

        lines = s.splitlines() or [s]
        truncated = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True
        s2 = "\n".join(lines)
        if len(s2) > max_chars:
            s2 = s2[:max_chars]
            truncated = True
        if truncated:
            s2 += " …"
        return s2, truncated

    @app.get("/")
    def index():  # type: ignore[override]
        # Build items with preview and full JSON once for the template
        kv: Dict[str, Any] = app.config["GGUF_KEY_VALUES"]
        items: List[Tuple[str, str, str, bool]] = []  # (key, preview, full_json, expandable)
        for k in sorted(kv.keys()):
            v = kv[k]
            preview, truncated = _make_preview(v)
            try:
                full_json = json.dumps(v, ensure_ascii=False, indent=2)
            except Exception:
                full_json = str(v)
            items.append((k, preview, full_json, truncated))

        return render_template(
            "index.html",
            model_path=app.config["GGUF_PATH"],
            items=items,
            keys_count=len(app.config["GGUF_KEY_VALUES"]),
        )

    @app.get("/api/keys")
    def api_keys():  # type: ignore[override]
        # Backwards-compatible endpoint: keys only
        kv: Dict[str, Any] = app.config["GGUF_KEY_VALUES"]
        return jsonify(
            {
                "model_path": app.config["GGUF_PATH"],
                "count": len(kv),
                "keys": sorted(kv.keys()),
            }
        )

    @app.get("/api/items")
    def api_items():  # type: ignore[override]
        kv: Dict[str, Any] = app.config["GGUF_KEY_VALUES"]
        data = []
        for k in sorted(kv.keys()):
            v = kv[k]
            preview, truncated = _make_preview(v)
            data.append({
                "key": k,
                "value": v,
                "preview": preview,
                "expandable": bool(truncated),
            })
        return jsonify(
            {
                "model_path": app.config["GGUF_PATH"],
                "count": len(kv),
                "items": data,
            }
        )

    return app
