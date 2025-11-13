from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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

    # New default home page: list Ollama files from ~/.ollama by manifest semantics
    @app.get("/")
    def ollama_home():  # type: ignore[override]
        registry = "registry.ollama.ai"
        
        home = Path.home()
        base = home / ".ollama"
        models_dir = base / "models"
        manifests_dir = models_dir / "manifests" / registry / "library"
        blobs_dir = models_dir / "blobs"

        entries: List[Dict[str, Any]] = []
        error: Optional[str] = None

        try:
            if manifests_dir.exists():
                for mf in sorted(manifests_dir.glob("**/*")):
                    if not mf.is_file():
                        continue
                    try:
                        with mf.open("r", encoding="utf-8") as f:
                            manifest = json.load(f)
                    except Exception:
                        # unreadable manifest; show as-is
                        entries.append({
                            "name": mf.name,
                            "path": str(mf),
                            "created": mf.stat().st_mtime,
                            "total_size": None,
                            "layers": [],
                            "missing": None,
                            "error": "Failed to parse manifest JSON",
                        })
                        continue

                    # Manifest semantics (best-effort):
                    # Expect fields like: schemaVersion, model (name:tag), layers: [{digest, size}]
                    name = manifest.get("model") or manifest.get("name") or mf.stem
                    layers = manifest.get("layers") or []
                    present_size = 0
                    total_layers = 0
                    missing = 0
                    resolved_layers: List[Dict[str, Any]] = []
                    for layer in layers:
                        digest = layer.get("digest") or layer.get("sha256") or layer.get("id")
                        size = layer.get("size")
                        total_layers += 1
                        blob_path = None
                        if isinstance(digest, str):
                            cand1 = blobs_dir / digest
                            # common pattern: sha256-<hex>
                            if not cand1.exists():
                                if not digest.startswith("sha256-"):
                                    cand2 = blobs_dir / f"sha256-{digest}"
                                else:
                                    cand2 = blobs_dir / digest.replace("sha256:", "sha256-")
                                blob_path = cand2 if cand2.exists() else cand1
                            else:
                                blob_path = cand1
                        if blob_path and blob_path.exists():
                            try:
                                sz = blob_path.stat().st_size
                                present_size += sz
                                resolved_layers.append({
                                    "digest": digest,
                                    "size": size,
                                    "path": str(blob_path),
                                    "present": True,
                                    "actual_size": sz,
                                })
                            except Exception:
                                missing += 1
                                resolved_layers.append({
                                    "digest": digest,
                                    "size": size,
                                    "path": str(blob_path),
                                    "present": False,
                                })
                        else:
                            missing += 1
                            resolved_layers.append({
                                "digest": digest,
                                "size": size,
                                "path": str(blob_path) if blob_path else None,
                                "present": False,
                            })

                    entries.append({
                        "name": name,
                        "path": str(mf),
                        "created": mf.stat().st_mtime,
                        "total_size": present_size,
                        "layers": resolved_layers,
                        "missing": missing,
                        "manifests_root": str(manifests_dir),
                        "blobs_root": str(blobs_dir),
                    })
            else:
                error = f"Manifests directory not found: {manifests_dir}"
        except Exception as e:
            error = f"Failed to scan Ollama directory: {e}"

        return render_template(
            "ollama_index.html",
            ollama_root=str(base),
            manifests_root=str(manifests_dir),
            blobs_root=str(blobs_dir),
            entries=entries,
            error=error,
        )

    # Previous index (GGUF model metadata) is now available at /model/metadata
    @app.get("/model/metadata")
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
