from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from flask import Flask, jsonify, render_template, request

from gguf_utils import GGUFLoadError, extract_all
from path_utils import (
    get_blobs_root,
    normalize_candidate_filename,
    is_valid_blob_filename,
    build_model_path_from_filename,
    digest_to_filename,
)
import json


def create_app() -> Flask:
    """Application factory.

    Defers loading GGUF keys until an endpoint that needs them is invoked.
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=None,
    )

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

        def _is_valid_manifest(m: Dict[str, Any]) -> bool:
            try:
                # Top-level mandatory fields
                if m.get("schemaVersion") != 2:
                    return False
                if m.get("mediaType") != "application/vnd.docker.distribution.manifest.v2+json":
                    return False

                # Config object validation
                cfg = m.get("config")
                if not isinstance(cfg, dict):
                    return False
                if cfg.get("mediaType") != "application/vnd.docker.container.image.v1+json":
                    return False
                digest = cfg.get("digest")
                if not isinstance(digest, str) or not digest.startswith("sha256:"):
                    return False
                size = cfg.get("size")
                if not isinstance(size, int) or size < 0:
                    return False

                # Layers validation
                layers = m.get("layers")
                if not isinstance(layers, list) or not layers:
                    return False
                allowed_layer_media_types = {
                    "application/vnd.ollama.image.model",
                    "application/vnd.ollama.image.template",
                    "application/vnd.ollama.image.license",
                    # Accept extended manifests that include params layer
                    "application/vnd.ollama.image.params",
                }
                # Must contain at least the model layer
                has_model_layer = False
                for layer in layers:
                    if not isinstance(layer, dict):
                        return False
                    lmt = layer.get("mediaType")
                    if lmt not in allowed_layer_media_types:
                        return False
                    if lmt == "application/vnd.ollama.image.model":
                        has_model_layer = True
                    ldigest = layer.get("digest")
                    if not isinstance(ldigest, str) or not ldigest.startswith("sha256:"):
                        return False
                    lsize = layer.get("size")
                    if not isinstance(lsize, int) or lsize < 0:
                        return False
                if not has_model_layer:
                    return False

                return True
            except Exception:
                return False

        try:
            if manifests_dir.exists():
                for mf in sorted(manifests_dir.glob("**/*")):
                    if not mf.is_file():
                        continue
                    try:
                        with mf.open("r", encoding="utf-8") as f:
                            manifest = json.load(f)
                    except Exception:
                        # Unreadable manifest; skip — only show files adhering to the schema
                        continue

                    # Only process and show files that adhere to the required JSON schema
                    if not _is_valid_manifest(manifest):
                        continue

                    # Manifest semantics (best-effort):
                    # Expect fields like: schemaVersion, model (name:tag), layers: [{digest, size}]
                    name = manifest.get("model") or manifest.get("name") or mf.stem
                    # Prefix the model name with the parent directory name of the JSON file
                    try:
                        parent_dir = mf.parent.name
                        # Avoid double-prefixing if name already appears to include the parent dir
                        if parent_dir and not (
                            str(name).startswith(parent_dir + "/")
                            or str(name).startswith(parent_dir + ":")
                            or ("/" in str(name) and str(name).split("/", 1)[0] == parent_dir)
                        ):
                            display_name = f"{parent_dir}/{name}"
                        else:
                            display_name = str(name)
                    except Exception:
                        display_name = str(name)
                    layers = manifest.get("layers") or []
                    # Extract the declared model size from the manifest: look for the
                    # layer with mediaType "application/vnd.ollama.image.model" and
                    # use its "size" (bytes). If multiple exist, take the first.
                    model_bytes: Optional[int] = None
                    model_blob_filename: Optional[str] = None
                    try:
                        for _layer in layers:
                            if (
                                isinstance(_layer, dict)
                                and _layer.get("mediaType") == "application/vnd.ollama.image.model"
                            ):
                                sz_val = _layer.get("size")
                                if isinstance(sz_val, int) and sz_val >= 0:
                                    model_bytes = sz_val
                                # Derive filename (sha256-<hex>) from digest via util
                                dg = _layer.get("digest")
                                if isinstance(dg, str):
                                    model_blob_filename = digest_to_filename(dg)
                                break
                    except Exception:
                        model_bytes = None
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
                                    "filename": digest_to_filename(digest) if isinstance(digest, str) else None,
                                    "size": size,
                                    "path": str(blob_path),
                                    "present": True,
                                    "actual_size": sz,
                                })
                            except Exception:
                                missing += 1
                                resolved_layers.append({
                                    "digest": digest,
                                    "filename": digest_to_filename(digest) if isinstance(digest, str) else None,
                                    "size": size,
                                    "path": str(blob_path),
                                    "present": False,
                                })
                        else:
                            missing += 1
                            resolved_layers.append({
                                "digest": digest,
                                "filename": digest_to_filename(digest) if isinstance(digest, str) else None,
                                "size": size,
                                "path": str(blob_path) if blob_path else None,
                                "present": False,
                            })

                    # Pre-format GB value (decimal GB as requested).
                    model_gb: Optional[str]
                    if isinstance(model_bytes, int):
                        gb = model_bytes / 1_000_000_000.0
                        model_gb = f"{gb:.2f}"
                    else:
                        model_gb = None

                    entries.append({
                        "name": display_name,
                        "path": str(mf),
                        "created": mf.stat().st_mtime,
                        "total_size": present_size,
                        "layers": resolved_layers,
                        "missing": missing,
                        "model_bytes": model_bytes,
                        "model_gb": model_gb,
                        "manifests_root": str(manifests_dir),
                        "blobs_root": str(blobs_dir),
                        "model_blob_filename": model_blob_filename,
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

    @app.get("/model/metadata")
    def get_model_metadata():  # type: ignore[override]
        # Split into blobs root and filename passed as GET parameter
        filename = request.args.get("filename", type=str)
        blobs_root = get_blobs_root()
        if not filename:
            # Render a simple error page when filename is missing
            return render_template(
                "index.html",
                model_path=str(blobs_root),
                items=[],
                keys_count=0,
                error="Missing required 'filename' parameter (expected like sha256-<hex>).",
            )
        # Normalize/validate filename
        fn = normalize_candidate_filename(filename)
        if not fn.startswith("sha256-"):
            return render_template(
                "index.html",
                model_path=str(blobs_root / fn),
                items=[],
                keys_count=0,
                error="Invalid filename. It must start with 'sha256-'.",
            )
        # Security: disallow path traversal
        if not is_valid_blob_filename(fn):
            return render_template(
                "index.html",
                model_path=str(blobs_root / fn),
                items=[],
                keys_count=0,
                error="Invalid filename.",
            )
        model_path = build_model_path_from_filename(fn)
        
        # Build items with preview and full JSON once for the template
        kv: Dict[str, Any]
        try:
            kv = extract_all(model_path)
        except Exception as e:
            # Render a simple error page
            return render_template(
                "index.html",
                model_path=model_path,
                items=[],
                keys_count=0,
                error=str(e),
            )
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
            model_path=model_path,
            items=items,
            keys_count=len(kv),
            filename=fn,
        )

    @app.get("/api/keys")
    def api_keys():  # type: ignore[override]
        filename = request.args.get("filename", type=str)
        blobs_root = get_blobs_root()
        if not filename:
            return jsonify({"error": "Missing 'filename' parameter", "blobs_root": str(blobs_root)}), 400
        fn = normalize_candidate_filename(filename)
        if not is_valid_blob_filename(fn):
            return jsonify({"error": "Invalid filename", "filename": filename}), 400
        model_path = build_model_path_from_filename(fn)
        
        # Backwards-compatible endpoint: keys only
        try:
            kv: Dict[str, Any] = extract_all(model_path)
            return jsonify(
                {
                    "model_path": model_path,
                    "count": len(kv),
                    "keys": sorted(kv.keys()),
                    "filename": fn,
                }
            )
        except Exception as e:
            return jsonify({"error": str(e), "model_path": model_path}), 500

    @app.get("/api/items")
    def api_items():  # type: ignore[override]
        filename = request.args.get("filename", type=str)
        blobs_root = get_blobs_root()
        if not filename:
            return jsonify({"error": "Missing 'filename' parameter", "blobs_root": str(blobs_root)}), 400
        fn = normalize_candidate_filename(filename)
        if not is_valid_blob_filename(fn):
            return jsonify({"error": "Invalid filename", "filename": filename}), 400
        model_path = build_model_path_from_filename(fn)
        
        try:
            kv: Dict[str, Any] = extract_all(model_path)
        except Exception as e:
            return jsonify({"error": str(e), "model_path": model_path}), 500
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
                "model_path": model_path,
                "count": len(kv),
                "items": data,
                "filename": fn,
            }
        )

    return app
