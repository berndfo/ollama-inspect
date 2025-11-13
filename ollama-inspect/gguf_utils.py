"""
Utilities for reading GGUF files.

We only need to extract the available keys from the file's fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


class GGUFLoadError(Exception):
    pass


def extract_keys(model_path: Path) -> List[str]:
    """Return a sorted list of keys present in the GGUF file.

    Raises GGUFLoadError if the file cannot be opened or parsed.
    """
    try:
        import gguf  # type: ignore
    except Exception as e:  # pragma: no cover - optional friendly message
        raise GGUFLoadError(
            "The 'gguf' package is required. Install it via 'pip install gguf'."
        ) from e

    try:
        reader = gguf.GGUFReader(str(model_path))
    except Exception as e:
        raise GGUFLoadError(f"Could not open GGUF file '{model_path}': {e}") from e

    try:
        keys: Iterable[str] = reader.fields.keys()
        return sorted(list(keys))
    except Exception as e:
        raise GGUFLoadError(f"Failed to read fields from '{model_path}': {e}") from e


def _coerce_to_python(obj: Any) -> Any:
    """Best-effort conversion of GGUF field values to JSON-serializable Python types.

    - Keeps primitives (str, int, float, bool, None) as-is.
    - Bytes are converted to a safe text placeholder with size info.
    - Lists/tuples are converted element-wise (limited depth).
    - Dicts are converted value-wise (limited depth).
    - Fallback: string representation via repr().
    """
    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Bytes: do not expose raw bytes; provide a description instead
    if isinstance(obj, (bytes, bytearray)):
        return {
            "__type__": "bytes",
            "length": len(obj),
            "preview_hex": obj[:32].hex(),  # small hex preview
        }

    # Containers (limit recursion depth to avoid pathological structures)
    def _convert_list(seq: Iterable[Any], depth: int = 0) -> Any:
        if depth > 2:
            return f"<list depth={depth} length={sum(1 for _ in seq)}>"
        out = []
        count = 0
        for item in seq:
            out.append(_coerce_to_python(item))
            count += 1
            if count > 1024:  # hard cap to avoid huge payloads
                out.append("<truncated>")
                break
        return out

    if isinstance(obj, (list, tuple)):
        return _convert_list(obj)

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        count = 0
        for k, v in obj.items():
            out[str(k)] = _coerce_to_python(v)
            count += 1
            if count > 2048:  # safety cap
                out["<truncated>"] = True
                break
        return out

    # Numpy types (if present) — convert to Python scalars/lists without importing numpy explicitly
    # Detect common numpy scalar attribute
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    # Detect array-like tolist
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return obj.tolist()
        except Exception:
            pass

    # Objects with a "value" attribute may store the underlying primitive
    for attr in ("get_value", "value", "data"):
        try:
            v = getattr(obj, attr)
            v = v() if callable(v) else v
            return _coerce_to_python(v)
        except Exception:
            continue

    # Fallback to repr for unknown types
    try:
        return repr(obj)
    except Exception:
        return "<unserializable>"


def extract_key_values(model_path: Path) -> Dict[str, Any]:
    """Return a mapping of key -> value (JSON-serializable best-effort).

    Raises GGUFLoadError if the file cannot be opened or parsed.
    """
    try:
        import gguf  # type: ignore
    except Exception as e:  # pragma: no cover - optional friendly message
        raise GGUFLoadError(
            "The 'gguf' package is required. Install it via 'pip install gguf'."
        ) from e

    try:
        reader = gguf.GGUFReader(str(model_path))
    except Exception as e:
        raise GGUFLoadError(f"Could not open GGUF file '{model_path}': {e}") from e

    try:
        items: Dict[str, Any] = {}
        for k, field in reader.fields.items():
            # Best-effort attempt to extract the underlying value
            value = None
            # Try common accessors first
            for attr in ("get_value", "value", "data"):
                try:
                    v = getattr(field, attr)
                    value = v() if callable(v) else v
                    break
                except Exception:
                    continue
            if value is None:
                # Some gguf versions store directly accessible value or have useful repr
                value = field
            items[str(k)] = _coerce_to_python(value)
        return items
    except Exception as e:
        raise GGUFLoadError(f"Failed to read fields from '{model_path}': {e}") from e
