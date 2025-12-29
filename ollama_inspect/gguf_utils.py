"""
Utilities for reading GGUF files.

We only need to extract the available keys from the file's fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from gguf import GGUFReader, GGUFValueType


class GGUFLoadError(Exception):
    pass


def extract_keys(model_path: Path) -> List[str]:
    """Return a sorted list of keys present in the GGUF file.

    Raises GGUFLoadError if the file cannot be opened or parsed.
    """
    keys, _ = extract_all(model_path)
    return keys


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
    _, values = extract_all(model_path)
    return values


def get_file_host_endian(reader: GGUFReader) -> tuple[str, str]:
    file_endian = reader.endianess.name
    if reader.byte_order == 'S':
        host_endian = 'BIG' if file_endian == 'LITTLE' else 'LITTLE'
    else:
        host_endian = file_endian
    return (host_endian, file_endian)

console_log = False

def extract_all(model_path: Path) -> Tuple[List[str], Dict[str, Any]]:
    """Open the GGUF file once and return (sorted_keys_list, values_dict).

    - Values are converted to JSON-serializable forms best-effort via _coerce_to_python.
    - Raises GGUFLoadError if the file cannot be opened or parsed.
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

    host_endian, file_endian = get_file_host_endian(reader)
    
    if console_log:
        print(f'* File is {file_endian} endian, script is running on a {host_endian} endian host.')  # noqa: NP100
        print(f'* Dumping {len(reader.fields)} key/value pair(s)')  # noqa: NP100
    
    items: Dict[str, Any] = {}
    
    for n, field in enumerate(reader.fields.values(), 1):
        if not field.types:
            pretty_type = 'N/A'
        elif field.types[0] == GGUFValueType.ARRAY:
            nest_count = len(field.types) - 1
            pretty_type = '[' * nest_count + str(field.types[-1].name) + ']' * nest_count
        else:
            pretty_type = str(field.types[-1].name)

        log_message = f'  {n:5}: {pretty_type:10} | {len(field.data):8} | {field.name}'
        if field.types:
            curr_type = field.types[0]
            if curr_type == GGUFValueType.STRING:
                content = field.contents()
                if len(content) > 60:
                    content = content[:57] + '...'
                log_message += ' = {0}'.format(repr(content))
            elif curr_type in reader.gguf_scalar_to_np:
                log_message += ' = {0}'.format(field.contents())
            else:
                content = repr(field.contents(slice(6)))
                if len(field.data) > 6:
                    content = content[:-1] + ', ...]'
                log_message += ' = {0}'.format(content)
        if console_log:
            print(log_message)  # noqa: NP100
        
        # Populate items dict with full, properly typed values (not truncated like the log)
        try:
            if field.types:
                curr_type = field.types[0]
                # Strings
                if curr_type == GGUFValueType.STRING:
                    value = field.contents()
                # Scalars (numeric/bool)
                elif curr_type in reader.gguf_scalar_to_np:
                    value = field.contents()
                # Arrays and other container-like fields
                else:
                    value = field.contents()
            else:
                # No type metadata; fall back to raw contents
                value = field.contents()
        except Exception as _:
            # As a last resort, attempt to store a representation
            value = repr(getattr(field, 'data', '<no-data>'))

        items[str(field.name)] = _coerce_to_python(value)


    return items
