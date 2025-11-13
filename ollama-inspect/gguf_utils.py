"""
Utilities for reading GGUF files.

We only need to extract the available keys from the file's fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


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
