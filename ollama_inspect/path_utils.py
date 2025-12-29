from __future__ import annotations

from pathlib import Path


def get_blobs_root() -> Path:
    """Return the default Ollama blobs directory under the user's home.

    Typically: ~/.ollama/models/blobs
    """
    home = Path.home()
    return home / ".ollama" / "models" / "blobs"


def normalize_candidate_filename(raw: str) -> str:
    """Normalize a user-provided filename/digest into a candidate blob filename.

    Rules:
    - Trim whitespace.
    - If it starts with 'sha256:' → convert to 'sha256-<hex>'.
    - If it already starts with 'sha256-' → keep as-is.
    - Otherwise return unchanged (validation is performed separately).
    """
    fn = (raw or "").strip()
    if fn.startswith("sha256:"):
        fn = "sha256-" + fn.split(":", 1)[1]
    return fn


def is_valid_blob_filename(fn: str) -> bool:
    """Check if the candidate blob filename is safe and well-formed.

    Requirements:
    - Must start with 'sha256-'
    - Must not contain path separators or traversal
    """
    if not fn.startswith("sha256-"):
        return False
    if "/" in fn or ".." in fn or "\\" in fn:
        return False
    return True


def digest_to_filename(digest: str) -> str:
    """Convert a digest like 'sha256:<hex>' or '<hex>' into 'sha256-<hex>'.

    This is lenient and intended for internal derivations based on manifests.
    """
    if not isinstance(digest, str):
        return str(digest)
    if digest.startswith("sha256:"):
        return "sha256-" + digest.split(":", 1)[1]
    if digest.startswith("sha256-"):
        return digest
    return "sha256-" + digest


def build_model_path_from_filename(filename: str) -> str:
    """Join the blobs root and the provided filename and return as string."""
    return str(get_blobs_root() / filename)
