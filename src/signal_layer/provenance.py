"""Source identity for safe historical replay and resumable experiments."""

import hashlib
from pathlib import Path


def code_fingerprint():
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()
