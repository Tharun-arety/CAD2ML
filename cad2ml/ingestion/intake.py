"""Untrusted-upload intake: size limits, extension allowlist, content sniffing, hashing.

Nothing here parses CAD geometry; it only decides whether bytes may be handed to the
isolated parser. Source bytes are stored content-addressed by SHA-256 and are never
modified afterwards.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePath

from cad2ml.config import IngestionConfig
from cad2ml.errors import PipelineError

STEP_MAGIC = b"ISO-10303-21;"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def step_files(directory: Path) -> list[Path]:
    """STEP files in a directory, matched case-insensitively on the allowlisted extensions (.step/.stp)."""
    exts = {e.lower() for e in IngestionConfig().allowed_extensions}
    return sorted(p for p in Path(directory).glob("*") if p.is_file() and p.suffix.lower() in exts)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_stream(chunks: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def sanitize_filename(name: str, max_len: int = 128) -> str:
    """Return a display-safe basename. Never used to build storage paths."""
    base = PurePath(name.replace("\\", "/")).name
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode()
    base = _SAFE.sub("_", base).strip("._") or "unnamed"
    if len(base) > max_len:
        stem, dot, ext = base.rpartition(".")
        base = (stem[: max_len - len(ext) - 1] + dot + ext) if dot and len(ext) <= 8 else base[:max_len]
    return base


@dataclass(frozen=True)
class IntakeResult:
    filename: str
    sha256: str
    size_bytes: int
    step_schema: str | None
    originating_system: str | None


def _header_field(head: str, pattern: str) -> str | None:
    m = re.search(pattern, head, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()[:200] or None


def inspect_upload(filename: str, data: bytes, cfg: IngestionConfig) -> IntakeResult:
    safe = sanitize_filename(filename)
    if len(data) == 0:
        raise PipelineError("EMPTY_FILE", "uploaded file is empty", "intake")
    if len(data) > cfg.max_file_bytes:
        raise PipelineError("FILE_TOO_LARGE", f"{len(data)} bytes > limit {cfg.max_file_bytes}", "intake")
    ext = ("." + safe.rsplit(".", 1)[-1].lower()) if "." in safe else ""
    if ext not in cfg.allowed_extensions:
        raise PipelineError("UNSUPPORTED_EXTENSION", f"extension {ext or '<none>'!r} not allowed", "intake")
    if not data.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(STEP_MAGIC):
        raise PipelineError("NOT_STEP_CONTENT", "missing ISO-10303-21 header", "intake")
    head = data[:65536].decode("latin-1", errors="replace")
    schema = _header_field(head, r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'")
    fn = re.search(r"FILE_NAME\s*\((.*?)\);", head, re.DOTALL)
    origin = None
    if fn:
        strings = re.findall(r"'([^']*)'", fn.group(1))
        # FILE_NAME(name, time_stamp, (author), (organization), preprocessor_version, originating_system, auth)
        if len(strings) >= 6:
            origin = strings[-2][:200] or None
    return IntakeResult(safe, sha256_bytes(data), len(data), schema, origin)
