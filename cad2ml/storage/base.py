"""Artifact storage abstraction.

v1 ships only ``LocalFSStore``. The protocol is intentionally small so an
S3-compatible backend can implement it later without touching pipeline code.
Keys are always POSIX-style relative paths; absolute paths and ``..`` segments are
rejected before any backend sees them.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable

_KEY_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")


class StorageKeyError(ValueError):
    pass


def validate_key(key: str) -> str:
    if not key or not _KEY_RE.match(key):
        raise StorageKeyError(f"illegal characters in storage key: {key!r}")
    if PurePosixPath(key).is_absolute() or any(part in ("..", ".", "") for part in key.split("/")):
        raise StorageKeyError(f"unsafe storage key: {key!r}")
    return key


@runtime_checkable
class ArtifactStore(Protocol):
    def put_bytes(self, key: str, data: bytes) -> int: ...
    def put_file(self, key: str, src: Path) -> int: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def size(self, key: str) -> int: ...
    def list(self, prefix: str) -> Iterator[str]: ...
    def local_path(self, key: str) -> Path: ...
    def delete_prefix(self, prefix: str) -> None: ...
    def promote_prefix(self, src_prefix: str, dst_prefix: str) -> None: ...


class LocalFSStore:
    """Filesystem store with atomic writes (temp file + ``os.replace``)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        validate_key(key)
        p = (self.root / key).resolve()
        if self.root != p and self.root not in p.parents:
            raise StorageKeyError(f"key escapes storage root: {key!r}")
        return p

    def put_bytes(self, key: str, data: bytes) -> int:
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, dst)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return len(data)

    def put_file(self, key: str, src: Path) -> int:
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / f".tmp-{uuid.uuid4().hex}"
        try:
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return dst.stat().st_size

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def list(self, prefix: str) -> Iterator[str]:
        base = self._path(prefix)
        if not base.exists():
            return
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.name.startswith(".tmp-"):
                yield p.relative_to(self.root).as_posix()

    def local_path(self, key: str) -> Path:
        return self._path(key)

    def delete_prefix(self, prefix: str) -> None:
        p = self._path(prefix)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()

    def promote_prefix(self, src_prefix: str, dst_prefix: str) -> None:
        """Atomically move a fully written staging directory into its final key.

        Readers only ever observe either no directory or a complete one. If the
        destination already exists (idempotent rerun) the staging copy is discarded.
        """
        src, dst = self._path(src_prefix), self._path(dst_prefix)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(src, ignore_errors=True)
            return
        os.replace(src, dst)
