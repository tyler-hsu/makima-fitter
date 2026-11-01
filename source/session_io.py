"""Portable, self-contained Makima Fitter session archives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
from typing import Any


SESSION_SCHEMA = "makima-fitter-session"
SESSION_SCHEMA_VERSION = 1
SESSION_EXTENSION = ".makima"
SESSION_METADATA_MEMBER = "session.json"
SESSION_SOURCE_MEMBER = "source/input-data"


def save_session_archive(
    destination: str | Path,
    source_path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """Atomically write metadata and an exact copy of the source CV."""
    destination = Path(destination)
    if destination.suffix.lower() != SESSION_EXTENSION:
        destination = destination.with_suffix(SESSION_EXTENSION)
    source_path = Path(source_path)
    source_bytes = source_path.read_bytes()
    source_member = SESSION_SOURCE_MEMBER + source_path.suffix.lower()
    document = dict(metadata)
    document.update(
        {
            "schema": SESSION_SCHEMA,
            "schema_version": SESSION_SCHEMA_VERSION,
            "source_member": source_member,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}-",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        with zipfile.ZipFile(
            temporary_name, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr(
                SESSION_METADATA_MEMBER,
                json.dumps(document, indent=2, ensure_ascii=False),
            )
            archive.writestr(source_member, source_bytes)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass
    return destination


def load_session_archive(path: str | Path) -> tuple[dict[str, Any], bytes, str]:
    """Validate and read a session without extracting arbitrary archive paths."""
    path = Path(path)
    with zipfile.ZipFile(path, "r") as archive:
        try:
            metadata = json.loads(archive.read(SESSION_METADATA_MEMBER).decode("utf-8"))
        except (KeyError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("This is not a valid Makima Fitter session file") from error
        if metadata.get("schema") != SESSION_SCHEMA:
            raise ValueError("This file is not a Makima Fitter session")
        version = int(metadata.get("schema_version", 0))
        if version > SESSION_SCHEMA_VERSION:
            raise ValueError(
                "This session was created by a newer Makima Fitter version"
            )
        member = str(metadata.get("source_member", ""))
        if not member.startswith(SESSION_SOURCE_MEMBER):
            raise ValueError("The session's embedded source-data entry is invalid")
        try:
            source_bytes = archive.read(member)
        except KeyError as error:
            raise ValueError("The session does not contain its source CV data") from error
    expected_hash = str(metadata.get("source_sha256", ""))
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if expected_hash and actual_hash != expected_hash:
        raise ValueError("The embedded source CV failed its integrity check")
    return metadata, source_bytes, Path(member).suffix
