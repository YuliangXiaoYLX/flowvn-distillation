from __future__ import annotations

import hashlib
import importlib
import os
import sys
import zipfile
from pathlib import Path
from typing import Callable


CHALLENGE_ARCHIVE_SHA256 = (
    "a03af38021c3687803ed41abf542d13255b44e7ea9b6634ac2b53ab1e30cebd4"
)
CHALLENGE_SOURCE_MEMBER = "CMRx4DFlowMaskGeneration/ktgaussian.py"
CHALLENGE_SOURCE_SHA256 = (
    "05f6ebb0c0524f7417c68d48b2d89d88adccd32a060961222a29035bda89e041"
)
SUPPORTED_MASK_BACKENDS = ("local", "challenge")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_challenge_archive(path: str | os.PathLike[str]) -> dict[str, str]:
    archive = Path(path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Challenge mask archive not found: {archive}")

    archive_sha256 = _sha256_file(archive)
    if archive_sha256 != CHALLENGE_ARCHIVE_SHA256:
        raise ValueError(
            "Challenge mask archive SHA-256 mismatch: "
            f"actual={archive_sha256}, expected={CHALLENGE_ARCHIVE_SHA256}"
        )
    with zipfile.ZipFile(archive) as package:
        try:
            source = package.read(CHALLENGE_SOURCE_MEMBER)
        except KeyError as exc:
            raise ValueError(
                f"Challenge mask archive is missing {CHALLENGE_SOURCE_MEMBER}"
            ) from exc
    source_sha256 = _sha256_bytes(source)
    if source_sha256 != CHALLENGE_SOURCE_SHA256:
        raise ValueError(
            "Challenge mask source SHA-256 mismatch: "
            f"actual={source_sha256}, expected={CHALLENGE_SOURCE_SHA256}"
        )
    return {
        "archive_path": str(archive),
        "archive_sha256": archive_sha256,
        "source_member": CHALLENGE_SOURCE_MEMBER,
        "source_sha256": source_sha256,
    }


def load_mask_generator(
    backend: str | None = None,
    challenge_archive: str | os.PathLike[str] | None = None,
) -> tuple[Callable, dict[str, str]]:
    selected = str(backend or os.environ.get("FLOWVN_MASK_BACKEND", "local"))
    selected = selected.strip().lower()
    if selected not in SUPPORTED_MASK_BACKENDS:
        raise ValueError(
            f"Unsupported FlowVN mask backend {selected!r}; "
            f"expected one of {SUPPORTED_MASK_BACKENDS}"
        )

    if selected == "local":
        from utils.mask_gen import fun_mask_gen_2d as local_generator

        return local_generator, {"backend": "local"}

    archive_value = challenge_archive or os.environ.get(
        "FLOWVN_CHALLENGE_MASK_ARCHIVE"
    )
    if not archive_value:
        raise ValueError(
            "FLOWVN_CHALLENGE_MASK_ARCHIVE is required for the challenge backend"
        )
    archive_report = verify_challenge_archive(archive_value)
    archive_path = archive_report["archive_path"]

    existing = sys.modules.get("CMRx4DFlowMaskGeneration.ktgaussian")
    if existing is not None:
        existing_file = str(getattr(existing, "__file__", ""))
        if not existing_file.startswith(f"{archive_path}/"):
            raise RuntimeError(
                "CMRx4DFlowMaskGeneration was already imported from an unexpected "
                f"location: {existing_file}"
            )

    sys.path.insert(0, archive_path)
    try:
        package = importlib.import_module("CMRx4DFlowMaskGeneration")
        source_module = importlib.import_module(
            "CMRx4DFlowMaskGeneration.ktgaussian"
        )
    finally:
        try:
            sys.path.remove(archive_path)
        except ValueError:
            pass

    source_file = str(getattr(source_module, "__file__", ""))
    if not source_file.startswith(f"{archive_path}/"):
        raise RuntimeError(
            "Challenge mask module resolved outside the verified archive: "
            f"{source_file}"
        )
    generator = getattr(package, "fun_mask_gen_2d", None)
    if not callable(generator):
        raise TypeError("Challenge mask package does not export fun_mask_gen_2d")
    return generator, {"backend": "challenge", **archive_report}


def require_mask_backend(options: dict) -> None:
    """Reject a real-data recipe when its required mask generator is not active."""
    required = options.get("required_mask_backend")
    if required and MASK_BACKEND_PROVENANCE["backend"] != required:
        raise ValueError(
            f"This recipe requires the {required} mask backend. "
            "Set FLOWVN_MASK_BACKEND=challenge and "
            "FLOWVN_CHALLENGE_MASK_ARCHIVE to your authorized mask archive."
        )


fun_mask_gen_2d, MASK_BACKEND_PROVENANCE = load_mask_generator()
