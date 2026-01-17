"""Environment snapshot capture for DetLLM."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from typing import Any

from detllm.version import __version__

ENV_VARS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDNN_DETERMINISTIC",
    "CUDNN_BENCHMARK",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
)


def _get_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _torch_device_info() -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None

    device_info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available(),
        "device_type": "cpu",
    }

    if device_info["cuda_available"]:
        device_info["device_type"] = "cuda"
        device_info["cuda_version"] = torch.version.cuda
        device_info["device_count"] = torch.cuda.device_count()
        device_info["devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    elif device_info["mps_available"]:
        device_info["device_type"] = "mps"

    # TODO: Capture richer device metadata (driver, compute capability) when stable.
    return device_info


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_env() -> dict[str, Any]:
    """Capture a deterministic environment snapshot."""

    snapshot: dict[str, Any] = {
        "schema_version": "1.0",
        "detllm_version": __version__,
        "artifact_type": "env_snapshot",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,  # TODO: Review sensitivity of exposing paths.
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch": {
            "version": _get_version("torch"),
        },
        "transformers": {
            "version": _get_version("transformers"),
        },
        "device": _torch_device_info(),
        "env_vars": {name: os.environ.get(name) for name in ENV_VARS},
        # TODO: Consider capturing driver/toolkit versions and CPU metadata.
        # TODO: Add a redaction/allowlist mechanism for sensitive fields.
    }

    fingerprint_payload = dict(snapshot)
    snapshot["fingerprint"] = _canonical_fingerprint(fingerprint_payload)
    return snapshot
