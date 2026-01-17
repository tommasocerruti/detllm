"""detLLM package."""

from detllm.api import check, run
from detllm.core.env import capture_env
from detllm.version import __version__

__all__ = ["__version__", "capture_env", "run", "check"]
