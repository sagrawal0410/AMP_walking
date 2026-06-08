"""LeRobot robot plugin for AMP velocity deploy on Unitree G1."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lerobot is not installed. Install with: pip install 'lerobot[unitree_g1]'"
    ) from exc

from .amp_g1 import AmpG1Robot
from .config_amp_g1 import AmpG1Config

__all__ = [
    "AmpG1Config",
    "AmpG1Robot",
]
