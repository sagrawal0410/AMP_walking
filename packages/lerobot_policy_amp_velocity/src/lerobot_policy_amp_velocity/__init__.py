"""LeRobot plugin for AMP velocity locomotion policies."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lerobot is not installed. Install with: pip install 'lerobot[unitree_g1]'"
    ) from exc

from .configuration_amp_velocity import AmpVelocityPolicyConfig
from .modeling_amp_velocity import AmpVelocityPolicy
from .processor_amp_velocity import make_amp_velocity_pre_post_processors

__all__ = [
    "AmpVelocityPolicyConfig",
    "AmpVelocityPolicy",
    "make_amp_velocity_pre_post_processors",
]
