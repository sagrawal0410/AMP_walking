"""Configuration for AMP velocity locomotion policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from .constants import G1_JOINT_NAMES, NUM_JOINTS, TOTAL_OBS, joint_q_key


@PreTrainedConfig.register_subclass("amp_velocity")
@dataclass
class AmpVelocityConfig(PreTrainedConfig):
    """AMP velocity policy: 585-dim proprioceptive obs, 29-dim joint actions."""

    device: str = "cpu"
    deploy_yaml: str = "deploy.yaml"
    onnx_filename: str = "policy.onnx"
    step_dt: float = 0.02

    horizon: int = 1
    n_action_steps: int = 1
    n_obs_steps: int = 1
    history_length: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    action_feature_names: list[str] = field(
        default_factory=lambda: [joint_q_key(name) for name in G1_JOINT_NAMES]
    )

    def __post_init__(self):
        super().__post_init__()
        self.input_features = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(TOTAL_OBS,)),
        }
        self.output_features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(NUM_JOINTS,)),
        }

    def validate_features(self) -> None:
        if OBS_STATE not in self.input_features:
            raise ValueError("AMP velocity policy requires 'observation.state' input.")
        if ACTION not in self.output_features:
            raise ValueError("AMP velocity policy requires 'action' output.")

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return [0]

    @property
    def reward_delta_indices(self) -> None:
        return None

    def resolved_deploy_yaml(self, model_dir: str | Path) -> Path:
        path = Path(model_dir) / self.deploy_yaml
        if path.is_file():
            return path
        raise FileNotFoundError(f"deploy.yaml not found at {path}")

    def resolved_onnx_path(self, model_dir: str | Path) -> Path:
        path = Path(model_dir) / self.onnx_filename
        if path.is_file():
            return path
        raise FileNotFoundError(f"ONNX policy not found at {path}")
