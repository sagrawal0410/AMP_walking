"""Pre/post processors for AMP velocity policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_amp_velocity import AmpVelocityConfig
from .constants import G1_JOINT_NAMES, NUM_JOINTS, joint_dq_key, joint_q_key
from .deploy_config import DeployConfig
from .obs_builder import AmpObsBuilder

# Pre- and post-processor steps must share a single AmpObsBuilder so that the
# action written by the postprocessor (set_last_action) feeds the next observation
# and the history buffers stay continuous. They are loaded independently from JSON
# at deploy time, so we key a shared instance on the resolved deploy.yaml path.
_OBS_BUILDER_CACHE: dict[str, AmpObsBuilder] = {}


def get_shared_obs_builder(deploy_yaml: str) -> AmpObsBuilder:
    key = str(Path(deploy_yaml).resolve())
    builder = _OBS_BUILDER_CACHE.get(key)
    if builder is None:
        builder = AmpObsBuilder(DeployConfig(key))
        _OBS_BUILDER_CACHE[key] = builder
    return builder


@ProcessorStepRegistry.register("amp_obs_builder")
@dataclass
class AmpObsBuilderProcessorStep(ObservationProcessorStep):
    """Build 585-dim observation.state from raw robot observations."""

    deploy_yaml: str = ""
    obs_builder: AmpObsBuilder = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.deploy_yaml:
            raise ValueError("amp_obs_builder step requires a 'deploy_yaml' path.")
        self.obs_builder = get_shared_obs_builder(self.deploy_yaml)

    def get_config(self) -> dict[str, Any]:
        return {"deploy_yaml": self.deploy_yaml}

    def observation(self, observation: dict) -> dict:
        motor_pos = np.zeros(NUM_JOINTS, dtype=np.float32)
        motor_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        for idx, name in enumerate(G1_JOINT_NAMES):
            motor_pos[idx] = float(observation.get(joint_q_key(name), 0.0))
            motor_vel[idx] = float(observation.get(joint_dq_key(name), 0.0))

        imu_quat = np.array(
            [
                float(observation.get("imu.quat.w", 1.0)),
                float(observation.get("imu.quat.x", 0.0)),
                float(observation.get("imu.quat.y", 0.0)),
                float(observation.get("imu.quat.z", 0.0)),
            ],
            dtype=np.float64,
        )
        imu_gyro = np.array(
            [
                float(observation.get("imu.gyro.x", 0.0)),
                float(observation.get("imu.gyro.y", 0.0)),
                float(observation.get("imu.gyro.z", 0.0)),
            ],
            dtype=np.float32,
        )
        command_vel = np.array(
            [
                float(observation.get("velocity_commands.0", 0.0)),
                float(observation.get("velocity_commands.1", 0.0)),
                float(observation.get("velocity_commands.2", 0.0)),
            ],
            dtype=np.float32,
        )

        obs_vec = self.obs_builder.compute(motor_pos, motor_vel, imu_quat, imu_gyro, command_vel)
        out = dict(observation)
        out[OBS_STATE] = torch.from_numpy(obs_vec)
        return out

    def reset(self) -> None:
        self.obs_builder.reset()

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("amp_action_postprocess")
@dataclass
class AmpActionPostprocessProcessorStep(PolicyActionProcessorStep):
    """Map raw policy actions to SDK-order joint position targets."""

    deploy_yaml: str = ""
    obs_builder: AmpObsBuilder = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.deploy_yaml:
            raise ValueError("amp_action_postprocess step requires a 'deploy_yaml' path.")
        self.obs_builder = get_shared_obs_builder(self.deploy_yaml)

    def get_config(self) -> dict[str, Any]:
        return {"deploy_yaml": self.deploy_yaml}

    def action(self, action: PolicyAction) -> PolicyAction:
        if isinstance(action, torch.Tensor):
            raw = action.detach().cpu().numpy().reshape(-1)[:NUM_JOINTS]
            self.obs_builder.set_last_action(raw)
            targets_sdk = self.obs_builder.policy_to_sdk_targets(raw)
            if action.dim() == 2:
                return torch.from_numpy(targets_sdk).to(action.device).unsqueeze(0)
            return torch.from_numpy(targets_sdk).to(action.device)

        raw = np.asarray(action[ACTION], dtype=np.float32).reshape(-1)[:NUM_JOINTS]
        self.obs_builder.set_last_action(raw)
        targets_sdk = self.obs_builder.policy_to_sdk_targets(raw)
        return {joint_q_key(name): float(targets_sdk[idx]) for idx, name in enumerate(G1_JOINT_NAMES)}

    def reset(self) -> None:
        pass

    def transform_features(self, features):
        return features


def make_amp_velocity_pre_post_processors(
    config: AmpVelocityConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    model_dir = config.pretrained_path
    if model_dir is None:
        raise ValueError("AMP velocity policy requires pretrained_path in config.")
    # Store an absolute path so the steps resolve correctly when reloaded from JSON
    # at deploy time (independent of the working directory).
    deploy_yaml = str(config.resolved_deploy_yaml(model_dir).resolve())

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            # lerobot's rollout always injects a `rename_observations_processor`
            # override; the step must exist or loading raises KeyError.
            RenameObservationsProcessorStep(),
            AmpObsBuilderProcessorStep(deploy_yaml=deploy_yaml),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=config.device),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[
            AmpActionPostprocessProcessorStep(deploy_yaml=deploy_yaml),
            DeviceProcessorStep(device="cpu"),
        ],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
