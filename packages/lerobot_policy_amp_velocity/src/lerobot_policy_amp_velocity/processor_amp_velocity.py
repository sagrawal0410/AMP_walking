"""Pre/post processors for AMP velocity policy."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_amp_velocity import AmpVelocityConfig
from .constants import G1_JOINT_NAMES, NUM_JOINTS, joint_dq_key, joint_q_key
from .deploy_config import DeployConfig
from .obs_builder import AmpObsBuilder


@ProcessorStepRegistry.register("amp_obs_builder")
@dataclass
class AmpObsBuilderProcessorStep(ObservationProcessorStep):
    """Build 585-dim observation.state from raw robot observations."""

    deploy_cfg: DeployConfig
    obs_builder: AmpObsBuilder = field(init=False)

    def __post_init__(self):
        self.obs_builder = AmpObsBuilder(self.deploy_cfg)

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

    deploy_cfg: DeployConfig
    obs_builder: AmpObsBuilder

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
    deploy_cfg = DeployConfig(config.resolved_deploy_yaml(model_dir))
    obs_builder = AmpObsBuilder(deploy_cfg)

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            AmpObsBuilderProcessorStep(deploy_cfg=deploy_cfg),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=config.device),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[
            AmpActionPostprocessProcessorStep(deploy_cfg=deploy_cfg, obs_builder=obs_builder),
            DeviceProcessorStep(device="cpu"),
        ],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
