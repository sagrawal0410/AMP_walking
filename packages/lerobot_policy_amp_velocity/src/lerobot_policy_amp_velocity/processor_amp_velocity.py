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
from .constants import RAW_OBS_DIM, G1_JOINT_NAMES, NUM_JOINTS, joint_q_key
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
        # lerobot concatenates the robot's raw .pos features into a single
        # `observation.state` vector (see constants.raw_obs_keys for the layout)
        # and runs this step on it inside the inference engine. We slice the
        # vector back into the components AmpObsBuilder expects, then replace
        # observation.state with the full 585-dim AMP observation.
        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError(
                "amp_obs_builder expected 'observation.state' in the frame. The robot must expose "
                "the raw .pos features from constants.raw_obs_keys()."
            )
        if isinstance(state, torch.Tensor):
            arr = state.detach().cpu().numpy()
        else:
            arr = np.asarray(state)
        arr = arr.reshape(-1).astype(np.float32)
        if arr.shape[0] != RAW_OBS_DIM:
            raise ValueError(
                f"observation.state has dim {arr.shape[0]}, expected raw dim {RAW_OBS_DIM}. "
                "Check that the robot's observation_features match constants.raw_obs_keys()."
            )

        motor_pos = arr[0:NUM_JOINTS]
        motor_vel = arr[NUM_JOINTS : 2 * NUM_JOINTS]
        base = 2 * NUM_JOINTS
        imu_quat = arr[base : base + 4].astype(np.float64)
        imu_gyro = arr[base + 4 : base + 7]
        command_vel = arr[base + 7 : base + 10]

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
