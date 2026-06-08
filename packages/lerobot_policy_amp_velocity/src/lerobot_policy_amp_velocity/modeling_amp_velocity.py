"""ONNX-backed AMP velocity policy for LeRobot inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_amp_velocity import AmpVelocityPolicyConfig
from .constants import NUM_JOINTS
from .deploy_config import DeployConfig
from .obs_builder import AmpObsBuilder

logger = logging.getLogger(__name__)


class AmpVelocityPolicy(PreTrainedPolicy):
    config_class = AmpVelocityPolicyConfig
    name = "amp_velocity"

    def __init__(self, config: AmpVelocityPolicyConfig, deploy_cfg: DeployConfig | None = None):
        super().__init__(config)
        self.config: AmpVelocityPolicyConfig = config
        config.validate_features()

        self.deploy_cfg = deploy_cfg
        self.obs_builder = AmpObsBuilder(deploy_cfg) if deploy_cfg is not None else None
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None

        if deploy_cfg is not None and hasattr(config, "_onnx_path"):
            self._load_onnx(config._onnx_path)

    def _load_onnx(self, onnx_path: str | Path) -> None:
        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        self._input_name = inputs[0].name
        logger.info(
            "Loaded AMP ONNX policy from %s: input=%s output=%s",
            onnx_path,
            inputs[0].shape,
            outputs[0].shape,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ) -> "AmpVelocityPolicy":
        model_dir = Path(pretrained_name_or_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"AMP policy directory not found: {model_dir}")

        if config is None:
            config = AmpVelocityPolicyConfig.from_pretrained(model_dir)

        if not isinstance(config, AmpVelocityPolicyConfig):
            raise TypeError(f"Expected AmpVelocityPolicyConfig, got {type(config)}")

        deploy_yaml = config.resolved_deploy_yaml(model_dir)
        onnx_path = config.resolved_onnx_path(model_dir)
        deploy_cfg = DeployConfig(deploy_yaml)

        config._onnx_path = str(onnx_path)  # type: ignore[attr-defined]
        instance = cls(config, deploy_cfg=deploy_cfg)
        instance.to(config.device)
        instance.eval()
        return instance

    def reset(self) -> None:
        if self.obs_builder is not None:
            self.obs_builder.reset()

    def get_optim_params(self) -> dict:
        return {"params": self.parameters()}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("AMP velocity policy is inference-only (trained in Isaac Lab).")

    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs
    ) -> Tensor:
        action = self.select_action(batch, **kwargs)
        return action.unsqueeze(1)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if self._session is None or self._input_name is None:
            raise RuntimeError("ONNX session not loaded.")

        obs = batch[OBS_STATE]
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        obs_np = obs.detach().cpu().numpy().astype(np.float32)

        result = self._session.run(None, {self._input_name: obs_np})
        raw_action = result[0][0, :NUM_JOINTS].astype(np.float32)

        if self.obs_builder is not None:
            self.obs_builder.set_last_action(raw_action)

        return torch.from_numpy(raw_action).to(obs.device)

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config._save_pretrained(save_directory)
