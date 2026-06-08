"""Configuration for AMP G1 robot deploy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots.config import RobotConfig


def _default_deploy_yaml() -> str:
    root = Path(__file__).resolve().parents[4]
    candidate = (
        root
        / "deploy"
        / "robots"
        / "g1_29dof"
        / "config"
        / "policy"
        / "velocity"
        / "v0"
        / "params"
        / "deploy_1.yaml"
    )
    return str(candidate) if candidate.is_file() else ""


@RobotConfig.register_subclass("amp_g1")
@dataclass
class AmpG1Config(RobotConfig):
    """AMP velocity deploy robot for Unitree G1 29-DOF."""

    is_simulation: bool = True
    network: str = "lo"
    domain_id: int = 0
    deploy_yaml: str = field(default_factory=_default_deploy_yaml)

    policy_step_dt: float = 0.02
    cmd_publish_hz: float = 500.0
    fixstand_duration_s: float = 3.0

    # Optional robot IP for ZMQ socket bridge on real hardware
    robot_ip: str = "192.168.123.164"
