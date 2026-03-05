"""
Smoothed velocity command generator for training policies that handle
gradual acceleration/deceleration (hold-to-move keyboard control).

Instead of the default "sample and hold constant for 10s" behavior,
this command generator:
  1. Samples a TARGET velocity from uniform distribution (as usual)
  2. Smoothly ramps the ACTUAL command toward the target using exponential smoothing
  3. Randomly "releases" the target to zero mid-episode (simulating key release)
  4. Smoothly decays the command to zero after release

This trains the policy to handle the same input pattern it will see during deployment
with keyboard control (hold W → ramp up, release W → smooth decay).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ── Implementation first (needs to be defined before the Cfg so we can
#    assign class_type = SmoothedVelocityCommand directly in the dataclass
#    field, which is how Isaac Lab's @configclass expects it).
#    Type annotations like `cfg: SmoothedVelocityCommandCfg` are safe because
#    `from __future__ import annotations` makes them lazy strings. ──


class SmoothedVelocityCommand(UniformVelocityCommand):
    """Velocity command with exponential smoothing and random release events.

    During training, this creates a realistic distribution of velocity command
    profiles that match deployment:
      - Smooth ramp-up when a new command is sampled
      - Smooth decay when the command "releases" to zero
      - Mix of smoothed and non-smoothed envs for robustness
    """

    cfg: SmoothedVelocityCommandCfg

    def __init__(self, cfg: SmoothedVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Target velocity (what we're smoothing toward)
        self.vel_target_b = torch.zeros_like(self.vel_command_b)

        # Whether each env currently has a "released" (zero) target
        self.is_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.release_time_left = torch.zeros(self.num_envs, device=self.device)

        # Whether each env uses smoothing
        self.is_smoothed_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Randomly assign smoothed envs
        self.is_smoothed_env[:] = (
            torch.rand(self.num_envs, device=self.device) < self.cfg.smoothed_env_fraction
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new target velocity and reset release state."""
        # Use parent's sampling to set vel_command_b
        super()._resample_command(env_ids)

        # Copy sampled command to target
        self.vel_target_b[env_ids] = self.vel_command_b[env_ids].clone()

        # For smoothed envs, don't jump — keep the current smoothed command
        # and let it ramp toward the new target
        smoothed_mask = self.is_smoothed_env[env_ids]
        smoothed_ids = env_ids[smoothed_mask] if isinstance(env_ids, torch.Tensor) else \
            torch.tensor(env_ids, device=self.device)[smoothed_mask]

        if len(smoothed_ids) > 0:
            # Don't overwrite vel_command_b for smoothed envs — keep current value
            # so it smoothly transitions to the new target
            pass  # vel_command_b will be updated in _update_command via smoothing

        # Reset release state
        self.is_released[env_ids] = False
        self.release_time_left[env_ids] = 0.0

        # Re-randomize which envs are smoothed
        if isinstance(env_ids, torch.Tensor):
            n = len(env_ids)
        else:
            n = len(env_ids)
        self.is_smoothed_env[env_ids] = (
            torch.rand(n, device=self.device) < self.cfg.smoothed_env_fraction
        )

    def _update_command(self):
        """Apply smoothing and handle random releases."""
        # First apply parent's update (heading control, standing envs)
        super()._update_command()

        # For non-smoothed envs, vel_command_b was already set by parent — leave it

        # For smoothed envs, update target and apply smoothing
        smoothed_ids = self.is_smoothed_env.nonzero(as_tuple=False).flatten()

        if len(smoothed_ids) == 0:
            return

        # --- Random release: with some probability, set target to zero ---
        # Only release envs that are NOT already released and NOT standing
        can_release = (
            self.is_smoothed_env
            & ~self.is_released
            & ~self.is_standing_env
        )
        release_roll = torch.rand(self.num_envs, device=self.device)
        newly_released = can_release & (release_roll < self.cfg.release_prob_per_step)
        if newly_released.any():
            released_ids = newly_released.nonzero(as_tuple=False).flatten()
            self.is_released[released_ids] = True
            self.release_time_left[released_ids] = torch.empty(
                len(released_ids), device=self.device
            ).uniform_(*self.cfg.release_duration_range)

        # --- Update release timer and un-release expired ones ---
        self.release_time_left -= self._env.step_dt
        un_release = self.is_released & (self.release_time_left <= 0.0)
        if un_release.any():
            un_release_ids = un_release.nonzero(as_tuple=False).flatten()
            self.is_released[un_release_ids] = False
            # Sample a new target velocity for these envs
            r = torch.empty(len(un_release_ids), device=self.device)
            self.vel_target_b[un_release_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
            self.vel_target_b[un_release_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
            self.vel_target_b[un_release_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # --- Compute effective target (zero for released envs) ---
        effective_target = self.vel_target_b.clone()
        effective_target[self.is_released] = 0.0

        # --- Apply exponential smoothing for smoothed envs ---
        alpha = self.cfg.smoothing
        self.vel_command_b[smoothed_ids] += alpha * (
            effective_target[smoothed_ids] - self.vel_command_b[smoothed_ids]
        )

        # Deadzone: snap near-zero values to exactly zero
        small = self.vel_command_b[smoothed_ids].abs() < 0.01
        self.vel_command_b[smoothed_ids] = torch.where(
            small,
            torch.zeros_like(self.vel_command_b[smoothed_ids]),
            self.vel_command_b[smoothed_ids],
        )

        # Enforce standing envs (override smoothed values if standing)
        standing_smoothed = (self.is_standing_env & self.is_smoothed_env).nonzero(as_tuple=False).flatten()
        if len(standing_smoothed) > 0:
            self.vel_command_b[standing_smoothed] = 0.0


# ── Configuration (defined AFTER the implementation class so we can
#    assign class_type directly in the field default) ──


@configclass
class SmoothedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for smoothed velocity command generator.

    Adds exponential smoothing and random release events on top of
    the standard uniform velocity command.
    """

    class_type: type = SmoothedVelocityCommand

    # Exponential smoothing factor (per policy step).
    # 0.05 = slow ramp (~1s to 95%), 0.15 = fast ramp (~400ms to 95%)
    # Should match the SMOOTHING value used in the deployment controller.
    smoothing: float = 0.15

    # Probability per step that the target will "release" to zero
    # (simulating the user releasing the key). Higher = more frequent stops.
    # At 50Hz with 0.002, average hold time ≈ 10s; with 0.01, ≈ 2s
    release_prob_per_step: float = 0.005

    # Once released, time range (seconds) before a new target is sampled
    release_duration_range: tuple[float, float] = (0.5, 3.0)

    # Fraction of envs that use smoothing (rest use standard hold behavior
    # for diversity). Set to 1.0 to apply smoothing to all envs.
    smoothed_env_fraction: float = 0.5
