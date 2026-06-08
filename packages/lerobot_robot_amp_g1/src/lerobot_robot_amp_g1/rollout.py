"""AMP-aware rollout entry point with FSM-gated policy inference.

NOTE: do NOT add `from __future__ import annotations` here. lerobot's
`parser.wrap()` reads the raw `cfg` annotation via inspect.getfullargspec; with
PEP 563 stringized annotations it would pass the str "RolloutConfig" to
draccus, which then fails with "must be called with a dataclass type".
"""

import logging
import time

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import RobotConfig  # noqa: F401
from lerobot.rollout import RolloutConfig, build_rollout_context
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.core import send_next_action
from lerobot.teleoperators import TeleoperatorConfig  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

from .amp_g1 import AmpG1  # noqa: F401  (ensures robot + config registration on import)

logger = logging.getLogger(__name__)


class AmpLocomotionStrategy(BaseStrategy):
    """Base rollout strategy that only runs policy inference in VELOCITY FSM state."""

    def run(self, ctx) -> None:
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator
        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("AMP locomotion control loop started (policy gated on VELOCITY FSM)")

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            inner = getattr(robot, "inner", robot)
            policy_enabled = getattr(inner, "policy_enabled", False)

            if policy_enabled:
                if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                    continue
                send_next_action(obs_processed, obs, ctx, interpolator)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)

    def teardown(self, ctx) -> None:
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("AMP locomotion strategy teardown complete")


@parser.wrap()
def amp_rollout(cfg: RolloutConfig):
    init_logging()

    if cfg.display_data:
        init_logging()
        init_rerun(session_name="amp_rollout", ip=cfg.display_ip, port=cfg.display_port)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    ctx = build_rollout_context(cfg, shutdown_event)
    # RolloutStrategy.__init__ requires the strategy config (cfg.strategy, set via
    # --strategy.type=base). We instantiate our subclass directly rather than via
    # create_strategy() since it isn't in lerobot's built-in registry.
    strategy = AmpLocomotionStrategy(cfg.strategy)
    logger.info("AMP rollout | robot=%s | fps=%.0f", cfg.robot.type, cfg.fps)

    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)

    logger.info("AMP rollout finished")


def main():
    register_third_party_plugins()
    amp_rollout()
