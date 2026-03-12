#!/usr/bin/env python3

import sys
import os

# Import everything from sim2sim_amp (they share 100% of the DDS logic)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2sim_amp import AmpController, DeployConfig, main as sim2sim_main


def main():
    # Override sys.argv defaults if not explicitly provided
    import argparse
    parser = argparse.ArgumentParser(
        description="Sim2Real AMP Policy Controller for Unitree G1 29-DOF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

Usage:
  python sim2real_amp.py --network eth0
  python sim2real_amp.py --network eth0 --policy /path/to/policy.onnx --deploy-yaml /path/to/deploy.yaml
""",
    )
    parser.add_argument("--network", "-n", type=str, default="eth0",
                        help="DDS network interface (default: eth0)")
    parser.add_argument("--domain-id", type=int, default=0,
                        help="DDS domain ID (default: 0)")
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to policy.onnx")
    parser.add_argument("--deploy-yaml", type=str, default=None,
                        help="Path to deploy.yaml")
    args = parser.parse_args()

    from sim2sim_amp import find_default_paths
    default_policy, default_yaml = find_default_paths()
    policy_path = args.policy or default_policy
    yaml_path = args.deploy_yaml or default_yaml

    for name, path in [("Policy", policy_path), ("deploy.yaml", yaml_path)]:
        if not os.path.isfile(path):
            print(f"[ERROR] {name} not found: {path}")
            sys.exit(1)
        print(f"[INFO] {name}: {path}")

    cfg = DeployConfig(yaml_path)
    controller = AmpController(policy_path=policy_path, deploy_cfg=cfg)
    controller.run(network=args.network, domain_id=args.domain_id)


if __name__ == "__main__":
    main()
