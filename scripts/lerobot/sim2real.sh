#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_PATH="${POLICY_PATH:-$ROOT/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/pretrained_model}"
DEPLOY_YAML="${DEPLOY_YAML:-$ROOT/deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy_1.yaml}"
NETWORK="${NETWORK:-eth0}"

amp-rollout \
  --strategy.type=base \
  --policy.path="$POLICY_PATH" \
  --robot.type=amp_g1 \
  --robot.is_simulation=false \
  --robot.network="$NETWORK" \
  --robot.domain_id=0 \
  --robot.deploy_yaml="$DEPLOY_YAML" \
  --fps=50 \
  --duration="${DURATION:-0}" \
  --display_data="${DISPLAY_DATA:-false}"
