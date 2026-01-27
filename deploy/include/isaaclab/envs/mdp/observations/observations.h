// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_vel.size());
        for(size_t i = 0; i < asset->data.joint_vel.size(); ++i)
        {
            data[i] = asset->data.joint_vel[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_vel[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);
    return obs;
}

REGISTER_OBSERVATION(root_local_rot_tan_norm)
{
    // AMP observation: root rotation in local frame (yaw-removed) as tan/norm representation
    // Implementation matches Python: root_local_rot_tan_norm in amp/mdp/observations.py
    auto & asset = env->robot;
    auto & root_quat_w = asset->data.root_quat_w;
    
    // Extract yaw quaternion (heading only)
    float yaw = std::atan2(2.0f * (root_quat_w.w() * root_quat_w.z() + root_quat_w.x() * root_quat_w.y()),
                           1.0f - 2.0f * (root_quat_w.y() * root_quat_w.y() + root_quat_w.z() * root_quat_w.z()));
    float half_yaw = yaw * 0.5f;
    Eigen::Quaternionf yaw_quat(std::cos(half_yaw), 0.0f, 0.0f, std::sin(half_yaw));
    yaw_quat.normalize();
    
    // Remove yaw: root_quat_local = yaw_quat^{-1} * root_quat_w
    Eigen::Quaternionf root_quat_local = yaw_quat.conjugate() * root_quat_w;
    
    // Convert to rotation matrix
    Eigen::Matrix3f rotm_local = root_quat_local.toRotationMatrix();
    
    // Extract first column (tan) and third column (norm)
    // Python uses columns 0 and 2: tan_vec = root_rotm_local[:, 0], norm_vec = root_rotm_local[:, 2]
    Eigen::Vector3f tan_vec = rotm_local.col(0);  // First column
    Eigen::Vector3f norm_vec = rotm_local.col(2);  // Third column
    
    // Concatenate: [tan.x, tan.y, tan.z, norm.x, norm.y, norm.z]
    std::vector<float> obs(6);
    obs[0] = tan_vec.x();
    obs[1] = tan_vec.y();
    obs[2] = tan_vec.z();
    obs[3] = norm_vec.x();
    obs[4] = norm_vec.y();
    obs[5] = norm_vec.z();
    
    return obs;
}

REGISTER_OBSERVATION(key_body_pos_b)
{
    // AMP observation: key body positions in base frame
    // Implementation matches Python: key_body_pos_b in deepmimic/mdp/observations.py
    // NOTE: For real robot, this requires Forward Kinematics (FK) to compute body positions
    // For now, this is a placeholder that returns zeros - must be implemented with FK for real deployment
    
    auto & asset = env->robot;
    
    // Get body names from params - handle null values
    std::vector<std::string> body_names;
    try {
        auto body_names_node = params["asset_cfg"]["body_names"];
        if(body_names_node.IsDefined() && !body_names_node.IsNull() && body_names_node.IsSequence()) {
            for(const auto& node : body_names_node) {
                if(!node.IsNull() && node.IsScalar()) {
                    body_names.push_back(node.as<std::string>());
                }
            }
        }
    } catch(const std::exception& e) {
        spdlog::warn("key_body_pos_b: Failed to parse body_names from params: {}", e.what());
    }
    
    // Default key body names for G1 (must match g1_amp_env_cfg.py)
    if(body_names.empty()) {
        body_names = {
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_shoulder_roll_link",
            "right_shoulder_roll_link",
        };
    }
    
    const size_t num_key_bodies = body_names.size();
    std::vector<float> obs(num_key_bodies * 3);  // 3D positions
    
    // TODO: Implement Forward Kinematics to compute body positions
    // For now, return zeros as placeholder
    // In real deployment, this must be replaced with FK computation:
    //  1. Get joint angles from asset->data.joint_pos
    //  2. Compute FK transforms for each key body
    //  3. Transform to base frame: p_body_base = R_base_world^T * (p_body_world - p_base_world)
    //  4. For real robot, p_base_world = 0, R_base_world = I, so p_body_base = FK(base->key_link).position
    
    static bool warned = false;
    if(!warned) {
        spdlog::warn("key_body_pos_b: Returning zeros - FK not implemented. "
                     "This observation will be incorrect until FK is implemented.");
        warned = true;
    }
    
    // Fill with zeros for now
    std::fill(obs.begin(), obs.end(), 0.0f);
    
    return obs;
}

}
}