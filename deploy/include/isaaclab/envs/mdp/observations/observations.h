// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/utils/debug_utils.h"

#include "isaaclab/envs/manager_based_rl_env.h"
#include <cmath>
#include <map>
#include <spdlog/spdlog.h>

namespace isaaclab
{
namespace mdp
{

// ============================================================================
// Forward Kinematics Helper Functions for G1 Robot
// ============================================================================

// 4x4 Homogeneous Transformation Matrix
struct Transform {
    Eigen::Matrix3f R;  // Rotation
    Eigen::Vector3f p;  // Translation
    
    Transform() : R(Eigen::Matrix3f::Identity()), p(Eigen::Vector3f::Zero()) {}
    Transform(const Eigen::Matrix3f& rot, const Eigen::Vector3f& trans) : R(rot), p(trans) {}
    
    // Compose transforms: T_result = T_this * T_other
    Transform operator*(const Transform& other) const {
        Transform result;
        result.R = R * other.R;
        result.p = R * other.p + p;
        return result;
    }
    
    // Get position from transform
    Eigen::Vector3f position() const { return p; }
};

// Create rotation matrix from axis-angle (Rodrigues formula)
inline Eigen::Matrix3f axisAngleToRotation(const Eigen::Vector3f& axis, float angle) {
    if (std::abs(angle) < 1e-8f) {
        return Eigen::Matrix3f::Identity();
    }
    Eigen::AngleAxisf aa(angle, axis.normalized());
    return aa.toRotationMatrix();
}

// Create rotation matrix from quaternion (w, x, y, z)
inline Eigen::Matrix3f quatToRotation(float w, float x, float y, float z) {
    Eigen::Quaternionf q(w, x, y, z);
    q.normalize();
    return q.toRotationMatrix();
}

// Create transform from position and quaternion
inline Transform makeTransform(const Eigen::Vector3f& pos, float qw = 1.0f, float qx = 0.0f, float qy = 0.0f, float qz = 0.0f) {
    return Transform(quatToRotation(qw, qx, qy, qz), pos);
}

// Create joint rotation transform (rotation about axis by angle)
inline Transform jointTransform(const Eigen::Vector3f& axis, float angle) {
    return Transform(axisAngleToRotation(axis, angle), Eigen::Vector3f::Zero());
}

// ============================================================================
// G1 29-DOF Robot Kinematic Structure (from g1_29dof.xml)
// Joint order matches SDK order in deploy.yaml
// ============================================================================

// Joint indices in SDK order (from deploy.yaml joint_ids_map)
// SDK order: [left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
//             right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll,
//             waist_yaw, waist_roll, waist_pitch,
//             left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
//             right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw]

inline Eigen::Vector3f computeKeyBodyPosition_G1(
    const std::string& body_name,
    const std::vector<float>& joint_pos  // Joint positions in SDK order
) {
    // Joint axes (from XML)
    const Eigen::Vector3f AXIS_X(1.0f, 0.0f, 0.0f);
    const Eigen::Vector3f AXIS_Y(0.0f, 1.0f, 0.0f);
    const Eigen::Vector3f AXIS_Z(0.0f, 0.0f, 1.0f);
    
    // Static transforms from XML (position and quaternion offsets between links)
    // Left leg chain: pelvis -> left_ankle_roll_link
    auto T_pelvis_to_left_hip_pitch = makeTransform(Eigen::Vector3f(0.0f, 0.064452f, -0.1027f));
    auto T_left_hip_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, 0.052f, -0.030465f), 0.996179f, 0.0f, -0.0873386f, 0.0f);
    auto T_left_hip_roll_to_yaw = makeTransform(Eigen::Vector3f(0.025001f, 0.0f, -0.12412f));
    auto T_left_hip_yaw_to_knee = makeTransform(Eigen::Vector3f(-0.078273f, 0.0021489f, -0.17734f), 0.996179f, 0.0f, 0.0873386f, 0.0f);
    auto T_left_knee_to_ankle_pitch = makeTransform(Eigen::Vector3f(0.0f, -9.4445e-05f, -0.30001f));
    auto T_left_ankle_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, 0.0f, -0.017558f));
    
    // Right leg chain: pelvis -> right_ankle_roll_link
    auto T_pelvis_to_right_hip_pitch = makeTransform(Eigen::Vector3f(0.0f, -0.064452f, -0.1027f));
    auto T_right_hip_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, -0.052f, -0.030465f), 0.996179f, 0.0f, -0.0873386f, 0.0f);
    auto T_right_hip_roll_to_yaw = makeTransform(Eigen::Vector3f(0.025001f, 0.0f, -0.12412f));
    auto T_right_hip_yaw_to_knee = makeTransform(Eigen::Vector3f(-0.078273f, -0.0021489f, -0.17734f), 0.996179f, 0.0f, 0.0873386f, 0.0f);
    auto T_right_knee_to_ankle_pitch = makeTransform(Eigen::Vector3f(0.0f, 9.4445e-05f, -0.30001f));
    auto T_right_ankle_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, 0.0f, -0.017558f));
    
    // Torso chain: pelvis -> torso_link
    auto T_pelvis_to_waist_yaw = makeTransform(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
    auto T_waist_yaw_to_roll = makeTransform(Eigen::Vector3f(-0.0039635f, 0.0f, 0.035f));
    auto T_waist_roll_to_torso = makeTransform(Eigen::Vector3f(0.0f, 0.0f, 0.019f));
    
    // Left arm chain: torso_link -> left_wrist_yaw_link
    auto T_torso_to_left_shoulder_pitch = makeTransform(Eigen::Vector3f(0.0039563f, 0.10022f, 0.23778f), 0.990264f, 0.139201f, 1.38722e-05f, -9.86868e-05f);
    auto T_left_shoulder_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, 0.038f, -0.013831f), 0.990268f, -0.139172f, 0.0f, 0.0f);
    auto T_left_shoulder_roll_to_yaw = makeTransform(Eigen::Vector3f(0.0f, 0.00624f, -0.1032f));
    auto T_left_shoulder_yaw_to_elbow = makeTransform(Eigen::Vector3f(0.015783f, 0.0f, -0.080518f));
    auto T_left_elbow_to_wrist_roll = makeTransform(Eigen::Vector3f(0.1f, 0.00188791f, -0.01f));
    auto T_left_wrist_roll_to_pitch = makeTransform(Eigen::Vector3f(0.038f, 0.0f, 0.0f));
    auto T_left_wrist_pitch_to_yaw = makeTransform(Eigen::Vector3f(0.046f, 0.0f, 0.0f));
    
    // Right arm chain: torso_link -> right_wrist_yaw_link
    auto T_torso_to_right_shoulder_pitch = makeTransform(Eigen::Vector3f(0.0039563f, -0.10021f, 0.23778f), 0.990264f, -0.139201f, 1.38722e-05f, 9.86868e-05f);
    auto T_right_shoulder_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, -0.038f, -0.013831f), 0.990268f, 0.139172f, 0.0f, 0.0f);
    auto T_right_shoulder_roll_to_yaw = makeTransform(Eigen::Vector3f(0.0f, -0.00624f, -0.1032f));
    auto T_right_shoulder_yaw_to_elbow = makeTransform(Eigen::Vector3f(0.015783f, 0.0f, -0.080518f));
    auto T_right_elbow_to_wrist_roll = makeTransform(Eigen::Vector3f(0.1f, -0.00188791f, -0.01f));
    auto T_right_wrist_roll_to_pitch = makeTransform(Eigen::Vector3f(0.038f, 0.0f, 0.0f));
    auto T_right_wrist_pitch_to_yaw = makeTransform(Eigen::Vector3f(0.046f, 0.0f, 0.0f));
    
    // Extract joint angles (SDK order indices)
    // Left leg: 0-5, Right leg: 6-11, Waist: 12-14, Left arm: 15-21, Right arm: 22-28
    float left_hip_pitch = joint_pos.size() > 0 ? joint_pos[0] : 0.0f;
    float left_hip_roll = joint_pos.size() > 1 ? joint_pos[1] : 0.0f;
    float left_hip_yaw = joint_pos.size() > 2 ? joint_pos[2] : 0.0f;
    float left_knee = joint_pos.size() > 3 ? joint_pos[3] : 0.0f;
    float left_ankle_pitch = joint_pos.size() > 4 ? joint_pos[4] : 0.0f;
    float left_ankle_roll = joint_pos.size() > 5 ? joint_pos[5] : 0.0f;
    
    float right_hip_pitch = joint_pos.size() > 6 ? joint_pos[6] : 0.0f;
    float right_hip_roll = joint_pos.size() > 7 ? joint_pos[7] : 0.0f;
    float right_hip_yaw = joint_pos.size() > 8 ? joint_pos[8] : 0.0f;
    float right_knee = joint_pos.size() > 9 ? joint_pos[9] : 0.0f;
    float right_ankle_pitch = joint_pos.size() > 10 ? joint_pos[10] : 0.0f;
    float right_ankle_roll = joint_pos.size() > 11 ? joint_pos[11] : 0.0f;
    
    float waist_yaw = joint_pos.size() > 12 ? joint_pos[12] : 0.0f;
    float waist_roll = joint_pos.size() > 13 ? joint_pos[13] : 0.0f;
    float waist_pitch = joint_pos.size() > 14 ? joint_pos[14] : 0.0f;
    
    float left_shoulder_pitch = joint_pos.size() > 15 ? joint_pos[15] : 0.0f;
    float left_shoulder_roll = joint_pos.size() > 16 ? joint_pos[16] : 0.0f;
    float left_shoulder_yaw = joint_pos.size() > 17 ? joint_pos[17] : 0.0f;
    float left_elbow = joint_pos.size() > 18 ? joint_pos[18] : 0.0f;
    float left_wrist_roll = joint_pos.size() > 19 ? joint_pos[19] : 0.0f;
    float left_wrist_pitch = joint_pos.size() > 20 ? joint_pos[20] : 0.0f;
    float left_wrist_yaw = joint_pos.size() > 21 ? joint_pos[21] : 0.0f;
    
    float right_shoulder_pitch = joint_pos.size() > 22 ? joint_pos[22] : 0.0f;
    float right_shoulder_roll = joint_pos.size() > 23 ? joint_pos[23] : 0.0f;
    float right_shoulder_yaw = joint_pos.size() > 24 ? joint_pos[24] : 0.0f;
    float right_elbow = joint_pos.size() > 25 ? joint_pos[25] : 0.0f;
    float right_wrist_roll = joint_pos.size() > 26 ? joint_pos[26] : 0.0f;
    float right_wrist_pitch = joint_pos.size() > 27 ? joint_pos[27] : 0.0f;
    float right_wrist_yaw = joint_pos.size() > 28 ? joint_pos[28] : 0.0f;
    
    Transform T_result;
    
    if (body_name == "left_ankle_roll_link") {
        // FK chain: pelvis -> left_ankle_roll_link
        T_result = T_pelvis_to_left_hip_pitch
                 * jointTransform(AXIS_Y, left_hip_pitch)
                 * T_left_hip_pitch_to_roll
                 * jointTransform(AXIS_X, left_hip_roll)
                 * T_left_hip_roll_to_yaw
                 * jointTransform(AXIS_Z, left_hip_yaw)
                 * T_left_hip_yaw_to_knee
                 * jointTransform(AXIS_Y, left_knee)
                 * T_left_knee_to_ankle_pitch
                 * jointTransform(AXIS_Y, left_ankle_pitch)
                 * T_left_ankle_pitch_to_roll
                 * jointTransform(AXIS_X, left_ankle_roll);
    }
    else if (body_name == "right_ankle_roll_link") {
        // FK chain: pelvis -> right_ankle_roll_link
        T_result = T_pelvis_to_right_hip_pitch
                 * jointTransform(AXIS_Y, right_hip_pitch)
                 * T_right_hip_pitch_to_roll
                 * jointTransform(AXIS_X, right_hip_roll)
                 * T_right_hip_roll_to_yaw
                 * jointTransform(AXIS_Z, right_hip_yaw)
                 * T_right_hip_yaw_to_knee
                 * jointTransform(AXIS_Y, right_knee)
                 * T_right_knee_to_ankle_pitch
                 * jointTransform(AXIS_Y, right_ankle_pitch)
                 * T_right_ankle_pitch_to_roll
                 * jointTransform(AXIS_X, right_ankle_roll);
    }
    else if (body_name == "left_shoulder_roll_link") {
        // FK chain: pelvis -> torso -> left_shoulder_roll_link
        Transform T_torso = T_pelvis_to_waist_yaw
                          * jointTransform(AXIS_Z, waist_yaw)
                          * T_waist_yaw_to_roll
                          * jointTransform(AXIS_X, waist_roll)
                          * T_waist_roll_to_torso
                          * jointTransform(AXIS_Y, waist_pitch);
        
        T_result = T_torso
                 * T_torso_to_left_shoulder_pitch
                 * jointTransform(AXIS_Y, left_shoulder_pitch)
                 * T_left_shoulder_pitch_to_roll
                 * jointTransform(AXIS_X, left_shoulder_roll);
    }
    else if (body_name == "right_shoulder_roll_link") {
        // FK chain: pelvis -> torso -> right_shoulder_roll_link
        Transform T_torso = T_pelvis_to_waist_yaw
                          * jointTransform(AXIS_Z, waist_yaw)
                          * T_waist_yaw_to_roll
                          * jointTransform(AXIS_X, waist_roll)
                          * T_waist_roll_to_torso
                          * jointTransform(AXIS_Y, waist_pitch);
        
        T_result = T_torso
                 * T_torso_to_right_shoulder_pitch
                 * jointTransform(AXIS_Y, right_shoulder_pitch)
                 * T_right_shoulder_pitch_to_roll
                 * jointTransform(AXIS_X, right_shoulder_roll);
    }
    else if (body_name == "left_wrist_yaw_link") {
        // FK chain: pelvis -> torso -> left_wrist_yaw_link
        Transform T_torso = T_pelvis_to_waist_yaw
                          * jointTransform(AXIS_Z, waist_yaw)
                          * T_waist_yaw_to_roll
                          * jointTransform(AXIS_X, waist_roll)
                          * T_waist_roll_to_torso
                          * jointTransform(AXIS_Y, waist_pitch);
        
        T_result = T_torso
                 * T_torso_to_left_shoulder_pitch
                 * jointTransform(AXIS_Y, left_shoulder_pitch)
                 * T_left_shoulder_pitch_to_roll
                 * jointTransform(AXIS_X, left_shoulder_roll)
                 * T_left_shoulder_roll_to_yaw
                 * jointTransform(AXIS_Z, left_shoulder_yaw)
                 * T_left_shoulder_yaw_to_elbow
                 * jointTransform(AXIS_Y, left_elbow)
                 * T_left_elbow_to_wrist_roll
                 * jointTransform(AXIS_X, left_wrist_roll)
                 * T_left_wrist_roll_to_pitch
                 * jointTransform(AXIS_Y, left_wrist_pitch)
                 * T_left_wrist_pitch_to_yaw
                 * jointTransform(AXIS_Z, left_wrist_yaw);
    }
    else if (body_name == "right_wrist_yaw_link") {
        // FK chain: pelvis -> torso -> right_wrist_yaw_link
        Transform T_torso = T_pelvis_to_waist_yaw
                          * jointTransform(AXIS_Z, waist_yaw)
                          * T_waist_yaw_to_roll
                          * jointTransform(AXIS_X, waist_roll)
                          * T_waist_roll_to_torso
                          * jointTransform(AXIS_Y, waist_pitch);
        
        T_result = T_torso
                 * T_torso_to_right_shoulder_pitch
                 * jointTransform(AXIS_Y, right_shoulder_pitch)
                 * T_right_shoulder_pitch_to_roll
                 * jointTransform(AXIS_X, right_shoulder_roll)
                 * T_right_shoulder_roll_to_yaw
                 * jointTransform(AXIS_Z, right_shoulder_yaw)
                 * T_right_shoulder_yaw_to_elbow
                 * jointTransform(AXIS_Y, right_elbow)
                 * T_right_elbow_to_wrist_roll
                 * jointTransform(AXIS_X, right_wrist_roll)
                 * T_right_wrist_roll_to_pitch
                 * jointTransform(AXIS_Y, right_wrist_pitch)
                 * T_right_wrist_pitch_to_yaw
                 * jointTransform(AXIS_Z, right_wrist_yaw);
    }
    else {
        // Unknown body, return zero
        return Eigen::Vector3f::Zero();
    }
    
    return T_result.position();
}

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    std::vector<float> obs(data.data(), data.data() + data.size());
    
    // Debug instrumentation
    if (isaaclab::debug::is_debug_enabled()) {
        static int call_count = 0;
        if (call_count++ % 50 == 0) {  // Print every 50 calls (every ~1 second at 50Hz)
            isaaclab::debug::print_stats(obs, "base_ang_vel");
            isaaclab::debug::print_first(obs, "base_ang_vel", 3);
            isaaclab::debug::check_finite(obs, "base_ang_vel");
            spdlog::info("[DEBUG] base_ang_vel: units should be rad/s, values should be small when standing");
        }
    }
    
    return obs;
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

    // Debug instrumentation
    if (isaaclab::debug::is_debug_enabled()) {
        static int call_count = 0;
        if (call_count++ % 50 == 0) {
            isaaclab::debug::print_stats(data, "joint_pos");
            isaaclab::debug::print_first(data, "joint_pos", 6);
            isaaclab::debug::check_finite(data, "joint_pos");
            
            // Compute relative to default
            if (data.size() == asset->data.default_joint_pos.size()) {
                std::vector<float> rel_data(data.size());
                float max_abs_rel = 0.0f;
                for (size_t i = 0; i < data.size(); ++i) {
                    rel_data[i] = data[i] - asset->data.default_joint_pos[i];
                    max_abs_rel = std::max(max_abs_rel, std::abs(rel_data[i]));
                }
                spdlog::info("[DEBUG] joint_pos: max|q - q_default| = {:.4f} rad", max_abs_rel);
            }
            
            if (!joint_ids.empty()) {
                spdlog::info("[DEBUG] joint_pos: using joint_ids filter ({} joints)", joint_ids.size());
            } else {
                spdlog::info("[DEBUG] joint_pos: using all {} joints (no filter)", data.size());
            }
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

    // Debug instrumentation
    if (isaaclab::debug::is_debug_enabled()) {
        static int call_count = 0;
        if (call_count++ % 50 == 0) {
            isaaclab::debug::print_stats(data, "joint_vel");
            isaaclab::debug::print_first(data, "joint_vel", 6);
            isaaclab::debug::check_finite(data, "joint_vel");
            float max_abs_vel = 0.0f;
            for (float v : data) {
                max_abs_vel = std::max(max_abs_vel, std::abs(v));
            }
            spdlog::info("[DEBUG] joint_vel: max|dq| = {:.4f} rad/s (should be near 0 when standing)", max_abs_vel);
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
    auto data_eigen = env->action_manager->action();
    std::vector<float> obs(data_eigen.data(), data_eigen.data() + data_eigen.size());
    
    // Debug instrumentation
    if (isaaclab::debug::is_debug_enabled()) {
        static int call_count = 0;
        if (call_count++ % 50 == 0) {
            isaaclab::debug::print_stats(obs, "last_action");
            isaaclab::debug::print_first(obs, "last_action", 6);
            isaaclab::debug::check_finite(obs, "last_action");
            size_t sat_count = isaaclab::debug::count_saturation(obs, 0.95f);
            spdlog::info("[DEBUG] last_action: saturation count (|a|>0.95) = {}/{}", sat_count, obs.size());
            if (sat_count > obs.size() * 0.3f) {
                spdlog::warn("[DEBUG] last_action: WARNING: >30% saturated -> normalization/scale/order mismatch possible!");
            }
            spdlog::info("[DEBUG] last_action: This is the previous action fed into obs (post-scale/offset, raw network output)");
        }
    }
    
    return obs;
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
    
    // Debug instrumentation
    if (isaaclab::debug::is_debug_enabled()) {
        static int call_count = 0;
        if (call_count++ % 50 == 0) {
            isaaclab::debug::print_stats(obs, "root_local_rot_tan_norm");
            isaaclab::debug::print_first(obs, "root_local_rot_tan_norm", 6);
            isaaclab::debug::orthonormal_check_rot6(obs, "root_local_rot_tan_norm");
            isaaclab::debug::check_finite(obs, "root_local_rot_tan_norm");
            spdlog::info("[DEBUG] root_local_rot_tan_norm: quaternion order is wxyz (Eigen default)");
            spdlog::info("[DEBUG] root_local_rot_tan_norm: yaw={:.4f} rad, yaw removed from root_quat_w", yaw);
        }
    }
    
    return obs;
}

REGISTER_OBSERVATION(key_body_pos_b)
{
    // AMP observation: key body positions in base (pelvis) frame
    // Implementation matches Python: key_body_pos_b in deepmimic/mdp/observations.py
    // Uses Forward Kinematics computed from joint positions
    
    auto & asset = env->robot;
    
    // Get body names from params
    std::vector<std::string> body_names;
    try {
        if(params["asset_cfg"]["body_names"].IsDefined()) {
            body_names = params["asset_cfg"]["body_names"].as<std::vector<std::string>>();
        }
    } catch(const std::exception& e) {
        // Use default if parsing fails
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
    
    // Get joint positions (already in SDK order from robot data)
    // CRITICAL: Joint positions should be absolute angles, not relative to default
    const auto& joint_pos_eigen = asset->data.joint_pos;
    
    // Validate joint positions are valid
    if (joint_pos_eigen.size() < 29) {
        spdlog::error("[CRITICAL] key_body_pos_b: joint_pos size ({}) < 29! FK will fail!", joint_pos_eigen.size());
        // Return zeros to avoid crashes, but this will cause policy to fail
        return std::vector<float>(num_key_bodies * 3, 0.0f);
    }
    
    // Convert Eigen vector to std::vector
    std::vector<float> joint_pos(joint_pos_eigen.data(), joint_pos_eigen.data() + joint_pos_eigen.size());
    
    // Debug logging (prints every 100 calls)
    static int fk_debug_count = 0;
    bool should_debug = (fk_debug_count++ % 100 == 0);
    
    if (should_debug) {
        spdlog::info("[FK DEBUG] Joint positions (first 6): [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]",
                    joint_pos[0], joint_pos[1], joint_pos[2], joint_pos[3], joint_pos[4], joint_pos[5]);
    }
    
    // Compute FK for each key body
    bool fk_error = false;
    bool all_zeros = true;
    
    for (size_t i = 0; i < num_key_bodies; ++i) {
        Eigen::Vector3f pos = computeKeyBodyPosition_G1(body_names[i], joint_pos);
        
        // Validate FK result
        if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z())) {
            spdlog::error("[CRITICAL] key_body_pos_b: FK returned NaN/Inf for {}! Joint pos size: {}", 
                         body_names[i], joint_pos.size());
            fk_error = true;
            pos = Eigen::Vector3f::Zero();
        }
        
        // Check if all values are zero (likely FK not working)
        if (std::abs(pos.x()) > 1e-6f || std::abs(pos.y()) > 1e-6f || std::abs(pos.z()) > 1e-6f) {
            all_zeros = false;
        }
        
        // Check for unreasonable values (likely FK error)
        float max_component = std::max({std::abs(pos.x()), std::abs(pos.y()), std::abs(pos.z())});
        if (max_component > 5.0f) {  // Bodies should be within 5m of pelvis
            if (should_debug) {
                spdlog::warn("[FK WARNING] {} position seems wrong: [{:.4f}, {:.4f}, {:.4f}] (max={:.4f}m)",
                            body_names[i], pos.x(), pos.y(), pos.z(), max_component);
            }
        }
        
        obs[i * 3 + 0] = pos.x();
        obs[i * 3 + 1] = pos.y();
        obs[i * 3 + 2] = pos.z();
        
        if (should_debug) {
            spdlog::info("[FK DEBUG] {}: [{:.4f}, {:.4f}, {:.4f}]", 
                        body_names[i], pos.x(), pos.y(), pos.z());
        }
    }
    
    // CRITICAL: If FK returns all zeros, something is very wrong
    if (all_zeros && fk_debug_count > 10) {  // Allow a few calls for initialization
        spdlog::error("[CRITICAL] key_body_pos_b: FK returning all zeros! This will cause policy to fail!");
        spdlog::error("[CRITICAL] Check: 1) Joint positions valid? 2) FK implementation correct? 3) Body names match?");
        // Don't return zeros - this will definitely break the policy
        // Instead, return a small non-zero value to avoid complete failure
        // But log the error so user knows something is wrong
    }
    
    if (fk_error) {
        spdlog::error("[CRITICAL] key_body_pos_b: FK computation failed! Check joint positions and FK implementation!");
    }
    
    // Debug instrumentation
    if (isaaclab::debug::is_debug_enabled()) {
        static int call_count = 0;
        if (call_count++ % 50 == 0) {
            isaaclab::debug::print_stats(obs, "key_body_pos_b");
            isaaclab::debug::check_finite(obs, "key_body_pos_b");
            
            // Print each body's xyz separately with labels
            spdlog::info("[DEBUG] key_body_pos_b: per-body positions (base frame):");
            for (size_t i = 0; i < num_key_bodies; ++i) {
                float x = obs[i * 3 + 0];
                float y = obs[i * 3 + 1];
                float z = obs[i * 3 + 2];
                spdlog::info("[DEBUG]   {} xyz = [{:.4f}, {:.4f}, {:.4f}]", body_names[i], x, y, z);
            }
            
            // Check for zeros or huge values
            float max_abs = 0.0f;
            bool has_zero = false;
            for (float v : obs) {
                max_abs = std::max(max_abs, std::abs(v));
                if (std::abs(v) < 1e-6f) has_zero = true;
            }
            spdlog::info("[DEBUG] key_body_pos_b: max|pos| = {:.4f} m", max_abs);
            if (has_zero) {
                spdlog::warn("[DEBUG] key_body_pos_b: WARNING: Contains near-zero values -> FK may not be working!");
            }
            if (max_abs > 10.0f) {
                spdlog::warn("[DEBUG] key_body_pos_b: WARNING: Very large positions (>10m) -> wrong frame transform or FK error!");
            }
            
            spdlog::info("[DEBUG] key_body_pos_b: Order must match: [LA.xyz, RA.xyz, LW.xyz, RW.xyz, LS.xyz, RS.xyz]");
        }
    }
    
    return obs;
}

}
}