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
// ============================================================================
//
// IMPORTANT: joint_pos is in POLICY order (after joint_ids_map remapping), NOT SDK order!
//
// joint_ids_map: [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
//
// SDK order (physical robot):
//   Left leg:  0=hip_pitch, 1=hip_roll, 2=hip_yaw, 3=knee, 4=ankle_pitch, 5=ankle_roll
//   Right leg: 6=hip_pitch, 7=hip_roll, 8=hip_yaw, 9=knee, 10=ankle_pitch, 11=ankle_roll
//   Waist: 12=yaw, 13=roll, 14=pitch
//   Left arm:  15=shoulder_pitch, 16=shoulder_roll, 17=shoulder_yaw, 18=elbow, 19=wrist_roll, 20=wrist_pitch, 21=wrist_yaw
//   Right arm: 22=shoulder_pitch, 23=shoulder_roll, 24=shoulder_yaw, 25=elbow, 26=wrist_roll, 27=wrist_pitch, 28=wrist_yaw
//
// Policy order (after joint_ids_map remapping):
//   [0]=SDK0, [1]=SDK6, [2]=SDK12, [3]=SDK1, [4]=SDK7, [5]=SDK13, [6]=SDK2, [7]=SDK8, [8]=SDK14,
//   [9]=SDK3, [10]=SDK9, [11]=SDK15, [12]=SDK22, [13]=SDK4, [14]=SDK10, [15]=SDK16, [16]=SDK23,
//   [17]=SDK5, [18]=SDK11, [19]=SDK17, [20]=SDK24, [21]=SDK18, [22]=SDK25, [23]=SDK19, [24]=SDK26,
//   [25]=SDK20, [26]=SDK27, [27]=SDK21, [28]=SDK28
//
// Inverse mapping (SDK index -> Policy index):
//   SDK 0->0, 1->3, 2->6, 3->9, 4->13, 5->17 (left leg)
//   SDK 6->1, 7->4, 8->7, 9->10, 10->14, 11->18 (right leg)
//   SDK 12->2, 13->5, 14->8 (waist)
//   SDK 15->11, 16->15, 17->19, 18->21, 19->23, 20->25, 21->27 (left arm)
//   SDK 22->12, 23->16, 24->20, 25->22, 26->24, 27->26, 28->28 (right arm)
// ============================================================================

inline Eigen::Vector3f computeKeyBodyPosition_G1(
    const std::string& body_name,
    const std::vector<float>& joint_pos  // Joint positions in POLICY order (after joint_ids_map)
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
    
    // Extract joint angles using POLICY order indices (NOT SDK order!)
    // The joint_pos array is in policy order after joint_ids_map remapping.
    // We use the inverse mapping: SDK index -> Policy index
    // 
    // Left leg (SDK 0-5 -> Policy 0,3,6,9,13,17):
    float left_hip_pitch = joint_pos.size() > 0 ? joint_pos[0] : 0.0f;    // SDK 0 -> Policy 0
    float left_hip_roll = joint_pos.size() > 3 ? joint_pos[3] : 0.0f;     // SDK 1 -> Policy 3
    float left_hip_yaw = joint_pos.size() > 6 ? joint_pos[6] : 0.0f;      // SDK 2 -> Policy 6
    float left_knee = joint_pos.size() > 9 ? joint_pos[9] : 0.0f;         // SDK 3 -> Policy 9
    float left_ankle_pitch = joint_pos.size() > 13 ? joint_pos[13] : 0.0f; // SDK 4 -> Policy 13
    float left_ankle_roll = joint_pos.size() > 17 ? joint_pos[17] : 0.0f;  // SDK 5 -> Policy 17
    
    // Right leg (SDK 6-11 -> Policy 1,4,7,10,14,18):
    float right_hip_pitch = joint_pos.size() > 1 ? joint_pos[1] : 0.0f;   // SDK 6 -> Policy 1
    float right_hip_roll = joint_pos.size() > 4 ? joint_pos[4] : 0.0f;    // SDK 7 -> Policy 4
    float right_hip_yaw = joint_pos.size() > 7 ? joint_pos[7] : 0.0f;     // SDK 8 -> Policy 7
    float right_knee = joint_pos.size() > 10 ? joint_pos[10] : 0.0f;      // SDK 9 -> Policy 10
    float right_ankle_pitch = joint_pos.size() > 14 ? joint_pos[14] : 0.0f; // SDK 10 -> Policy 14
    float right_ankle_roll = joint_pos.size() > 18 ? joint_pos[18] : 0.0f;  // SDK 11 -> Policy 18
    
    // Waist (SDK 12-14 -> Policy 2,5,8):
    float waist_yaw = joint_pos.size() > 2 ? joint_pos[2] : 0.0f;         // SDK 12 -> Policy 2
    float waist_roll = joint_pos.size() > 5 ? joint_pos[5] : 0.0f;        // SDK 13 -> Policy 5
    float waist_pitch = joint_pos.size() > 8 ? joint_pos[8] : 0.0f;       // SDK 14 -> Policy 8
    
    // Left arm (SDK 15-21 -> Policy 11,15,19,21,23,25,27):
    float left_shoulder_pitch = joint_pos.size() > 11 ? joint_pos[11] : 0.0f;  // SDK 15 -> Policy 11
    float left_shoulder_roll = joint_pos.size() > 15 ? joint_pos[15] : 0.0f;   // SDK 16 -> Policy 15
    float left_shoulder_yaw = joint_pos.size() > 19 ? joint_pos[19] : 0.0f;    // SDK 17 -> Policy 19
    float left_elbow = joint_pos.size() > 21 ? joint_pos[21] : 0.0f;           // SDK 18 -> Policy 21
    float left_wrist_roll = joint_pos.size() > 23 ? joint_pos[23] : 0.0f;      // SDK 19 -> Policy 23
    float left_wrist_pitch = joint_pos.size() > 25 ? joint_pos[25] : 0.0f;     // SDK 20 -> Policy 25
    float left_wrist_yaw = joint_pos.size() > 27 ? joint_pos[27] : 0.0f;       // SDK 21 -> Policy 27
    
    // Right arm (SDK 22-28 -> Policy 12,16,20,22,24,26,28):
    float right_shoulder_pitch = joint_pos.size() > 12 ? joint_pos[12] : 0.0f; // SDK 22 -> Policy 12
    float right_shoulder_roll = joint_pos.size() > 16 ? joint_pos[16] : 0.0f;  // SDK 23 -> Policy 16
    float right_shoulder_yaw = joint_pos.size() > 20 ? joint_pos[20] : 0.0f;   // SDK 24 -> Policy 20
    float right_elbow = joint_pos.size() > 22 ? joint_pos[22] : 0.0f;          // SDK 25 -> Policy 22
    float right_wrist_roll = joint_pos.size() > 24 ? joint_pos[24] : 0.0f;     // SDK 26 -> Policy 24
    float right_wrist_pitch = joint_pos.size() > 26 ? joint_pos[26] : 0.0f;    // SDK 27 -> Policy 26
    float right_wrist_yaw = joint_pos.size() > 28 ? joint_pos[28] : 0.0f;      // SDK 28 -> Policy 28
    
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
    
    // DIAGNOSTIC FLAG: Set to true to use fixed default positions instead of FK
    // This helps isolate whether FK errors are causing instability
    // If robot is stable with this = true, then FK is the problem
    static const bool USE_DEFAULT_POSITIONS = false;  // Set to true to test without FK
    
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
        // Print joint positions in policy order with joint names for verification
        // Policy order: [0]=L_hip_p, [1]=R_hip_p, [2]=waist_y, [3]=L_hip_r, [4]=R_hip_r, [5]=waist_r
        spdlog::info("[FK DEBUG] Joint positions in POLICY order:");
        spdlog::info("[FK DEBUG]   [0] L_hip_pitch={:.4f}, [1] R_hip_pitch={:.4f}, [2] waist_yaw={:.4f}",
                    joint_pos[0], joint_pos[1], joint_pos[2]);
        spdlog::info("[FK DEBUG]   [3] L_hip_roll={:.4f}, [4] R_hip_roll={:.4f}, [5] waist_roll={:.4f}",
                    joint_pos[3], joint_pos[4], joint_pos[5]);
        spdlog::info("[FK DEBUG]   Left leg: hip_p[0]={:.3f}, hip_r[3]={:.3f}, hip_y[6]={:.3f}, knee[9]={:.3f}, ank_p[13]={:.3f}, ank_r[17]={:.3f}",
                    joint_pos[0], joint_pos[3], joint_pos[6], joint_pos[9], joint_pos[13], joint_pos[17]);
        spdlog::info("[FK DEBUG]   Right leg: hip_p[1]={:.3f}, hip_r[4]={:.3f}, hip_y[7]={:.3f}, knee[10]={:.3f}, ank_p[14]={:.3f}, ank_r[18]={:.3f}",
                    joint_pos[1], joint_pos[4], joint_pos[7], joint_pos[10], joint_pos[14], joint_pos[18]);
    }
    
    // Compute FK for each key body
    bool fk_error = false;
    bool all_zeros = true;
    
    // Default positions at zero pose (pre-computed from Isaac Lab)
    // These are used for diagnostic testing when USE_DEFAULT_POSITIONS = true
    static const std::map<std::string, Eigen::Vector3f> default_body_positions = {
        {"left_ankle_roll_link", Eigen::Vector3f(0.0f, 0.1165f, -0.756f)},
        {"right_ankle_roll_link", Eigen::Vector3f(0.0f, -0.1165f, -0.756f)},
        {"left_wrist_yaw_link", Eigen::Vector3f(0.188f, 0.244f, 0.054f)},
        {"right_wrist_yaw_link", Eigen::Vector3f(0.188f, -0.244f, 0.054f)},
        {"left_shoulder_roll_link", Eigen::Vector3f(0.0f, 0.138f, 0.292f)},
        {"right_shoulder_roll_link", Eigen::Vector3f(0.0f, -0.138f, 0.292f)},
    };
    
    for (size_t i = 0; i < num_key_bodies; ++i) {
        Eigen::Vector3f pos;
        
        if (USE_DEFAULT_POSITIONS) {
            // Use fixed default positions for diagnostic testing
            auto it = default_body_positions.find(body_names[i]);
            if (it != default_body_positions.end()) {
                pos = it->second;
            } else {
                pos = Eigen::Vector3f::Zero();
            }
            if (should_debug && i == 0) {
                spdlog::warn("[DIAGNOSTIC] Using DEFAULT positions instead of FK!");
            }
        } else {
            // Normal FK computation
            pos = computeKeyBodyPosition_G1(body_names[i], joint_pos);
        }
        
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