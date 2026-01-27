// Test script for Forward Kinematics
// Compile: g++ -std=c++17 -I../../include -I/usr/include/eigen3 test_fk.cpp -o test_fk
// Run: ./test_fk

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "isaaclab/envs/mdp/observations/observations.h"

using namespace isaaclab::mdp;

void printVector(const std::string& name, const Eigen::Vector3f& vec) {
    std::cout << std::setw(30) << name << ": ["
              << std::fixed << std::setprecision(4)
              << std::setw(8) << vec.x() << ", "
              << std::setw(8) << vec.y() << ", "
              << std::setw(8) << vec.z() << "]" << std::endl;
}

void testFK(const std::string& body_name, const std::vector<float>& joint_pos) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Testing FK for: " << body_name << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    Eigen::Vector3f pos = computeKeyBodyPosition_G1(body_name, joint_pos);
    
    printVector("Computed Position", pos);
    
    // Check for invalid values
    bool has_nan = std::isnan(pos.x()) || std::isnan(pos.y()) || std::isnan(pos.z());
    bool has_inf = std::isinf(pos.x()) || std::isinf(pos.y()) || std::isinf(pos.z());
    
    if (has_nan) {
        std::cout << "WARNING: Position contains NaN!" << std::endl;
    }
    if (has_inf) {
        std::cout << "WARNING: Position contains Inf!" << std::endl;
    }
    
    // Print joint angles used
    std::cout << "\nJoint angles used:" << std::endl;
    std::vector<std::string> joint_names = {
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"
    };
    
    for (size_t i = 0; i < std::min(joint_pos.size(), joint_names.size()); ++i) {
        std::cout << "  " << std::setw(25) << joint_names[i] << ": " 
                  << std::fixed << std::setprecision(4) << joint_pos[i] << " rad" << std::endl;
    }
}

int main() {
    std::cout << "Forward Kinematics Diagnostic Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Test 1: Zero pose (all joints at 0)
    std::cout << "\n\nTEST 1: Zero Pose (all joints = 0)" << std::endl;
    std::vector<float> zero_pose(29, 0.0f);
    
    std::vector<std::string> key_bodies = {
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link"
    };
    
    for (const auto& body : key_bodies) {
        testFK(body, zero_pose);
    }
    
    // Test 2: Known pose (standing pose with arms down)
    std::cout << "\n\nTEST 2: Standing Pose" << std::endl;
    std::vector<float> standing_pose(29, 0.0f);
    // Set some known angles for standing
    standing_pose[3] = 0.5f;  // left_knee slightly bent
    standing_pose[9] = 0.5f;  // right_knee slightly bent
    standing_pose[4] = -0.2f; // left_ankle_pitch
    standing_pose[10] = -0.2f; // right_ankle_pitch
    
    for (const auto& body : key_bodies) {
        testFK(body, standing_pose);
    }
    
    // Test 3: Extreme pose (to check for numerical issues)
    std::cout << "\n\nTEST 3: Extreme Pose (large angles)" << std::endl;
    std::vector<float> extreme_pose(29, 0.0f);
    extreme_pose[0] = 1.0f;   // left_hip_pitch
    extreme_pose[1] = 0.5f;   // left_hip_roll
    extreme_pose[3] = 1.5f;   // left_knee
    extreme_pose[15] = 1.0f;  // left_shoulder_pitch
    extreme_pose[18] = 1.0f;  // left_elbow
    
    for (const auto& body : key_bodies) {
        testFK(body, extreme_pose);
    }
    
    // Test 4: Check symmetry (left vs right should be symmetric in zero pose)
    std::cout << "\n\nTEST 4: Symmetry Check (zero pose)" << std::endl;
    Eigen::Vector3f left_ankle = computeKeyBodyPosition_G1("left_ankle_roll_link", zero_pose);
    Eigen::Vector3f right_ankle = computeKeyBodyPosition_G1("right_ankle_roll_link", zero_pose);
    
    std::cout << "Left ankle:  [" << left_ankle.x() << ", " << left_ankle.y() << ", " << left_ankle.z() << "]" << std::endl;
    std::cout << "Right ankle: [" << right_ankle.x() << ", " << right_ankle.y() << ", " << right_ankle.z() << "]" << std::endl;
    
    // In zero pose, left and right should be symmetric (y-coordinate should be opposite)
    float y_diff = std::abs(left_ankle.y() + right_ankle.y());  // Should be ~0
    float x_diff = std::abs(left_ankle.x() - right_ankle.x());  // Should be ~0
    float z_diff = std::abs(left_ankle.z() - right_ankle.z());  // Should be ~0
    
    std::cout << "\nSymmetry check:" << std::endl;
    std::cout << "  Y difference (should be ~0): " << y_diff << std::endl;
    std::cout << "  X difference (should be ~0): " << x_diff << std::endl;
    std::cout << "  Z difference (should be ~0): " << z_diff << std::endl;
    
    if (y_diff > 0.01f || x_diff > 0.01f || z_diff > 0.01f) {
        std::cout << "WARNING: Asymmetry detected! FK may have issues." << std::endl;
    } else {
        std::cout << "OK: Symmetry check passed." << std::endl;
    }
    
    // Test 5: Check expected ranges (bodies should be within reasonable bounds)
    std::cout << "\n\nTEST 5: Range Check" << std::endl;
    std::cout << "Expected ranges (approximate):" << std::endl;
    std::cout << "  Ankles: x ~ [-0.5, 0.5], y ~ [-0.2, 0.2], z ~ [-1.0, -0.5]" << std::endl;
    std::cout << "  Wrists: x ~ [0.0, 0.5], y ~ [-0.5, 0.5], z ~ [0.0, 0.5]" << std::endl;
    std::cout << "  Shoulders: x ~ [0.0, 0.2], y ~ [-0.2, 0.2], z ~ [0.2, 0.4]" << std::endl;
    
    for (const auto& body : key_bodies) {
        Eigen::Vector3f pos = computeKeyBodyPosition_G1(body, zero_pose);
        std::cout << "\n" << body << ":" << std::endl;
        printVector("  Position", pos);
        
        // Check if position is reasonable (not too far from origin)
        float dist = pos.norm();
        if (dist > 2.0f) {
            std::cout << "  WARNING: Distance from origin (" << dist << ") seems too large!" << std::endl;
        }
    }
    
    std::cout << "\n\nDiagnostic complete!" << std::endl;
    return 0;
}
