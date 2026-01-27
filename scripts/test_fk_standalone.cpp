// Standalone test script for Forward Kinematics
// Compile: g++ -std=c++17 -I../../include -I/usr/include/eigen3 test_fk_standalone.cpp -o test_fk
// Run: ./test_fk

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

// Copy FK helper functions (without dependencies)
struct Transform {
    Eigen::Matrix3f R;
    Eigen::Vector3f p;
    
    Transform() : R(Eigen::Matrix3f::Identity()), p(Eigen::Vector3f::Zero()) {}
    Transform(const Eigen::Matrix3f& rot, const Eigen::Vector3f& trans) : R(rot), p(trans) {}
    
    Transform operator*(const Transform& other) const {
        Transform result;
        result.R = R * other.R;
        result.p = R * other.p + p;
        return result;
    }
    
    Eigen::Vector3f position() const { return p; }
};

inline Eigen::Matrix3f axisAngleToRotation(const Eigen::Vector3f& axis, float angle) {
    if (std::abs(angle) < 1e-8f) {
        return Eigen::Matrix3f::Identity();
    }
    Eigen::AngleAxisf aa(angle, axis.normalized());
    return aa.toRotationMatrix();
}

inline Eigen::Matrix3f quatToRotation(float w, float x, float y, float z) {
    Eigen::Quaternionf q(w, x, y, z);
    q.normalize();
    return q.toRotationMatrix();
}

inline Transform makeTransform(const Eigen::Vector3f& pos, float qw = 1.0f, float qx = 0.0f, float qy = 0.0f, float qz = 0.0f) {
    return Transform(quatToRotation(qw, qx, qy, qz), pos);
}

inline Transform jointTransform(const Eigen::Vector3f& axis, float angle) {
    return Transform(axisAngleToRotation(axis, angle), Eigen::Vector3f::Zero());
}

// Copy the FK function from observations.h
Eigen::Vector3f computeKeyBodyPosition_G1(
    const std::string& body_name,
    const std::vector<float>& joint_pos
) {
    const Eigen::Vector3f AXIS_X(1.0f, 0.0f, 0.0f);
    const Eigen::Vector3f AXIS_Y(0.0f, 1.0f, 0.0f);
    const Eigen::Vector3f AXIS_Z(0.0f, 0.0f, 1.0f);
    
    // Static transforms from XML
    auto T_pelvis_to_left_hip_pitch = makeTransform(Eigen::Vector3f(0.0f, 0.064452f, -0.1027f));
    auto T_left_hip_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, 0.052f, -0.030465f), 0.996179f, 0.0f, -0.0873386f, 0.0f);
    auto T_left_hip_roll_to_yaw = makeTransform(Eigen::Vector3f(0.025001f, 0.0f, -0.12412f));
    auto T_left_hip_yaw_to_knee = makeTransform(Eigen::Vector3f(-0.078273f, 0.0021489f, -0.17734f), 0.996179f, 0.0f, 0.0873386f, 0.0f);
    auto T_left_knee_to_ankle_pitch = makeTransform(Eigen::Vector3f(0.0f, -9.4445e-05f, -0.30001f));
    auto T_left_ankle_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, 0.0f, -0.017558f));
    
    auto T_pelvis_to_right_hip_pitch = makeTransform(Eigen::Vector3f(0.0f, -0.064452f, -0.1027f));
    auto T_right_hip_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, -0.052f, -0.030465f), 0.996179f, 0.0f, -0.0873386f, 0.0f);
    auto T_right_hip_roll_to_yaw = makeTransform(Eigen::Vector3f(0.025001f, 0.0f, -0.12412f));
    auto T_right_hip_yaw_to_knee = makeTransform(Eigen::Vector3f(-0.078273f, -0.0021489f, -0.17734f), 0.996179f, 0.0f, 0.0873386f, 0.0f);
    auto T_right_knee_to_ankle_pitch = makeTransform(Eigen::Vector3f(0.0f, 9.4445e-05f, -0.30001f));
    auto T_right_ankle_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, 0.0f, -0.017558f));
    
    auto T_pelvis_to_waist_yaw = makeTransform(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
    auto T_waist_yaw_to_roll = makeTransform(Eigen::Vector3f(-0.0039635f, 0.0f, 0.035f));
    auto T_waist_roll_to_torso = makeTransform(Eigen::Vector3f(0.0f, 0.0f, 0.019f));
    
    auto T_torso_to_left_shoulder_pitch = makeTransform(Eigen::Vector3f(0.0039563f, 0.10022f, 0.23778f), 0.990264f, 0.139201f, 1.38722e-05f, -9.86868e-05f);
    auto T_left_shoulder_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, 0.038f, -0.013831f), 0.990268f, -0.139172f, 0.0f, 0.0f);
    auto T_left_shoulder_roll_to_yaw = makeTransform(Eigen::Vector3f(0.0f, 0.00624f, -0.1032f));
    auto T_left_shoulder_yaw_to_elbow = makeTransform(Eigen::Vector3f(0.015783f, 0.0f, -0.080518f));
    auto T_left_elbow_to_wrist_roll = makeTransform(Eigen::Vector3f(0.1f, 0.00188791f, -0.01f));
    auto T_left_wrist_roll_to_pitch = makeTransform(Eigen::Vector3f(0.038f, 0.0f, 0.0f));
    auto T_left_wrist_pitch_to_yaw = makeTransform(Eigen::Vector3f(0.046f, 0.0f, 0.0f));
    
    auto T_torso_to_right_shoulder_pitch = makeTransform(Eigen::Vector3f(0.0039563f, -0.10021f, 0.23778f), 0.990264f, -0.139201f, 1.38722e-05f, 9.86868e-05f);
    auto T_right_shoulder_pitch_to_roll = makeTransform(Eigen::Vector3f(0.0f, -0.038f, -0.013831f), 0.990268f, 0.139172f, 0.0f, 0.0f);
    auto T_right_shoulder_roll_to_yaw = makeTransform(Eigen::Vector3f(0.0f, -0.00624f, -0.1032f));
    auto T_right_shoulder_yaw_to_elbow = makeTransform(Eigen::Vector3f(0.015783f, 0.0f, -0.080518f));
    auto T_right_elbow_to_wrist_roll = makeTransform(Eigen::Vector3f(0.1f, -0.00188791f, -0.01f));
    auto T_right_wrist_roll_to_pitch = makeTransform(Eigen::Vector3f(0.038f, 0.0f, 0.0f));
    auto T_right_wrist_pitch_to_yaw = makeTransform(Eigen::Vector3f(0.046f, 0.0f, 0.0f));
    
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
        return Eigen::Vector3f::Zero();
    }
    
    return T_result.position();
}

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
    
    bool has_nan = std::isnan(pos.x()) || std::isnan(pos.y()) || std::isnan(pos.z());
    bool has_inf = std::isinf(pos.x()) || std::isinf(pos.y()) || std::isinf(pos.z());
    
    if (has_nan) {
        std::cout << "WARNING: Position contains NaN!" << std::endl;
    }
    if (has_inf) {
        std::cout << "WARNING: Position contains Inf!" << std::endl;
    }
}

int main() {
    std::cout << "Forward Kinematics Diagnostic Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    std::vector<float> zero_pose(29, 0.0f);
    
    std::vector<std::string> key_bodies = {
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link"
    };
    
    std::cout << "\n\nTEST 1: Zero Pose (all joints = 0)" << std::endl;
    for (const auto& body : key_bodies) {
        testFK(body, zero_pose);
    }
    
    std::cout << "\n\nTEST 2: Standing Pose" << std::endl;
    std::vector<float> standing_pose(29, 0.0f);
    standing_pose[3] = 0.5f;
    standing_pose[9] = 0.5f;
    standing_pose[4] = -0.2f;
    standing_pose[10] = -0.2f;
    
    for (const auto& body : key_bodies) {
        testFK(body, standing_pose);
    }
    
    std::cout << "\n\nTEST 3: Symmetry Check (zero pose)" << std::endl;
    Eigen::Vector3f left_ankle = computeKeyBodyPosition_G1("left_ankle_roll_link", zero_pose);
    Eigen::Vector3f right_ankle = computeKeyBodyPosition_G1("right_ankle_roll_link", zero_pose);
    
    std::cout << "Left ankle:  [" << left_ankle.x() << ", " << left_ankle.y() << ", " << left_ankle.z() << "]" << std::endl;
    std::cout << "Right ankle: [" << right_ankle.x() << ", " << right_ankle.y() << ", " << right_ankle.z() << "]" << std::endl;
    
    float y_diff = std::abs(left_ankle.y() + right_ankle.y());
    float x_diff = std::abs(left_ankle.x() - right_ankle.x());
    float z_diff = std::abs(left_ankle.z() - right_ankle.z());
    
    std::cout << "\nSymmetry check:" << std::endl;
    std::cout << "  Y difference (should be ~0): " << y_diff << std::endl;
    std::cout << "  X difference (should be ~0): " << x_diff << std::endl;
    std::cout << "  Z difference (should be ~0): " << z_diff << std::endl;
    
    if (y_diff > 0.01f || x_diff > 0.01f || z_diff > 0.01f) {
        std::cout << "WARNING: Asymmetry detected!" << std::endl;
    } else {
        std::cout << "OK: Symmetry check passed." << std::endl;
    }
    
    std::cout << "\n\nDiagnostic complete!" << std::endl;
    return 0;
}
