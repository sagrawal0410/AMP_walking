// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
// Debug utilities for AMP policy deployment diagnostics

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <spdlog/spdlog.h>

namespace isaaclab {
namespace debug {

// Environment variable flags
inline bool is_debug_enabled() {
    static bool cached = false;
    static bool value = false;
    if (!cached) {
        const char* env = std::getenv("DEPLOY_DEBUG");
        value = (env != nullptr && std::string(env) == "1");
        cached = true;
    }
    return value;
}

inline bool is_dump_enabled() {
    static bool cached = false;
    static bool value = false;
    if (!cached) {
        const char* env = std::getenv("DEPLOY_DUMP");
        value = (env != nullptr && std::string(env) == "1");
        cached = true;
    }
    return value;
}

inline std::string get_dump_dir() {
    static std::string cached_dir = "";
    if (cached_dir.empty()) {
        const char* env = std::getenv("DEPLOY_DUMP_DIR");
        if (env != nullptr) {
            cached_dir = std::string(env);
        } else {
            cached_dir = "/tmp/amp_debug";
        }
    }
    return cached_dir;
}

// Check for NaN/Inf in vector
inline bool check_finite(const std::vector<float>& vec, const std::string& name) {
    for (size_t i = 0; i < vec.size(); ++i) {
        if (!std::isfinite(vec[i])) {
            spdlog::error("[DEBUG] {}[{}] = {} (NaN/Inf detected!)", name, i, vec[i]);
            return false;
        }
    }
    return true;
}

// Compute statistics: min, max, mean, std, L2 norm
struct VecStats {
    float min_val;
    float max_val;
    float mean_val;
    float std_val;
    float l2_norm;
    size_t size;
};

inline VecStats compute_stats(const std::vector<float>& vec) {
    VecStats stats;
    stats.size = vec.size();
    
    if (vec.empty()) {
        stats.min_val = stats.max_val = stats.mean_val = stats.std_val = stats.l2_norm = 0.0f;
        return stats;
    }
    
    auto minmax = std::minmax_element(vec.begin(), vec.end());
    stats.min_val = *minmax.first;
    stats.max_val = *minmax.second;
    
    float sum = std::accumulate(vec.begin(), vec.end(), 0.0f);
    stats.mean_val = sum / vec.size();
    
    float var_sum = 0.0f;
    float l2_sum = 0.0f;
    for (float v : vec) {
        float diff = v - stats.mean_val;
        var_sum += diff * diff;
        l2_sum += v * v;
    }
    stats.std_val = std::sqrt(var_sum / vec.size());
    stats.l2_norm = std::sqrt(l2_sum);
    
    return stats;
}

// Print first k values
inline void print_first(const std::vector<float>& vec, const std::string& name, size_t k = 6) {
    k = std::min(k, vec.size());
    std::stringstream ss;
    ss << "[DEBUG] " << name << " first " << k << " values: [";
    for (size_t i = 0; i < k; ++i) {
        if (i > 0) ss << ", ";
        ss << std::fixed << std::setprecision(4) << vec[i];
    }
    ss << "]";
    spdlog::info(ss.str());
}

// Print statistics
inline void print_stats(const std::vector<float>& vec, const std::string& name) {
    VecStats stats = compute_stats(vec);
    spdlog::info("[DEBUG] {} stats: size={}, min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}, L2={:.4f}",
                 name, stats.size, stats.min_val, stats.max_val, stats.mean_val, stats.std_val, stats.l2_norm);
}

// Check orthonormality of rot6 (tan/norm representation)
inline void orthonormal_check_rot6(const std::vector<float>& rot6, const std::string& name) {
    if (rot6.size() != 6) {
        spdlog::warn("[DEBUG] {} rot6 size mismatch: expected 6, got {}", name, rot6.size());
        return;
    }
    
    // Extract tan (first 3) and norm (last 3)
    Eigen::Vector3f tan(rot6[0], rot6[1], rot6[2]);
    Eigen::Vector3f norm(rot6[3], rot6[4], rot6[5]);
    
    float tan_norm = tan.norm();
    float norm_norm = norm.norm();
    float dot_product = tan.dot(norm);
    
    spdlog::info("[DEBUG] {} rot6 orthonormality: ||tan||={:.4f}, ||norm||={:.4f}, tan·norm={:.4f}",
                 name, tan_norm, norm_norm, dot_product);
    
    if (std::abs(tan_norm - 1.0f) > 0.1f) {
        spdlog::warn("[DEBUG] {} WARNING: ||tan|| should be ~1.0, got {:.4f} -> quaternion/rotation conversion may be wrong!",
                     name, tan_norm);
    }
    if (std::abs(norm_norm - 1.0f) > 0.1f) {
        spdlog::warn("[DEBUG] {} WARNING: ||norm|| should be ~1.0, got {:.4f} -> quaternion/rotation conversion may be wrong!",
                     name, norm_norm);
    }
    if (std::abs(dot_product) > 0.1f) {
        spdlog::warn("[DEBUG] {} WARNING: tan·norm should be ~0.0, got {:.4f} -> vectors not orthogonal!",
                     name, dot_product);
    }
}

// Dump vector to CSV file
inline void dump_csv(const std::string& filepath, const std::vector<float>& vec) {
    // Create directory if it doesn't exist
    size_t last_slash = filepath.find_last_of("/");
    if (last_slash != std::string::npos) {
        std::string dir = filepath.substr(0, last_slash);
        std::string cmd = "mkdir -p " + dir;
        system(cmd.c_str());
    }
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        spdlog::warn("[DEBUG] Failed to open dump file: {}", filepath);
        return;
    }
    
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) file << ",";
        file << std::fixed << std::setprecision(8) << vec[i];
    }
    file << "\n";
    file.close();
}

// Count saturation (values > threshold)
inline size_t count_saturation(const std::vector<float>& vec, float threshold = 0.95f) {
    return std::count_if(vec.begin(), vec.end(), 
                         [threshold](float v) { return std::abs(v) > threshold; });
}

// Step counter (thread-safe static)
inline int& get_step_counter() {
    static int step_counter = 0;
    return step_counter;
}

inline void increment_step_counter() {
    get_step_counter()++;
}

inline int get_current_step() {
    return get_step_counter();
}

// Print slice boundaries for 585-dim obs
inline void print_slice_info() {
    spdlog::info("[DEBUG] ========== OBSERVATION SLICE BOUNDARIES (585-dim) ==========");
    spdlog::info("[DEBUG] base_ang_vel:               [0:15)   = 15 dims (3*5)");
    spdlog::info("[DEBUG] root_local_rot_tan_norm:    [15:45)  = 30 dims (6*5)");
    spdlog::info("[DEBUG] keyboard_velocity_commands: [45:60)  = 15 dims (3*5)");
    spdlog::info("[DEBUG] joint_pos:                  [60:205) = 145 dims (29*5)");
    spdlog::info("[DEBUG] joint_vel:                  [205:350)= 145 dims (29*5)");
    spdlog::info("[DEBUG] actions:                    [350:495)= 145 dims (29*5)");
    spdlog::info("[DEBUG] key_body_pos_b:             [495:585) = 90 dims (18*5)");
    spdlog::info("[DEBUG] =============================================================");
    spdlog::info("[DEBUG] INTERPRETATION GUIDE:");
    spdlog::info("[DEBUG] - If rot6 tan/norm norms not ~1.0 -> quaternion/rotation conversion wrong");
    spdlog::info("[DEBUG] - If key_body_pos_b near zeros -> wrong body mapping or FK not working");
    spdlog::info("[DEBUG] - If actions slice != previous raw action -> action obs semantics mismatch");
    spdlog::info("[DEBUG] - If many actions saturated during standing -> normalization/scale/order mismatch");
    spdlog::info("[DEBUG] - If policy Hz differs from training -> control rate mismatch causing twitch");
    spdlog::info("[DEBUG] =============================================================");
}

// Extract slice from vector
inline std::vector<float> extract_slice(const std::vector<float>& vec, size_t start, size_t end) {
    if (end > vec.size()) end = vec.size();
    if (start >= end) return {};
    return std::vector<float>(vec.begin() + start, vec.begin() + end);
}

// Print slice stats
inline void print_slice_stats(const std::vector<float>& obs, size_t start, size_t end, 
                              const std::string& slice_name) {
    auto slice = extract_slice(obs, start, end);
    if (slice.empty()) {
        spdlog::warn("[DEBUG] {} slice [{}, {}) is empty!", slice_name, start, end);
        return;
    }
    print_stats(slice, slice_name);
    print_first(slice, slice_name, 6);
}

} // namespace debug
} // namespace isaaclab
