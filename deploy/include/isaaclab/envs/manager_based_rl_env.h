// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include "isaaclab/utils/debug_utils.h"
#include <iostream>
#include <spdlog/spdlog.h>
#include <chrono>
#include "isaaclab/utils/utils.h"

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration with error handling
        try {
            spdlog::debug("Parsing step_dt...");
            this->step_dt = cfg["step_dt"].as<float>();
        } catch (const std::exception& e) {
            spdlog::error("Failed to parse step_dt: {}", e.what());
            throw;
        }
        
        try {
            spdlog::debug("Parsing joint_ids_map...");
            auto joint_ids_node = cfg["joint_ids_map"];
            if (!joint_ids_node.IsDefined()) {
                throw std::runtime_error("joint_ids_map not found in config");
            }
            if (!joint_ids_node.IsSequence()) {
                throw std::runtime_error("joint_ids_map is not a sequence/array");
            }
            
            // Use iterator-based parsing (more robust with multi-line YAML arrays)
            robot->data.joint_ids_map.clear();
            size_t idx = 0;
            for (YAML::const_iterator it = joint_ids_node.begin(); it != joint_ids_node.end(); ++it, ++idx) {
                int val = 0;
                if (it->IsNull()) {
                    // Skip null values or treat as 0
                    spdlog::warn("joint_ids_map[{}] is null, using 0", idx);
                } else if (it->IsScalar()) {
                    // Get raw scalar string and parse
                    std::string str_val = it->Scalar();
                    try {
                        val = std::stoi(str_val);
                    } catch (const std::exception& e) {
                        spdlog::error("Failed to convert joint_ids_map[{}] '{}' to int: {}", idx, str_val, e.what());
                        throw;
                    }
                } else {
                    spdlog::error("joint_ids_map[{}] has unexpected type: {}", idx, it->Type());
                    throw std::runtime_error("Unexpected node type in joint_ids_map");
                }
                robot->data.joint_ids_map.push_back(static_cast<float>(val));
            }
            spdlog::debug("Parsed {} joint IDs", robot->data.joint_ids_map.size());
            
            // Joint map debug (PART 7)
            if (isaaclab::debug::is_debug_enabled()) {
                spdlog::info("[DEBUG] ========== JOINT MAP DEBUG ==========");
                spdlog::info("[DEBUG] joint_ids_map size: {}", robot->data.joint_ids_map.size());
                
                // Print map
                std::stringstream ss;
                ss << "[DEBUG] joint_ids_map = [";
                for (size_t i = 0; i < robot->data.joint_ids_map.size() && i < 10; ++i) {
                    if (i > 0) ss << ", ";
                    ss << static_cast<int>(robot->data.joint_ids_map[i]);
                }
                if (robot->data.joint_ids_map.size() > 10) ss << ", ...";
                ss << "]";
                spdlog::info(ss.str());
                
                // Compute mapping checksum
                int64_t checksum = 0;
                const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109};
                for (size_t i = 0; i < robot->data.joint_ids_map.size() && i < 29; ++i) {
                    checksum += static_cast<int>(robot->data.joint_ids_map[i]) * primes[i];
                }
                spdlog::info("[DEBUG] Joint map checksum: {}", checksum);
                
                // Validate: check for duplicates and out-of-range
                std::vector<bool> seen(robot->data.joint_ids_map.size(), false);
                bool has_duplicate = false;
                bool out_of_range = false;
                for (size_t i = 0; i < robot->data.joint_ids_map.size(); ++i) {
                    int idx = static_cast<int>(robot->data.joint_ids_map[i]);
                    if (idx < 0 || idx >= static_cast<int>(robot->data.joint_ids_map.size())) {
                        out_of_range = true;
                        spdlog::warn("[DEBUG] WARNING: joint_ids_map[{}] = {} is out of range!", i, idx);
                    }
                    if (seen[idx]) {
                        has_duplicate = true;
                        spdlog::warn("[DEBUG] WARNING: joint_ids_map has duplicate index {}!", idx);
                    }
                    seen[idx] = true;
                }
                if (!has_duplicate && !out_of_range) {
                    spdlog::info("[DEBUG] Joint map validation: OK (no duplicates, all in range)");
                }
                spdlog::info("[DEBUG] =====================================");
            }
        } catch (const std::exception& e) {
            spdlog::error("Failed to parse joint_ids_map: {}", e.what());
            throw;
        }
        
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        try {
            spdlog::debug("Parsing default_joint_pos...");
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        } catch (const std::exception& e) {
            spdlog::error("Failed to parse default_joint_pos: {}", e.what());
            throw;
        }
        
        try {
            spdlog::debug("Parsing stiffness...");
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
        } catch (const std::exception& e) {
            spdlog::error("Failed to parse stiffness: {}", e.what());
            throw;
        }
        
        try {
            spdlog::debug("Parsing damping...");
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        } catch (const std::exception& e) {
            spdlog::error("Failed to parse damping: {}", e.what());
            throw;
        }

        robot->update();

        // load managers with detailed error handling
        try {
            spdlog::debug("Creating ActionManager...");
            action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
            spdlog::debug("ActionManager created successfully");
        } catch (const std::exception& e) {
            spdlog::error("Failed to create ActionManager: {}", e.what());
            throw;
        }
        
        try {
            spdlog::debug("Creating ObservationManager...");
            observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
            spdlog::debug("ObservationManager created successfully");
        } catch (const std::exception& e) {
            spdlog::error("Failed to create ObservationManager: {}", e.what());
            throw;
        }
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {
        // Timing debug
        static auto last_step_time = std::chrono::steady_clock::now();
        static int step_count = 0;
        auto now = std::chrono::steady_clock::now();
        auto dt_actual = std::chrono::duration<double>(now - last_step_time).count();
        last_step_time = now;
        step_count++;
        
        if (isaaclab::debug::is_debug_enabled() && step_count % 50 == 0) {
            double policy_hz = 1.0 / dt_actual;
            spdlog::info("[DEBUG] ========== CONTROL RATE DEBUG ==========");
            spdlog::info("[DEBUG] Policy step {}: dt_actual={:.4f}s, policy_hz={:.2f}Hz", step_count, dt_actual, policy_hz);
            spdlog::info("[DEBUG] Expected: step_dt={:.4f}s, expected_hz={:.2f}Hz", step_dt, 1.0/step_dt);
            if (std::abs(policy_hz - 1.0/step_dt) > 5.0) {
                spdlog::warn("[DEBUG] WARNING: Policy Hz differs from training! This can cause twitching!");
            }
            spdlog::info("[DEBUG] ========================================");
        }
        
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute();
        auto action = alg->act(obs);
        action_manager->process_action(action);
        
        // Increment step counter for dumps
        isaaclab::debug::increment_step_counter();
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;
};

};