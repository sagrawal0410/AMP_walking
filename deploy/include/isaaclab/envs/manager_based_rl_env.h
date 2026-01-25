// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include <spdlog/spdlog.h>
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
            for (YAML::const_iterator it = joint_ids_node.begin(); it != joint_ids_node.end(); ++it) {
                if (it->IsScalar()) {
                    int val = it->as<int>();
                    robot->data.joint_ids_map.push_back(static_cast<float>(val));
                } else {
                    throw std::runtime_error("joint_ids_map contains non-scalar element");
                }
            }
            spdlog::debug("Parsed {} joint IDs", robot->data.joint_ids_map.size());
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

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
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
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute();
        auto action = alg->act(obs);
        action_manager->process_action(action);
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