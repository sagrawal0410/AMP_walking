// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/manager/action_manager.h"

namespace isaaclab
{

class JointAction : public ActionTerm
{
public:
    JointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :ActionTerm(cfg, env)
    {
        spdlog::debug("JointAction: Parsing config...");
        
        // Parse joint_ids - if not specified, null, or not a sequence, use all joints
        // Extra defensive: also check IsSequence() to handle yaml-cpp edge cases
        if(cfg["joint_ids"].IsDefined() && !cfg["joint_ids"].IsNull() && cfg["joint_ids"].IsSequence()) {
            spdlog::debug("JointAction: joint_ids defined as sequence, parsing as vector<int>...");
            try {
                _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
                _action_dim = _joint_ids.size();
                spdlog::debug("JointAction: Parsed {} joint_ids", _action_dim);
            } catch(const std::exception& e) {
                // If parsing fails, use all joints
                spdlog::warn("Failed to parse joint_ids, using all joints: {}", e.what());
                _action_dim = env->robot->data.joint_ids_map.size();
            }
        } else {
            if(cfg["joint_ids"].IsDefined()) {
                spdlog::debug("JointAction: joint_ids defined but is null or not a sequence (type={}), using all joints", 
                             cfg["joint_ids"].Type());
            } else {
                spdlog::debug("JointAction: joint_ids not defined, using all joints");
            }
            _action_dim = env->robot->data.joint_ids_map.size();
            spdlog::debug("JointAction: Using all {} joints", _action_dim);
        }
        
        _raw_actions.resize(_action_dim, 0.0f);
        _processed_actions.resize(_action_dim, 0.0f);
        
        // Parse scale - must be a sequence of floats
        try {
            if(cfg["scale"].IsDefined() && !cfg["scale"].IsNull() && cfg["scale"].IsSequence()) {
                spdlog::debug("JointAction: Parsing scale...");
                _scale = cfg["scale"].as<std::vector<float>>();
                spdlog::debug("JointAction: Parsed {} scale values", _scale.size());
            } else if(cfg["scale"].IsDefined() && !cfg["scale"].IsNull()) {
                spdlog::warn("JointAction: scale defined but not a sequence (type={}), ignoring", cfg["scale"].Type());
            }
        } catch(const std::exception& e) {
            spdlog::error("Failed to parse scale: {}", e.what());
            throw;
        }
        
        // Parse offset - must be a sequence of floats
        try {
            if(cfg["offset"].IsDefined() && !cfg["offset"].IsNull() && cfg["offset"].IsSequence()) {
                spdlog::debug("JointAction: Parsing offset...");
                _offset = cfg["offset"].as<std::vector<float>>();
                spdlog::debug("JointAction: Parsed {} offset values", _offset.size());
            } else if(cfg["offset"].IsDefined() && !cfg["offset"].IsNull()) {
                spdlog::warn("JointAction: offset defined but not a sequence (type={}), ignoring", cfg["offset"].Type());
            }
        } catch(const std::exception& e) {
            spdlog::error("Failed to parse offset: {}", e.what());
            throw;
        }
        
        // Parse clip - must be a sequence of sequences of floats
        try {
            if(cfg["clip"].IsDefined() && !cfg["clip"].IsNull() && cfg["clip"].IsSequence()) {
                spdlog::debug("JointAction: Parsing clip...");
                _clip = cfg["clip"].as<std::vector<std::vector<float> >>();
                spdlog::debug("JointAction: Parsed {} clip entries", _clip.size());
            } else if(cfg["clip"].IsDefined() && !cfg["clip"].IsNull()) {
                spdlog::warn("JointAction: clip defined but not a sequence (type={}), ignoring", cfg["clip"].Type());
            }
        } catch(const std::exception& e) {
            spdlog::error("Failed to parse clip: {}", e.what());
            throw;
        }
        
        spdlog::debug("JointAction: Config parsing complete");
    }

    virtual void process_actions(std::vector<float> actions)
    {
        // TODO: modify action by joint_ids
        _raw_actions = actions;
        for(int i(0); i<_action_dim; ++i)
        {
            if(!_scale.empty()) {
                _processed_actions[i] = _raw_actions[i] * _scale[i];
            } else {
                _processed_actions[i] = _raw_actions[i];
            }
            if(!_offset.empty()) {
                _processed_actions[i] += _offset[i];
            }
        }
        if(!_clip.empty())
        {
            for(int i(0); i<_action_dim; ++i) {
                _processed_actions[i] = std::clamp(_processed_actions[i], _clip[i][0], _clip[i][1]);
            }
        }
    }


    int action_dim() 
    {
        return _action_dim;
    }

    std::vector<float> raw_actions() 
    {
        return _raw_actions;
    }
    
    std::vector<float> processed_actions() 
    {
        return _processed_actions;
    }

    void reset()
    {
        _raw_actions.assign(_action_dim, 0.0f);
    }

protected:
    int _action_dim;
    std::vector<int> _joint_ids;

    std::vector<float> _raw_actions;
    std::vector<float> _processed_actions;

    std::vector<float> _scale;
    std::vector<float> _offset;
    std::vector<std::vector<float> > _clip;
};


class JointPositionAction : public JointAction
{
public:
    JointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

class JointVelocityAction : public JointAction
{
public:
    JointVelocityAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

REGISTER_ACTION(JointPositionAction);
REGISTER_ACTION(JointVelocityAction);

};