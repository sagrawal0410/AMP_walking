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
        if(cfg["joint_ids"].IsDefined() && !cfg["joint_ids"].IsNull()) {
            try {
                _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
                _action_dim = _joint_ids.size();
            } catch(const std::exception& e) {
                // If parsing fails, use all joints
                spdlog::warn("Failed to parse joint_ids, using all joints: {}", e.what());
                _action_dim = env->robot->data.joint_ids_map.size();
            }
        } else {
            _action_dim = env->robot->data.joint_ids_map.size();
        }
        _raw_actions.resize(_action_dim, 0.0f);
        _processed_actions.resize(_action_dim, 0.0f);
        if(cfg["scale"].IsDefined() && !cfg["scale"].IsNull()) {
            _scale = cfg["scale"].as<std::vector<float>>();
        }
        if(cfg["offset"].IsDefined() && !cfg["offset"].IsNull()) {
            _offset = cfg["offset"].as<std::vector<float>>();
        }
        if(cfg["clip"].IsDefined() && !cfg["clip"].IsNull()) {
            _clip = cfg["clip"].as<std::vector<std::vector<float> >>();
        }
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
        
        // Debug: Print action statistics every 100 calls
        static int action_debug_count = 0;
        if (action_debug_count++ % 100 == 0) {
            // Check raw actions for saturation
            float max_raw = 0.0f;
            int saturated_count = 0;
            for (size_t i = 0; i < _raw_actions.size(); ++i) {
                max_raw = std::max(max_raw, std::abs(_raw_actions[i]));
                if (std::abs(_raw_actions[i]) > 0.95f) saturated_count++;
            }
            
            // Check processed actions
            float max_proc = 0.0f;
            for (size_t i = 0; i < _processed_actions.size(); ++i) {
                max_proc = std::max(max_proc, std::abs(_processed_actions[i]));
            }
            
            spdlog::info("[ACTION DEBUG] Raw: max={:.4f}, saturated={}/{}, Processed: max={:.4f} rad",
                        max_raw, saturated_count, _raw_actions.size(), max_proc);
            
            // Print first few raw and processed actions
            if (_raw_actions.size() >= 6 && _processed_actions.size() >= 6) {
                spdlog::info("[ACTION DEBUG] Raw[0:5]: [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]",
                            _raw_actions[0], _raw_actions[1], _raw_actions[2],
                            _raw_actions[3], _raw_actions[4], _raw_actions[5]);
                spdlog::info("[ACTION DEBUG] Proc[0:5]: [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]",
                            _processed_actions[0], _processed_actions[1], _processed_actions[2],
                            _processed_actions[3], _processed_actions[4], _processed_actions[5]);
            }
            
            // Warning if actions seem wrong
            if (max_raw > 10.0f) {
                spdlog::warn("[ACTION DEBUG] Raw actions very large (>10)! Policy output may be wrong.");
            }
            if (saturated_count > _raw_actions.size() / 3) {
                spdlog::warn("[ACTION DEBUG] Many actions saturated (>1/3)! Possible observation mismatch.");
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