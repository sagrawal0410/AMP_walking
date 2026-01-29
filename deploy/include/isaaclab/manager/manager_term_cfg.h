// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <deque>
#include <vector>
#include <functional>
#include <numeric>
#include "isaaclab/utils/debug_utils.h"
#include <spdlog/spdlog.h>

namespace isaaclab
{

class ManagerBasedRLEnv;

using ObsFunc = std::function<std::vector<float>(ManagerBasedRLEnv*, YAML::Node)>;

struct ObservationTermCfg
{
    YAML::Node params;
    ObsFunc func;
    std::vector<float> clip;
    std::vector<float> scale;
    int history_length = 1;
    bool scale_first = false;

    void reset(std::vector<float> obs)
    {
        buff_.clear();
        // Safe initialization: fill history with first observed frame repeated (not zeros)
        for(int i(0); i < history_length; ++i) {
            add(obs);  // This will apply scale/clip, but we want the same value repeated
        }
        
        // Debug: log when buffer is fully warmed
        if (isaaclab::debug::is_debug_enabled()) {
            if (buff_.size() == static_cast<size_t>(history_length)) {
                spdlog::info("[DEBUG] History buffer fully warmed: size={}, history_length={}", 
                            buff_.size(), history_length);
            }
        }
    }

    void add(std::vector<float> obs)
    {
        for(int j = 0; j < obs.size(); ++j)
        {
            if(scale_first) {
                if(!scale.empty()) obs[j] *= scale[j];
                if (!clip.empty()) {
                    obs[j] = std::clamp(obs[j], clip[0], clip[1]);
                }
            } else {
                if (!clip.empty()) {
                    obs[j] = std::clamp(obs[j], clip[0], clip[1]);
                }
                if(!scale.empty()) obs[j] *= scale[j];
            }
        }
        buff_.push_back(obs);

        if (buff_.size() > history_length) buff_.pop_front();
    }

    const std::vector<float> & get(int n) const { return buff_[n]; }

    const std::vector<float> get() const
    {
        std::vector<float> concatenated;
        for (const auto& entry : buff_) {
            concatenated.insert(concatenated.end(), entry.begin(), entry.end());
        }
        return concatenated;
    }

    const std::size_t size() const { return std::accumulate(buff_.begin(), buff_.end(), 0,
        [](std::size_t sum, const auto& v) { return sum + v.size(); }); }

private:
    // Complete circular buffer with most recent entry at the end and oldest entry at the beginning.
    std::deque<std::vector<float>> buff_;
};

};