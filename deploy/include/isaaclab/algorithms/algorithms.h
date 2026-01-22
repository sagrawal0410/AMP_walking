// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Get output shape
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.release());

        action.resize(output_shape[1]);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Validate observations before passing to policy
        bool obs_valid = true;
        for (const auto& name : input_names) {
            const std::string name_str(name);
            auto& input_data = obs.at(name_str);
            for(size_t i = 0; i < input_data.size(); ++i) {
                if(!std::isfinite(input_data[i])) {
                    spdlog::error("Invalid observation[{}][{}] = {} (NaN/Inf detected)!", name_str, i, input_data[i]);
                    obs_valid = false;
                }
            }
        }
        
        // If observations are invalid, return zero actions
        if(!obs_valid) {
            spdlog::error("Invalid observations detected! Returning zero actions.");
            std::lock_guard<std::mutex> lock(act_mtx_);
            std::fill(action.begin(), action.end(), 0.0f);
            return action;
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            
            // Handle size mismatch: pad with zeros if observation is smaller than expected
            if(input_data.size() < input_sizes[i]) {
                spdlog::warn("Observation[{}] size mismatch: got {}, expected {}. Padding with zeros.", name_str, input_data.size(), input_sizes[i]);
                // Create a padded copy of the observation
                std::vector<float> padded_data(input_sizes[i], 0.0f);
                std::copy(input_data.begin(), input_data.end(), padded_data.begin());
                auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, padded_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
                input_tensors.push_back(std::move(input_tensor));
            }
            else if(input_data.size() > input_sizes[i]) {
                spdlog::error("Observation[{}] size mismatch: got {}, expected {}. Truncating.", name_str, input_data.size(), input_sizes[i]);
                // Create a truncated copy of the observation
                std::vector<float> truncated_data(input_data.begin(), input_data.begin() + input_sizes[i]);
                auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, truncated_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
                input_tensors.push_back(std::move(input_tensor));
            }
            else {
                // Size matches exactly
                auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
                input_tensors.push_back(std::move(input_tensor));
            }
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        
        // Validate policy output
        bool action_valid = true;
        for(size_t i = 0; i < action.size(); ++i) {
            if(!std::isfinite(action[i])) {
                spdlog::error("Policy output NaN/Inf at action[{}] = {}! Setting to zero.", i, action[i]);
                action[i] = 0.0f;
                action_valid = false;
            }
        }
        
        if(!action_valid) {
            spdlog::error("Policy produced invalid outputs! All actions set to zero.");
        }
        
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};
};