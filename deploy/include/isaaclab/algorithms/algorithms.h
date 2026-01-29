// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include "isaaclab/utils/debug_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
    explicit OrtRunner(const std::string& model_path)
    {
        // Init ORT runtime
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        // ---- Inputs ----
        const size_t num_inputs = session->GetInputCount();
        input_names_str.reserve(num_inputs);
        input_names.reserve(num_inputs);
        input_shapes.reserve(num_inputs);
        input_sizes.reserve(num_inputs);

        for (size_t i = 0; i < num_inputs; ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);

            // Get name safely (copy into std::string so no leaks)
            auto input_name_alloc = session->GetInputNameAllocated(i, allocator);
            input_names_str.emplace_back(input_name_alloc.get());

            // Store C-string pointers that remain valid (strings won't reallocate after reserve)
            input_names.push_back(input_names_str.back().c_str());

            // Get shape and sanitize dynamic dims (-1 -> 1)
            auto shape = input_type.GetTensorTypeAndShapeInfo().GetShape();
            for (auto& d : shape) {
                if (d < 0) d = 1;  // dynamic -> assume batch=1
            }
            input_shapes.push_back(shape);

            // Compute expected input size (numel)
            size_t sz = 1;
            for (const auto& dim : shape) {
                sz *= static_cast<size_t>(dim);
            }
            input_sizes.push_back(sz);
        }

        // ---- Outputs ----
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        for (auto& d : output_shape) {
            if (d < 0) d = 1;
        }

        auto output_name_alloc = session->GetOutputNameAllocated(0, allocator);
        output_names_str.emplace_back(output_name_alloc.get());
        output_names.push_back(output_names_str.back().c_str());

        // Try to infer action dim from output shape (usually [1, action_dim])
        size_t action_dim = 0;
        if (output_shape.size() >= 2) {
            action_dim = static_cast<size_t>(output_shape[1]);
        } else if (!output_shape.empty()) {
            action_dim = static_cast<size_t>(output_shape.back());
        }

        if (action_dim == 0) {
            // Fallback: safe default; will resize dynamically after first Run()
            action_dim = 1;
        }

        action.resize(action_dim, 0.0f);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Ensure all required input names exist
        for (const auto& name_c : input_names) {
            const std::string name(name_c);
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name '" + name + "' not found in observations.");
            }
        }

        // Validate observations (NaN/Inf guard)
        bool obs_valid = true;
        for (const auto& name_c : input_names) {
            const std::string name(name_c);
            auto& input_data = obs.at(name);
            for (size_t i = 0; i < input_data.size(); ++i) {
                if (!std::isfinite(input_data[i])) {
                    spdlog::error("Invalid observation['{}'][{}] = {} (NaN/Inf)!", name, i, input_data[i]);
                    obs_valid = false;
                }
            }
        }

        // If invalid obs, return zero actions (safe behavior)
        if (!obs_valid) {
            spdlog::error("Invalid observations detected! Returning zero actions.");
            std::lock_guard<std::mutex> lock(act_mtx_);
            std::fill(action.begin(), action.end(), 0.0f);
            return action;
        }

        // IMPORTANT:
        // We must keep any padded/truncated buffers alive until AFTER session->Run completes.
        // Otherwise input tensors will point to freed memory -> haywire behavior.
        std::vector<std::vector<float>> owned_buffers;
        owned_buffers.reserve(input_names.size());

        // ONNX Input Debug (PART 8)
        if (isaaclab::debug::is_debug_enabled()) {
            static int onnx_call_count = 0;
            if (onnx_call_count++ % 50 == 0) {
                spdlog::info("[DEBUG] ========== ONNX INPUT DEBUG ==========");
                for (size_t i = 0; i < input_names.size(); ++i) {
                    const std::string name(input_names[i]);
                    auto& input_data = obs.at(name);
                    spdlog::info("[DEBUG] ONNX input[{}] '{}': expected={}, got={}", 
                                i, name, input_sizes[i], input_data.size());
                }
                spdlog::info("[DEBUG] ======================================");
            }
        }

        // Build input tensors
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(input_names.size());

        for (size_t i = 0; i < input_names.size(); ++i) {
            const std::string name(input_names[i]);
            auto& input_data = obs.at(name);

            const size_t expected = input_sizes[i];
            const size_t got = input_data.size();

            if (got != expected) {
                spdlog::error("[DEBUG] CRITICAL: Observation['{}'] size mismatch: got {}, expected {}. DO NOT PAD/TRUNCATE!", 
                    name, got, expected);
                spdlog::error("[DEBUG] This indicates a serious bug in observation assembly!");
                // Hard assert - do not silently pad
                throw std::runtime_error("Observation size mismatch: " + name + " got " + std::to_string(got) + 
                                        " expected " + std::to_string(expected));

                // Allocate owned buffer of correct size and fill with zeros
                owned_buffers.emplace_back(expected, 0.0f);
                auto& buf = owned_buffers.back();

                const size_t copy_n = std::min(got, expected);
                std::copy(input_data.begin(), input_data.begin() + copy_n, buf.begin());

                input_tensors.emplace_back(
                    Ort::Value::CreateTensor<float>(
                        memory_info,
                        buf.data(),
                        buf.size(),
                        input_shapes[i].data(),
                        input_shapes[i].size()
                    )
                );
            } else {
                // Use input_data directly (no extra allocation)
                input_tensors.emplace_back(
                    Ort::Value::CreateTensor<float>(
                        memory_info,
                        input_data.data(),
                        expected,
                        input_shapes[i].data(),
                        input_shapes[i].size()
                    )
                );
            }
        }

        // Dump observations if enabled (PART 9)
        if (isaaclab::debug::is_dump_enabled()) {
            int step = isaaclab::debug::get_current_step();
            if (step <= 200) {  // Only dump first 200 steps
                std::string dump_dir = isaaclab::debug::get_dump_dir();
                for (const auto& name_c : input_names) {
                    const std::string name(name_c);
                    auto& input_data = obs.at(name);
                    std::string filepath = dump_dir + "/obs_" + name + "_step_" + 
                                         std::to_string(step) + ".csv";
                    isaaclab::debug::dump_csv(filepath, input_data);
                }
            }
        }

        // Run the model
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );

        if (outputs.empty()) {
            spdlog::error("ONNXRuntime returned no outputs! Returning zero actions.");
            std::lock_guard<std::mutex> lock(act_mtx_);
            std::fill(action.begin(), action.end(), 0.0f);
            return action;
        }

        // Read output tensor and resize action dynamically if needed
        auto& out0 = outputs.front();
        auto out_shape = out0.GetTensorTypeAndShapeInfo().GetShape();

        // Infer actual output length (numel)
        size_t out_numel = 1;
        for (auto d : out_shape) {
            if (d < 0) d = 1;
            out_numel *= static_cast<size_t>(d);
        }

        float* out_ptr = out0.GetTensorMutableData<float>();

        {
            std::lock_guard<std::mutex> lock(act_mtx_);
            if (action.size() != out_numel) {
                action.resize(out_numel, 0.0f);
            }
            std::memcpy(action.data(), out_ptr, action.size() * sizeof(float));
        }

        // ONNX Output Debug (PART 8)
        if (isaaclab::debug::is_debug_enabled()) {
            static int output_debug_count = 0;
            if (output_debug_count++ % 50 == 0) {
                std::lock_guard<std::mutex> lock(act_mtx_);
                spdlog::info("[DEBUG] ========== ONNX OUTPUT DEBUG ==========");
                spdlog::info("[DEBUG] Output shape: [{}]", out_numel);
                isaaclab::debug::print_stats(action, "action_raw");
                isaaclab::debug::print_first(action, "action_raw", 10);
                size_t sat_count = isaaclab::debug::count_saturation(action, 0.95f);
                spdlog::info("[DEBUG] Action saturation count (|a|>0.95) = {}/{}", sat_count, action.size());
                if (sat_count > action.size() * 0.3f) {
                    spdlog::warn("[DEBUG] WARNING: >30% saturated during standing -> normalization/scale/order mismatch!");
                }
                spdlog::info("[DEBUG] ========================================");
            }
        }

        // Dump actions if enabled (PART 9)
        if (isaaclab::debug::is_dump_enabled()) {
            int step = isaaclab::debug::get_current_step();
            if (step <= 200) {
                std::lock_guard<std::mutex> lock(act_mtx_);
                std::string dump_dir = isaaclab::debug::get_dump_dir();
                std::string filepath = dump_dir + "/act_step_" + std::to_string(step) + ".csv";
                isaaclab::debug::dump_csv(filepath, action);
            }
        }

        // Validate action outputs
        bool action_valid = true;
        {
            std::lock_guard<std::mutex> lock(act_mtx_);
            for (size_t i = 0; i < action.size(); ++i) {
                if (!std::isfinite(action[i])) {
                    spdlog::error("[DEBUG] CRITICAL: Policy output NaN/Inf at action[{}] = {}. Setting to zero.", i, action[i]);
                    action[i] = 0.0f;
                    action_valid = false;
                }
            }
        }

        if (!action_valid) {
            spdlog::error("[DEBUG] CRITICAL: Policy produced invalid outputs! Actions sanitized.");
            if (isaaclab::debug::is_dump_enabled()) {
                // Dump obs for debugging
                int step = isaaclab::debug::get_current_step();
                std::string dump_dir = isaaclab::debug::get_dump_dir();
                for (const auto& name_c : input_names) {
                    const std::string name(name_c);
                    auto& input_data = obs.at(name);
                    std::string filepath = dump_dir + "/obs_" + name + "_NAN_step_" + 
                                         std::to_string(step) + ".csv";
                    isaaclab::debug::dump_csv(filepath, input_data);
                }
            }
        }

        return get_action();
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    // Input/Output names stored safely
    std::vector<std::string> input_names_str;
    std::vector<const char*> input_names;

    std::vector<std::string> output_names_str;
    std::vector<const char*> output_names;

    // Shapes + sizes
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<size_t> input_sizes;

    std::vector<int64_t> output_shape;
};

} // namespace isaaclab
