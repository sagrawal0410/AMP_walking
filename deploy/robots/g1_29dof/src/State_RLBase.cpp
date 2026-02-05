#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];
    
    // Key command mappings - must be defined BEFORE use
    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        // Letter keys
        {"w", {0.4f, 0.0f, 0.0f}},    // Walk forward
        {"s", {-0.3f, 0.0f, 0.0f}},   // Walk backward
        {"a", {0.0f, 0.25f, 0.0f}},   // Strafe left
        {"d", {0.0f, -0.25f, 0.0f}},  // Strafe right
        {"q", {0.0f, 0.0f, 0.5f}},    // Turn left
        {"e", {0.0f, 0.0f, -0.5f}},   // Turn right
        // Arrow keys (if keyboard returns these)
        {"up", {0.4f, 0.0f, 0.0f}},
        {"down", {-0.3f, 0.0f, 0.0f}},
        {"left", {0.0f, 0.25f, 0.0f}},
        {"right", {0.0f, -0.25f, 0.0f}},
        // Space to stop
        {"space", {0.0f, 0.0f, 0.0f}}
    };
    
    // Log key presses for debugging
    static std::string last_logged_key = "";
    if(key != last_logged_key && !key.empty()) {
        bool key_found = key_commands.find(key) != key_commands.end();
        if (key_found) {
            spdlog::info("Key detected: '{}' -> Command will be updated", key);
        } else {
            spdlog::warn("Key detected: '{}' -> NOT IN KEY MAP! Available: w,s,a,d,q,e,up,down,left,right,space", key);
        }
        last_logged_key = key;
    }
    
    // Maintain last command state (static) to avoid jumping to zero when no key is pressed
    // This matches training behavior where commands persist until changed
    static std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    static std::string last_processed_key = "";
    
    // Only update command when a NEW valid key is pressed (not on every call)
    // This ensures consistency when observation is called multiple times per step
    if (!key.empty() && key != last_processed_key && key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
        last_processed_key = key;
        spdlog::info("Command updated: [{:.3f}, {:.3f}, {:.3f}]", cmd[0], cmd[1], cmd[2]);
    }
    else if (key.empty())
    {
        // When no key is pressed, clear the last processed key but keep the command
        // This allows the same key to be processed again if pressed later
        last_processed_key = "";
    }
    // If no key pressed or same key, cmd retains its previous value (don't reset to zero)
    
    // Clamp to training ranges (matching velocity_commands behavior)
    cmd[0] = std::clamp(cmd[0], cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    cmd[1] = std::clamp(cmd[1], cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    cmd[2] = std::clamp(cmd[2], cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
    
    // Debug instrumentation
    if (isaaclab::debug::is_debug_enabled()) {
        static int call_count = 0;
        if (call_count++ % 50 == 0) {
            isaaclab::debug::print_stats(cmd, "keyboard_velocity_commands");
            isaaclab::debug::print_first(cmd, "keyboard_velocity_commands", 3);
            isaaclab::debug::check_finite(cmd, "keyboard_velocity_commands");
            spdlog::info("[DEBUG] keyboard_velocity_commands: [vx={:.4f}, vy={:.4f}, yaw_rate={:.4f}]", 
                        cmd[0], cmd[1], cmd[2]);
            spdlog::info("[DEBUG] keyboard_velocity_commands: ranges should match training: lin_vel_x[-0.5,3.0], lin_vel_y[-0.5,0.5], ang_vel_z[-1.0,1.0]");
        }
    }
    
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    spdlog::info("========================================");
    spdlog::info("Loading RL Policy from:");
    spdlog::info("  Policy Directory: {}", policy_dir.string());
    
    auto deploy_yaml = policy_dir / "params" / "deploy.yaml";
    auto policy_onnx = policy_dir / "exported" / "policy.onnx";
    
    spdlog::info("  Deploy Config: {}", deploy_yaml.string());
    spdlog::info("  Policy ONNX: {}", policy_onnx.string());
    
    // Check if files exist
    if(!std::filesystem::exists(deploy_yaml)) {
        spdlog::critical("Deploy YAML not found: {}", deploy_yaml.string());
        throw std::runtime_error("Deploy YAML file missing!");
    }
    if(!std::filesystem::exists(policy_onnx)) {
        spdlog::critical("Policy ONNX not found: {}", policy_onnx.string());
        throw std::runtime_error("Policy ONNX file missing!");
    }
    
    // Log file sizes and timestamps
    auto onnx_size = std::filesystem::file_size(policy_onnx);
    auto onnx_time = std::filesystem::last_write_time(policy_onnx);
    spdlog::info("  ONNX File Size: {} bytes ({:.2f} MB)", onnx_size, onnx_size / (1024.0 * 1024.0));
    spdlog::info("========================================");

    YAML::Node deploy_cfg;
    try {
        spdlog::info("Loading deploy.yaml from: {}", deploy_yaml.string());
        deploy_cfg = YAML::LoadFile(deploy_yaml.string());
        spdlog::info("YAML file loaded successfully");
    } catch (const YAML::BadFile& e) {
        spdlog::error("Failed to open deploy.yaml file: {}", e.what());
        throw;
    } catch (const YAML::ParserException& e) {
        spdlog::error("YAML parsing error at line {}, column {}: {}", e.mark.line + 1, e.mark.column + 1, e.what());
        throw;
    } catch (const std::exception& e) {
        spdlog::error("Error loading deploy.yaml: {}", e.what());
        throw;
    }

    try {
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
            deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    } catch (const std::exception& e) {
        spdlog::error("Failed to create ManagerBasedRLEnv: {}", e.what());
        throw;
    }
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_onnx);
    
    spdlog::info("Policy loaded successfully!");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    // Action smoothing: 0.0 = no smoothing (responsive), 1.0 = full smoothing (slow)
    // Lower values = faster response, needed for balance recovery
    // REDUCED from 0.5 to 0.2 to allow faster balance corrections
    static const float ACTION_SMOOTHING = 0.2f;
    static std::vector<float> smoothed_action;
    static int warmup_frames = 0;
    static const int WARMUP_FRAMES = 5;  // Skip first few frames for policy to warm up
    
    auto action = env->action_manager->processed_actions();
    
    // Diagnostic: Log action statistics periodically
    static int action_diag_count = 0;
    if (action_diag_count++ % 100 == 0 && !action.empty()) {
        float action_max = *std::max_element(action.begin(), action.end(), 
            [](float a, float b) { return std::abs(a) < std::abs(b); });
        float action_mean = std::accumulate(action.begin(), action.end(), 0.0f) / action.size();
        int action_nonzero = std::count_if(action.begin(), action.end(), 
            [](float a) { return std::abs(a) > 0.01f; });
        spdlog::info("[ACTION DIAG] Processed actions: max_abs={:.4f} rad, mean={:.4f}, nonzero={}/{}", 
                    std::abs(action_max), action_mean, action_nonzero, action.size());
        if (std::abs(action_max) < 0.1f) {
            spdlog::warn("[ACTION DIAG] WARNING: Actions very small (<0.1 rad)! Robot may not move.");
        }
        if (std::abs(action_max) > 2.0f) {
            spdlog::warn("[ACTION DIAG] WARNING: Actions very large (>2.0 rad)! May cause instability.");
        }
    }
    
    // During warmup, hold current positions to let policy stabilize
    if (warmup_frames < WARMUP_FRAMES) {
        warmup_frames++;
        spdlog::info("[WARMUP] Frame {}/{} - holding current positions", warmup_frames, WARMUP_FRAMES);
        // Keep current joint positions during warmup
        for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
            int motor_idx = env->robot->data.joint_ids_map[i];
            lowcmd->msg_.motor_cmd()[motor_idx].q() = lowstate->msg_.motor_state()[motor_idx].q();
        }
        return;
    }
    
    // Initialize smoothed action with CURRENT joint positions (not zeros!)
    // This prevents the initial jump when transitioning to velocity mode
    if (smoothed_action.empty()) {
        smoothed_action.resize(action.size());
        for (size_t i = 0; i < action.size(); ++i) {
            int motor_idx = env->robot->data.joint_ids_map[i];
            // Start from current position, not from action (which may be zero initially)
            smoothed_action[i] = lowstate->msg_.motor_state()[motor_idx].q();
        }
        spdlog::info("[INIT] Smoothed action initialized from current joint positions");
    }
    
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        float action_val = action[i];
        int motor_idx = env->robot->data.joint_ids_map[i];
        
        // Validate action
        if(!std::isfinite(action_val)) {
            spdlog::error("Invalid action[{}]: {} (NaN/Inf detected)! Using current position.", i, action_val);
            action_val = lowstate->msg_.motor_state()[motor_idx].q();
        }
        
        // Apply smoothing: blend between previous smoothed action and new action
        // smoothed = alpha * old + (1-alpha) * new
        if (ACTION_SMOOTHING > 0.0f && i < smoothed_action.size()) {
            smoothed_action[i] = ACTION_SMOOTHING * smoothed_action[i] + (1.0f - ACTION_SMOOTHING) * action_val;
            action_val = smoothed_action[i];
        }
        
        // Clamp to reasonable joint position limits
        action_val = std::clamp(action_val, -3.14f, 3.14f);
        
        lowcmd->msg_.motor_cmd()[motor_idx].q() = action_val;
    }
}
