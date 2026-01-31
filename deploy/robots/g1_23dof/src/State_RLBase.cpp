#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>
#include <algorithm>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];
       
    static std::string last_logged_key = "";
    if(key != last_logged_key && !key.empty()) {
        spdlog::info("Key detected: '{}' -> Command will be generated", key);
        last_logged_key = key;
    }

    // Optimized keyboard values based on curriculum training analysis
    // Forward/backward: policy generalizes well beyond training range (trained 0.1, works at 0.4)
    // Lateral/turning: limited to 50% of training max (NO curriculum, stayed at 0.1 entire training)
    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {0.4f, 0.0f, 0.0f}},    // Walk forward - generalizes well
        {"s", {-0.3f, 0.0f, 0.0f}},   // Walk backward - generalizes well  
        {"a", {0.0f, 0.05f, 0.0f}},   // Strafe left (50% of training max)
        {"d", {0.0f, -0.05f, 0.0f}},  // Strafe right (50% of training max)
        {"q", {0.0f, 0.0f, 0.05f}},   // Turn left (50% max - CRITICAL: no ang curriculum)
        {"e", {0.0f, 0.0f, -0.05f}}   // Turn right (50% max - CRITICAL: no ang curriculum)
    };
    
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

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(deploy_yaml),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
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
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
