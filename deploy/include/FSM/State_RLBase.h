// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // Set PD gains. Note: stiffness/damping in deploy.yaml are in SDK order
        // (the export script does: stiffness[joint_ids_map] = internal_stiffness),
        // so we index them by the SDK motor index, not by policy index.
        for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i)
        {
            int motor_idx = env->robot->data.joint_ids_map[i];
            lowcmd->msg_.motor_cmd()[motor_idx].kp() = env->robot->data.joint_stiffness[motor_idx];
            lowcmd->msg_.motor_cmd()[motor_idx].kd() = env->robot->data.joint_damping[motor_idx];
            lowcmd->msg_.motor_cmd()[motor_idx].dq() = 0;
            lowcmd->msg_.motor_cmd()[motor_idx].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
