//
// Created by baoyicui on 2/22/26.
//

#include "oge/simcore/math.h"
#include "oge/simcore/utils.h"

#include "orbital_game_environment.h"

namespace oge
{
    OrbitalGameEnvironment::OrbitalGameEnvironment(
        const OGESettings& settings_
    ) :
        settings(settings_),
        random_seed(settings_.getInt("random_seed", true)),
        // simulation settings
        dv_init_red(settings_.getFloat("dv_init_red")),
        dv_init_blue(settings_.getFloat("dv_init_blue")),
        dv_max_per_step_red(settings_.getFloat("dv_max_per_step_red")),
        dv_max_per_step_blue(settings_.getFloat("dv_max_per_step_blue")),
        capture_distance(settings_.getFloat("capture_distance")),
        timestep(settings_.getFloat("timestep")),
        terminal_time(settings_.getFloat("terminal_time")),
        // random initialization settings
        sma_perturb_max(settings_.getFloat("sma_perturb_max")),
        dist_init_offset_max(settings_.getFloat("dist_init_offset_max")),
        dist_init_offset_min(settings_.getFloat("dist_init_offset_min")),
        // reward settings
        reward_time_weight(settings_.getFloat("reward_time_weight")),
        reward_formation_weight(settings_.getFloat("reward_formation_weight")),
        reward_fuel_weight(settings_.getFloat("reward_fuel_weight")),
        reward_capture_weight(settings_.getFloat("reward_capture_weight")),
        reward_timeout_weight(settings_.getFloat("reward_timeout_weight")),
        reward_fuelout_weight(settings_.getFloat("reward_fuelout_weight")),
        reward_phase_dist_weight(settings_.getFloat("reward_phase_dist_weight")),
        // distance reward sub-parameters
        reward_far_sma_penalty_scale(settings_.getFloat("reward_far_sma_penalty_scale")),
        reward_far_drift_scale(settings_.getFloat("reward_far_drift_scale")),
        reward_far_drift_max(settings_.getFloat("reward_far_drift_max")),
        reward_far_angle_weight(settings_.getFloat("reward_far_angle_weight")),
        reward_near_energy_scale(settings_.getFloat("reward_near_energy_scale")),
        reward_near_energy_weight(settings_.getFloat("reward_near_energy_weight")),
        reward_dist_capture_bonus(settings_.getFloat("reward_dist_capture_bonus")),
        reward_dist_min(settings_.getFloat("reward_dist_min")),
        reward_alpha_scale(settings_.getFloat("reward_alpha_scale"))
    {
        settings.validate();
        /* Initialize random generator */
        _rng.seed(random_seed);
        sma_perturb_distrib = std::uniform_real_distribution<double>(
            -sma_perturb_max,
            sma_perturb_max
        );
        true_anomaly_distrib = std::uniform_real_distribution<double>(
            0.0,
            2 * M_PI
        );
        dist_init_offset_distrib = std::uniform_real_distribution<double>(
            capture_distance + dist_init_offset_min,
            capture_distance + dist_init_offset_max
        );
        TA_lead_distrib = std::uniform_int_distribution<int>(0, 1);

        // initialize agents: agents_states[0] = blue_sat, agents_states[1] = red_sat
        agent_ids = {"blue_sat", "red_sat"};
        agents_states.resize(num_agents);

        current_time = 0.0;
        reset();
    }

    bool OrbitalGameEnvironment::isTerminal() const
    {
        // blue_sat captured
        if (!agents_states[0].is_alive) return true;

        // red_sat fuel exhausted
        if (!agents_states[1].is_alive) return true;

        return false;
    }

    bool OrbitalGameEnvironment::isTruncated() const
    {
        return current_time >= terminal_time;
    }

    int OrbitalGameEnvironment::getObsSize(int agent_idx) const
    {
        // [0:3]   own R J2000 (km)
        // [3:6]   own V J2000 (km/s)
        // [6:9]   target pos in own LVLH (km)
        // [9]     solar_angle (rad)
        // [10]    dv_remain (km/s)
        // [11:14] target rel_vel in own LVLH (m/s)   -- 新增
        // [14]    dist / 20km                         -- 新增
        // [15:18] sun dir in TARGET(blue) LVLH        -- 新增，修正坐标系
        // [18]    time_progress [0,1]                 -- 新增
        // [19]    dv_remain / dv_init (燃料剩余率)    -- 新增
        return 20;
    }

    double OrbitalGameEnvironment::getCurrentTime() const
    {
        return current_time;
    }

    double OrbitalGameEnvironment::getSolarIlluminationAngle() const
    {
        // 初始时间 UTC+8 2027-09-02 00:00:00 = UTC 2027-09-01 16:00:00
        // 对应儒略日 JD = 2461650.166667
        constexpr double JD_EPOCH = 2461650.166667;
        Eigen::Vector3d pos_sun;
        solar_position(JD_EPOCH + current_time / 86400.0, pos_sun);
        return solar_illumination_angle(pos_sun, agents_states[0].r_j2000, agents_states[1].r_j2000);
    }

    void OrbitalGameEnvironment::getObservations(std::vector<Eigen::VectorXd>& observations) const
    {
        observations.resize(num_agents);
        double solar_angle = getSolarIlluminationAngle();

        // 太阳方向（J2000），用于计算LVLH下的太阳方向
        constexpr double JD_EPOCH = 2461650.166667;
        Eigen::Vector3d pos_sun;
        solar_position(JD_EPOCH + current_time / 86400.0, pos_sun);

        // 时间进度 [0, 1]
        double time_progress = current_time / terminal_time;

        // blue_sat (index 0)
        {
            observations[0].resize(getObsSize(0));
            observations[0].segment<3>(0) = agents_states[0].r_j2000;
            observations[0].segment<3>(3) = agents_states[0].v_j2000;
            Eigen::Vector3d r_red_lvlh, v_red_lvlh;
            RV_J20002LVLH(
                agents_states[0].r_j2000, agents_states[0].v_j2000,
                agents_states[1].r_j2000, agents_states[1].v_j2000,
                r_red_lvlh, v_red_lvlh
            );
            observations[0].segment<3>(6) = r_red_lvlh;
            observations[0](9)  = solar_angle;
            observations[0](10) = agents_states[0].dv_remain;
            // 新增
            observations[0].segment<3>(11) = v_red_lvlh * 1000.0; // rel_vel m/s
            observations[0](14) = r_red_lvlh.norm() / 20.0;       // dist / 20km
            Eigen::Matrix3d dcm;
            DCM_J2000_to_LVLH(agents_states[0].r_j2000, agents_states[0].v_j2000, dcm);
            observations[0].segment<3>(15) = dcm * (pos_sun - agents_states[0].r_j2000).normalized();
            observations[0](18) = time_progress;
            observations[0](19) = agents_states[0].dv_remain / dv_init_blue; // 燃料剩余率
        }

        // red_sat (index 1)
        {
            observations[1].resize(getObsSize(1));
            observations[1].segment<3>(0) = agents_states[1].r_j2000;
            observations[1].segment<3>(3) = agents_states[1].v_j2000;
            Eigen::Vector3d r_blue_lvlh, v_blue_lvlh;
            RV_J20002LVLH(
                agents_states[1].r_j2000, agents_states[1].v_j2000,
                agents_states[0].r_j2000, agents_states[0].v_j2000,
                r_blue_lvlh, v_blue_lvlh
            );
            observations[1].segment<3>(6) = r_blue_lvlh;
            observations[1](9)  = solar_angle;
            observations[1](10) = agents_states[1].dv_remain;
            // 新增
            observations[1].segment<3>(11) = v_blue_lvlh * 1000.0; // rel_vel m/s
            observations[1](14) = r_blue_lvlh.norm() / 20.0;       // dist / 20km
            // 太阳方向在 blue(目标) LVLH 下表示，与光照角定义一致
            Eigen::Matrix3d dcm_blue;
            DCM_J2000_to_LVLH(agents_states[0].r_j2000, agents_states[0].v_j2000, dcm_blue);
            observations[1].segment<3>(15) = dcm_blue * (pos_sun - agents_states[0].r_j2000).normalized();
            observations[1](18) = time_progress;
            observations[1](19) = agents_states[1].dv_remain / dv_init_red; // 燃料剩余率
        }
    }

    void OrbitalGameEnvironment::getRewards(const std::vector<Eigen::Vector3d>& agent_actions,
                                            std::vector<double>& rewards) const
    {
        rewards.assign(num_agents, 0.0);
        // TODO: blue_sat's reward

        // red_sat reward (index 1)
        rewards[1] += getFormationReward();
        rewards[1] += getDistanceRewardNew(1);
        rewards[1] += getCaptureReward(1);
        rewards[1] += getFuelReward(1, agent_actions[1]);
        rewards[1] += getTimeReward();
    }


    void OrbitalGameEnvironment::reset()
    {
        current_time = 0.0;
        // initialize blue_sat's state
        const Eigen::Matrix<double, 6, 1> coe_base(
            settings.getFloat("sma_base", true),
            settings.getFloat("ecc_base", true),
            settings.getFloat("incl_base", true),
            settings.getFloat("RA_base", true),
            settings.getFloat("w_base", true),
            settings.getFloat("TA_base", true));
        Eigen::Matrix<double, 6, 1> coe_blue = coe_base;
        coe_blue[0] += sma_perturb_distrib(_rng);
        coe_blue[5] = true_anomaly_distrib(_rng);
        coe2rv(coe_blue, agents_states[0].r_j2000, agents_states[0].v_j2000);

        // initialize red_sat's state
        {
            Eigen::Matrix<double, 6, 1> coe_red = coe_base;
            double TA_lead = TA_lead_distrib(_rng) == 0 ? -1.0 : 1.0; // 相位超前还是滞后
            double distance_offset = dist_init_offset_distrib(_rng);
            coe_red[0] += sma_perturb_distrib(_rng);
            coe_red[5] = coe_blue[5] + TA_lead * distance_offset / coe_red[0]; // 基于blue_sat的TA加偏移
            coe2rv(coe_red, agents_states[1].r_j2000, agents_states[1].v_j2000);
        }

        // make every agent alive and reset fuel
        agents_states[0].is_alive = true;
        agents_states[0].dv_remain = dv_init_blue;
        agents_states[1].is_alive = true;
        agents_states[1].dv_remain = dv_init_red;
    }

    std::unordered_map<std::string, SatState> OrbitalGameEnvironment::getSatStates() const
    {
        std::unordered_map<std::string, SatState> result;
        for (int i = 0; i < num_agents; ++i)
        {
            result[agent_ids[i]] = agents_states[i];
        }
        return result;
    }

    void OrbitalGameEnvironment::resetWithStates(const std::unordered_map<std::string, SatState>& states)
    {
        current_time = 0.0;
        for (int i = 0; i < num_agents; ++i)
        {
            auto it = states.find(agent_ids[i]);
            if (it != states.end())
            {
                agents_states[i] = it->second;
            }
        }
    }

    void OrbitalGameEnvironment::processDynamics(const std::vector<Eigen::Vector3d>& actions)
    {
        if (actions.size() != static_cast<size_t>(num_agents))
        {
            throw std::invalid_argument("actions.size() != num_agents");
        }

        for (int i = 0; i < num_agents; ++i)
        {
            // Apply thrust only when fuel remains; propagate orbit for all agents regardless.
            Eigen::Vector3d dv_modified = Eigen::Vector3d::Zero();
            if (agents_states[i].dv_remain > 0.0 && !almost_equal(actions[i].norm(), 0.0))
            {
                // index 0 = blue_sat, index 1 = red_sat
                double dv_max_per_step = (i == 0) ? dv_max_per_step_blue : dv_max_per_step_red;
                if (actions[i].norm() > std::min(dv_max_per_step, agents_states[i].dv_remain))
                {
                    dv_modified = std::min(dv_max_per_step, agents_states[i].dv_remain) * actions[i].normalized();
                }
                else
                {
                    dv_modified = actions[i];
                }
            }
            // 这里传入的动作是 LVLH坐标系下的，需要转换回J2000坐标系再更新 agents_states[i].v_j2000
            Eigen::Vector3d r_j2000, v_j2000;
            RV_LVLH2J2000(
                agents_states[i].r_j2000, agents_states[i].v_j2000,
                Eigen::Vector3d::Zero(), dv_modified,
                r_j2000, v_j2000
            );

            // update agent's velocity in J2000
            agents_states[i].v_j2000 = v_j2000;
            // update agent's fuel
            agents_states[i].dv_remain -= dv_modified.norm();

            // propagation
            Eigen::Vector3d r_j2000_new, v_j2000_new;
            rv_from_r0v0(agents_states[i].r_j2000, agents_states[i].v_j2000, timestep, r_j2000_new, v_j2000_new);
            agents_states[i].r_j2000 = r_j2000_new;
            agents_states[i].v_j2000 = v_j2000_new;
        }
        current_time += timestep;
    }

    void OrbitalGameEnvironment::checkAlive()
    {
        // check if collision (distance < 3km)
        if ((agents_states[0].r_j2000 - agents_states[1].r_j2000).norm() < 3.0)
        {
            agents_states[0].is_alive = false;
        }

        // 不因燃料耗尽终止，只在奖励中惩罚
    }

    double OrbitalGameEnvironment::getFormationReward() const
    {
        // Formation reward only makes sense when there are multiple red_sats.
        // With a single red_sat, this reward is always 0.
        return 0.0;
    }

    double OrbitalGameEnvironment::getDistanceRewardNew(int red_idx) const
    {
        const double distance = (agents_states[red_idx].r_j2000 - agents_states[0].r_j2000).norm();
        return -reward_phase_dist_weight * (distance - capture_distance) / capture_distance;
    }

    double OrbitalGameEnvironment::getDistanceReward(int red_idx) const
    {
        double distance = (agents_states[red_idx].r_j2000 - agents_states[0].r_j2000).norm();

        Eigen::Matrix<double, 6, 1> coe_red, coe_blue;
        rv2coe(agents_states[red_idx].r_j2000, agents_states[red_idx].v_j2000, coe_red);
        rv2coe(agents_states[0].r_j2000, agents_states[0].v_j2000, coe_blue);

        double TA_delta = std::fmod((coe_red - coe_blue)(5) + M_PI, 2.0 * M_PI) - M_PI;
        double sma_diff_ratio = (coe_red - coe_blue)(0) / coe_blue(0);

        // Far field
        double drift_product = TA_delta * sma_diff_ratio;
        double reward_far;
        if (drift_product < 0.0)
        {
            reward_far = -1.0 - std::abs(sma_diff_ratio) * reward_far_sma_penalty_scale;
        }
        else
        {
            double r_drift = std::clamp(std::abs(sma_diff_ratio) * reward_far_drift_scale, 0.0, reward_far_drift_max);
            double r_angle = (M_PI - std::abs(TA_delta)) / M_PI;
            reward_far = 1.0 * r_drift + reward_far_angle_weight * r_angle;
        }

        double dist_normalized = distance / capture_distance;
        double reward_dist = 0.0;
        if (dist_normalized <= 1.0)
        {
            reward_dist = 1.0 + reward_dist_capture_bonus * (1.0 - dist_normalized);
        }
        else if (dist_normalized <= 2.0)
        {
            reward_dist = 2.0 - dist_normalized;
        }
        else
        {
            reward_dist = std::clamp(2.0 - dist_normalized, reward_dist_min, 0.0);
        }

        double reward_energy = -std::abs(sma_diff_ratio) * reward_near_energy_scale;
        double reward_near = 1.0 * reward_dist + reward_near_energy_weight * reward_energy;

        double alpha = std::clamp(std::abs(sma_diff_ratio) * reward_alpha_scale, 0.0, 1.0);
        double total_reward = alpha * reward_far + (1.0 - alpha) * reward_near;


        return reward_phase_dist_weight * total_reward;
    }

    double OrbitalGameEnvironment::getCaptureReward(int red_idx) const
    {
        if ((agents_states[red_idx].r_j2000 - agents_states[0].r_j2000).norm() < capture_distance)
        {
            return reward_capture_weight; // capture bonus
        }

        return 0.0; // no capture
    }

    double OrbitalGameEnvironment::getFuelReward(const int red_idx, const Eigen::Vector3d& action) const
    {
        double fuel_used = action.norm();
        fuel_used = std::min(std::min(fuel_used, agents_states[red_idx].dv_remain), dv_max_per_step_red);

        return reward_fuel_weight * fuel_used;
    }

    double OrbitalGameEnvironment::getTimeReward() const
    {
        return reward_time_weight;
    }

    bool OrbitalGameEnvironment::isCaptured() const
    {
        return !agents_states[0].is_alive;
    }

    void OrbitalGameEnvironment::act(const std::vector<Eigen::Vector3d>& agents_actions)
    {
        if (agents_actions.size() != static_cast<size_t>(num_agents))
        {
            throw std::invalid_argument("agents_actions.size() != num_agents");
        }
        processDynamics(agents_actions);
        checkAlive();
    }
}
