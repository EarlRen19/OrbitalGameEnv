//
// Created by baoyicui on 2/22/26.
//

#ifndef ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H
#define ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H


#include "oge/environment/oge_state.h"
#include "oge/environment/oge_settings.h"

#include <string>
#include <vector>
#include <random>
#include <unordered_map>


namespace oge
{
    class OrbitalGameEnvironment
    {
    public:
        explicit OrbitalGameEnvironment(const OGESettings& settings_);

        /** Reset the environment to its start state. */
        void reset();

        void act(const std::vector<Eigen::Vector3d>& agents_actions);

        bool isTerminal() const;

        bool isTruncated() const;

        void getObservations(std::vector<Eigen::VectorXd>& observations) const;
        void getRewards(const std::vector<Eigen::Vector3d>& agent_actions, std::vector<double>& rewards) const;

        /** Returns the observation vector size for agent at index agent_idx. */
        int getObsSize(int agent_idx) const;

        double getCurrentTime() const;

        double getSolarIlluminationAngle() const;

        /** Returns true if blue_sat has been captured (is_alive == false). */
        bool isCaptured() const;

        /** Returns a map from agent_id to SatState for all satellites. */
        std::unordered_map<std::string, SatState> getSatStates() const;

        /** Resets the environment using the provided states map (agent_id -> SatState). */
        void resetWithStates(const std::unordered_map<std::string, SatState>& states);

        static bool almost_equal(double a, double b, double epsilon = 1e-12)
        {
            return std::abs(a - b) < epsilon;
        }

    private:
        void processDynamics(const std::vector<Eigen::Vector3d>& actions);
        void checkAlive();

        double getFormationReward() const;
        double getDistanceRewardNew(int red_idx) const;
        double getDistanceReward(int red_idx) const;
        double getCaptureReward(int red_idx) const;
        double getFuelReward(int red_idx, const Eigen::Vector3d& action) const;
        double getTimeReward() const;

    private:
        const OGESettings& settings;

        /** Settings cache */

        const int random_seed;
        static constexpr int num_agents = 2; // 1 blue_sat + 1 red_sat
        // simulation settings
        const double dv_init_red;
        const double dv_init_blue;
        const double dv_max_per_step_red;
        const double dv_max_per_step_blue;
        const double capture_distance;
        const double timestep;
        const double terminal_time;
        // random initialization settings
        const double sma_perturb_max;
        const double dist_init_offset_max;
        const double dist_init_offset_min;
        // reward settings
        const double reward_time_weight;
        const double reward_formation_weight;
        const double reward_fuel_weight;
        const double reward_capture_weight;
        const double reward_timeout_weight;
        const double reward_fuelout_weight;
        const double reward_phase_dist_weight;
        // distance reward sub-parameters
        const double reward_far_sma_penalty_scale;
        const double reward_far_drift_scale;
        const double reward_far_drift_max;
        const double reward_far_angle_weight;
        const double reward_near_energy_scale;
        const double reward_near_energy_weight;
        const double reward_dist_capture_bonus;
        const double reward_dist_min;
        const double reward_alpha_scale;
        /** Settings cache end */

        std::vector<std::string> agent_ids;
        // agents_states[0] = blue_sat, agents_states[1] = red_sat
        std::vector<SatState> agents_states;

        double current_time; // s

        // random generator
        std::mt19937 _rng;
        std::uniform_real_distribution<double> sma_perturb_distrib;
        std::uniform_real_distribution<double> true_anomaly_distrib;
        std::uniform_real_distribution<double> dist_init_offset_distrib;
        std::uniform_int_distribution<int> TA_lead_distrib;
    };
}


#endif //ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H
