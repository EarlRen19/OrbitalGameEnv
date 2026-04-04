// Created for multi-agent pursuit-evasion extension.
// Independent from OrbitalGameEnvironment — does NOT modify any existing files.
//
// Agent ordering:
//   [0 .. num_evaders-1]          blue team: evaders (index 0 = HVT, 1+ = interceptors)
//   [num_evaders .. num_agents-1]  red  team: pursuers
//
// Observation layout (same size for every agent, 13 + 7*(N-1) dims):
//   [0:3]                    own r_j2000 (km)
//   [3:6]                    own v_j2000 (km/s)
//   for each other agent j in order (N-1 blocks of 7):
//     [6+7*j : 6+7*j+3]      rel_pos in own LVLH (km)
//     [6+7*j+3 : 6+7*j+6]    rel_vel in own LVLH (m/s)
//     [6+7*j+6]               dist(i,j) / 20km
//   [6+7*(N-1)+0]            solar_angle w.r.t. HVT (rad)
//   [6+7*(N-1)+1]            dv_remain (km/s)
//   [6+7*(N-1)+2]            time_progress [0,1]
//   [6+7*(N-1)+3]            dv_ratio (dv_remain / dv_init)
//   [6+7*(N-1)+4 : +7]       sun_dir in HVT LVLH (unit vector)
// Total = 6 + 7*(N-1) + 7 = 13 + 7*(N-1)
// For N=2: 20 — identical to OrbitalGameEnvironment (backward compatible).

#pragma once

#include "oge/environment/oge_state.h"
#include "oge/environment/oge_settings.h"

#include <Eigen/Dense>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace oge
{

class MultiAgentOGE
{
public:
    /**
     * @param settings          Existing OGESettings (orbital + simulation params, same keys as before)
     * @param num_evaders       Blue satellites: index 0 = HVT, 1+ = interceptors
     * @param num_pursuers      Red satellites: all are pursuers
     * @param intercept_distance  km — distance at which interceptor captures a pursuer
     */
    explicit MultiAgentOGE(const OGESettings& settings,
                           int    num_evaders          = 1,
                           int    num_pursuers         = 1,
                           double intercept_distance   = 30.0);

    /** Reset to random initial conditions. */
    void reset();

    /** Reset using externally provided states (agent_id -> SatState). */
    void resetWithStates(const std::unordered_map<std::string, SatState>& states);

    /** Step: apply actions (LVLH delta-v, km/s) for all agents, propagate orbits. */
    void act(const std::vector<Eigen::Vector3d>& actions);

    // ── Termination ─────────────────────────────────────────────────────────
    bool isTerminal()  const;  // any agent no longer alive
    bool isTruncated() const;  // time >= terminal_time
    bool isHVTCaptured()          const;  // HVT (blue[0]) within capture_distance of any pursuer
    bool isPursuerIntercepted()   const;  // any pursuer within intercept_distance of any interceptor

    // ── Observations ────────────────────────────────────────────────────────
    void getObservations(std::vector<Eigen::VectorXd>& observations) const;

    /** Returns 13 + 7*(num_agents-1).  Same for all agents. */
    int getObsSize() const { return 13 + 7 * (num_agents - 1); }

    // ── Accessors ───────────────────────────────────────────────────────────
    int    getNumAgents()   const { return num_agents;   }
    int    getNumEvaders()  const { return num_evaders;  }
    int    getNumPursuers() const { return num_pursuers; }
    double getCurrentTime() const { return current_time; }

    std::unordered_map<std::string, SatState> getSatStates() const;

private:
    void processDynamics(const std::vector<Eigen::Vector3d>& actions);
    void checkAlive();
    Eigen::Vector3d computeSunPosition() const;

    // ── Settings reference ───────────────────────────────────────────────────
    const OGESettings& settings;

    const int    num_evaders;
    const int    num_pursuers;
    const int    num_agents;
    const double intercept_distance;

    // ── Cached settings ──────────────────────────────────────────────────────
    const double dv_init_evader;
    const double dv_init_pursuer;
    const double dv_max_per_step_evader;
    const double dv_max_per_step_pursuer;
    const double capture_distance;    // km — HVT capture by pursuer
    const double timestep;            // s
    const double terminal_time;       // s
    const double sma_perturb_max;     // km
    const double dist_init_offset_min;// km
    const double dist_init_offset_max;// km

    // ── State ────────────────────────────────────────────────────────────────
    std::vector<std::string> agent_ids;     // "blue_sat_0", "red_sat_0", …
    std::vector<SatState>    agents_states; // size = num_agents

    double current_time; // s

    // ── RNG ──────────────────────────────────────────────────────────────────
    std::mt19937                             _rng;
    std::uniform_real_distribution<double>   sma_perturb_distrib;
    std::uniform_real_distribution<double>   true_anomaly_distrib;
    std::uniform_real_distribution<double>   dist_init_offset_distrib;
    std::uniform_int_distribution<int>       ta_lead_distrib;

    // JD epoch — same as OrbitalGameEnvironment (2027-09-01 16:00 UTC)
    static constexpr double JD_EPOCH = 2461650.166667;
};

} // namespace oge
