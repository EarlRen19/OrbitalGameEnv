// Created for multi-agent pursuit-evasion extension.
// Independent from OrbitalGameEnvironment — does NOT modify any existing files.
//
// Agent ordering:
//   [0 .. num_evaders-1]          blue team: evaders (index 0 = HVT, 1+ = interceptors)
//   [num_evaders .. num_agents-1]  red  team: pursuers
//
// Full observation layout (same size for every agent, 13 + 7*(N-1) dims):
//   [0:3]                    own r_j2000 (km)
//   [3:6]                    own v_j2000 (km/s)
//   for each other agent j in order (N-1 blocks of 7):
//     [6+7*j : 6+7*j+3]      rel_pos in own LVLH (km)
//     [6+7*j+3 : 6+7*j+6]    rel_vel in own LVLH (m/s)
//     [6+7*j+6]               dist(i,j) / 20km
//   tail (7 scalars): solar_angle, dv_remain, time_progress, dv_ratio, sun_dir(3)
//
// Task-specific observation layout (17 dims, via getTaskObservations):
//   [0:3]   rel_pos to task-target in own LVLH / 200km
//   [3:6]   rel_vel to task-target in own LVLH * 10  (m/s)
//   [6]     dist_to_target / 20km
//   [7]     task angle / pi:
//             STRIKE / RECON  → solar_illumination_angle (vertex = target)
//             JAM             → jamming_angle            (vertex = target)
//             OPERATE         → relative speed * 10 (m/s), no angle
//   [8:11]  auxiliary direction (unit vector) in target LVLH:
//             STRIKE / RECON  → sun direction
//             JAM             → target-to-earth direction
//             OPERATE         → zeros
//   [11]    dv_ratio (dv_remain / dv_init)
//   [12]    time_progress [0,1]
//   [13]    dist_to_threat / 20km
//   [14:17] rel_pos_to_threat in own LVLH / 200km

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

// ── Task type enum ────────────────────────────────────────────────────────────
enum class TaskType : int
{
    STRIKE  = 0,   // 打击：solar_angle (vertex=target), threshold 90°, 40s
    RECON   = 1,   // 侦照：solar_angle (vertex=target), threshold 60°, 120s
    JAM     = 2,   // 干扰：jamming_angle (vertex=target), threshold 5°, 600s
    OPERATE = 3,   // 操控：relative speed constraint, dist ≤ 2km
};

// ── Per-agent task assignment ─────────────────────────────────────────────────
struct AgentTask
{
    TaskType task_type  = TaskType::RECON;
    int      target_idx = 0;   // global agent index of the task target
    int      threat_idx = -1;  // global agent index of the main threat (-1 = none)
};

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
    /** Full observation: 13 + 7*(N-1) dims per agent. Kept for compatibility. */
    void getObservations(std::vector<Eigen::VectorXd>& observations) const;

    /** Returns 13 + 7*(num_agents-1).  Same for all agents. */
    int getObsSize() const { return 13 + 7 * (num_agents - 1); }

    /**
     * Set per-agent task assignments. Must be called before getTaskObservations.
     * @param assignments  vector of size num_agents; index matches agent global index.
     *                     Agents with no meaningful task (e.g. HVT) can use default.
     */
    void setTaskAssignment(const std::vector<AgentTask>& assignments);

    /**
     * Task-specific 17-dim observations. Requires setTaskAssignment to have been called.
     * Each agent's obs is computed using its assigned target and threat indices,
     * with the correct angle type for its task.
     * @param observations  output, size = num_agents, each vector is 17 dims.
     */
    void getTaskObservations(std::vector<Eigen::VectorXd>& observations) const;

    /** Fixed size for task observations. */
    static constexpr int TASK_OBS_SIZE = 17;

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

    // ── Task assignment (set via setTaskAssignment) ───────────────────────────
    std::vector<AgentTask> task_assignments_; // size = num_agents, default-initialised

    // ── RNG ──────────────────────────────────────────────────────────────────
    std::mt19937                             _rng;
    std::uniform_real_distribution<double>   sma_perturb_distrib;
    std::uniform_real_distribution<double>   true_anomaly_distrib;
    std::uniform_real_distribution<double>   dist_init_offset_distrib;
    std::uniform_int_distribution<int>       ta_lead_distrib;

    // JD epoch — default: 2027-09-01 16:00 UTC (same as OrbitalGameEnvironment).
    // Can be overridden via settings key "jd_epoch" (double, positive value).
    static constexpr double JD_EPOCH_DEFAULT = 2461650.166667;
    double jd_epoch_;   // runtime value, initialised in constructor
};

} // namespace oge
