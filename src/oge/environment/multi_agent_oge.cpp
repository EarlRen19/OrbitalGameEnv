// Multi-agent pursuit-evasion extension.
// Independent from OrbitalGameEnvironment — does NOT modify any existing files.

#include "multi_agent_oge.h"

#include "oge/simcore/utils.h"
#include "oge/simcore/math.h"

#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace oge
{

// ── Helper ──────────────────────────────────────────────────────────────────
static inline bool almost_equal(double a, double b, double tol = 1e-12)
{
    return std::abs(a - b) < tol;
}

// ── Constructor ─────────────────────────────────────────────────────────────

MultiAgentOGE::MultiAgentOGE(
    const OGESettings& settings_,
    int    num_evaders_,
    int    num_pursuers_,
    double intercept_distance_
) :
    settings(settings_),
    num_evaders(num_evaders_),
    num_pursuers(num_pursuers_),
    num_agents(num_evaders_ + num_pursuers_),
    intercept_distance(intercept_distance_),
    // Re-use the same settings keys as OrbitalGameEnvironment
    dv_init_evader(settings_.getFloat("dv_init_blue")),
    dv_init_pursuer(settings_.getFloat("dv_init_red")),
    dv_max_per_step_evader(settings_.getFloat("dv_max_per_step_blue")),
    dv_max_per_step_pursuer(settings_.getFloat("dv_max_per_step_red")),
    capture_distance(settings_.getFloat("capture_distance")),
    timestep(settings_.getFloat("timestep")),
    terminal_time(settings_.getFloat("terminal_time")),
    sma_perturb_max(settings_.getFloat("sma_perturb_max")),
    dist_init_offset_min(settings_.getFloat("dist_init_offset_min")),
    dist_init_offset_max(settings_.getFloat("dist_init_offset_max"))
{
    if (num_evaders < 1)
        throw std::invalid_argument("num_evaders must be >= 1 (index 0 = HVT)");
    if (num_pursuers < 1)
        throw std::invalid_argument("num_pursuers must be >= 1");

    // Build agent ID list: evaders first, then pursuers
    agent_ids.reserve(num_agents);
    for (int i = 0; i < num_evaders; ++i)
        agent_ids.push_back("blue_sat_" + std::to_string(i));
    for (int i = 0; i < num_pursuers; ++i)
        agent_ids.push_back("red_sat_" + std::to_string(i));

    agents_states.resize(num_agents);

    // Seed RNG
    int seed = settings_.getInt("random_seed", true);
    _rng.seed(static_cast<unsigned>(seed));

    // JD epoch: use "jd_epoch" from settings if positive, else fall back to default
    {
        float jd_from_cfg = settings_.getFloat("jd_epoch");  // returns -1 if absent
        jd_epoch_ = (jd_from_cfg > 0.0f) ? static_cast<double>(jd_from_cfg)
                                           : JD_EPOCH_DEFAULT;
    }

    sma_perturb_distrib       = std::uniform_real_distribution<double>(-sma_perturb_max, sma_perturb_max);
    true_anomaly_distrib      = std::uniform_real_distribution<double>(0.0, 2.0 * M_PI);
    dist_init_offset_distrib  = std::uniform_real_distribution<double>(
        capture_distance + dist_init_offset_min,
        capture_distance + dist_init_offset_max
    );
    ta_lead_distrib = std::uniform_int_distribution<int>(0, 1);

    current_time = 0.0;
    reset();
}

// ── Reset ────────────────────────────────────────────────────────────────────

void MultiAgentOGE::reset()
{
    current_time = 0.0;

    // Base orbit parameters (shared)
    const Eigen::Matrix<double, 6, 1> coe_base = (
        Eigen::Matrix<double, 6, 1>() <<
        settings.getFloat("sma_base",  true),
        settings.getFloat("ecc_base",  true),
        settings.getFloat("incl_base", true),
        settings.getFloat("RA_base",   true),
        settings.getFloat("w_base",    true),
        settings.getFloat("TA_base",   true)
    ).finished();

    // ── HVT (evader[0]) ─────────────────────────────────────────────────────
    {
        Eigen::Matrix<double, 6, 1> coe = coe_base;
        coe[0] += sma_perturb_distrib(_rng);
        coe[5]  = true_anomaly_distrib(_rng);
        coe2rv(coe, agents_states[0].r_j2000, agents_states[0].v_j2000);
        agents_states[0].dv_remain = dv_init_evader;
        agents_states[0].is_alive  = true;
    }

    // ── Interceptors (evader[1 .. num_evaders-1]) ────────────────────────────
    // Place each interceptor at a small TA offset from HVT
    for (int i = 1; i < num_evaders; ++i)
    {
        Eigen::Matrix<double, 6, 1> coe = coe_base;
        coe[0] += sma_perturb_distrib(_rng);
        // Small phase offset relative to HVT: ±(i * 5 km / sma) radians
        double sign = (ta_lead_distrib(_rng) == 0) ? -1.0 : 1.0;
        double offset_km = 5.0 * static_cast<double>(i); // 5km per interceptor index
        coe[5] = agents_states[0].r_j2000.norm() > 0.0
            ? true_anomaly_distrib(_rng)   // fallback: random TA
            : 0.0;
        // Derive approximate TA from HVT COE
        Eigen::Matrix<double, 6, 1> hvt_coe;
        rv2coe(agents_states[0].r_j2000, agents_states[0].v_j2000, hvt_coe);
        coe[5] = hvt_coe[5] + sign * offset_km / coe[0];
        coe2rv(coe, agents_states[i].r_j2000, agents_states[i].v_j2000);
        agents_states[i].dv_remain = dv_init_evader;
        agents_states[i].is_alive  = true;
    }

    // ── Pursuers (red[0 .. num_pursuers-1]) ──────────────────────────────────
    // Each pursuer starts dist_init_offset away from HVT
    for (int p = 0; p < num_pursuers; ++p)
    {
        int idx = num_evaders + p;
        Eigen::Matrix<double, 6, 1> coe = coe_base;
        coe[0] += sma_perturb_distrib(_rng);

        Eigen::Matrix<double, 6, 1> hvt_coe;
        rv2coe(agents_states[0].r_j2000, agents_states[0].v_j2000, hvt_coe);

        double ta_lead = (ta_lead_distrib(_rng) == 0) ? -1.0 : 1.0;
        double distance_offset = dist_init_offset_distrib(_rng);
        coe[5] = hvt_coe[5] + ta_lead * distance_offset / coe[0];

        coe2rv(coe, agents_states[idx].r_j2000, agents_states[idx].v_j2000);
        agents_states[idx].dv_remain = dv_init_pursuer;
        agents_states[idx].is_alive  = true;
    }
}

void MultiAgentOGE::resetWithStates(
    const std::unordered_map<std::string, SatState>& states)
{
    current_time = 0.0;
    for (int i = 0; i < num_agents; ++i)
    {
        auto it = states.find(agent_ids[i]);
        if (it != states.end())
            agents_states[i] = it->second;
    }
}

// ── Act ──────────────────────────────────────────────────────────────────────

void MultiAgentOGE::act(const std::vector<Eigen::Vector3d>& actions)
{
    if (static_cast<int>(actions.size()) != num_agents)
        throw std::invalid_argument("actions.size() != num_agents");

    processDynamics(actions);
    checkAlive();
}

void MultiAgentOGE::processDynamics(const std::vector<Eigen::Vector3d>& actions)
{
    for (int i = 0; i < num_agents; ++i)
    {
        bool is_evader = (i < num_evaders);
        double dv_max  = is_evader ? dv_max_per_step_evader : dv_max_per_step_pursuer;

        Eigen::Vector3d dv = Eigen::Vector3d::Zero();
        if (agents_states[i].dv_remain > 0.0 && !almost_equal(actions[i].norm(), 0.0))
        {
            double cap = std::min(dv_max, agents_states[i].dv_remain);
            if (actions[i].norm() > cap)
                dv = cap * actions[i].normalized();
            else
                dv = actions[i];
        }

        // Convert LVLH delta-v to J2000
        Eigen::Vector3d r_tmp, v_tmp;
        RV_LVLH2J2000(
            agents_states[i].r_j2000, agents_states[i].v_j2000,
            Eigen::Vector3d::Zero(), dv,
            r_tmp, v_tmp
        );
        agents_states[i].v_j2000  = v_tmp;
        agents_states[i].dv_remain -= dv.norm();

        // Propagate
        Eigen::Vector3d r_new, v_new;
        rv_from_r0v0(
            agents_states[i].r_j2000, agents_states[i].v_j2000,
            timestep, r_new, v_new
        );
        agents_states[i].r_j2000 = r_new;
        agents_states[i].v_j2000 = v_new;
    }
    current_time += timestep;
}

// ── Termination ──────────────────────────────────────────────────────────────

void MultiAgentOGE::checkAlive()
{
    // HVT captured: any pursuer within capture_distance of HVT (blue[0])
    for (int p = 0; p < num_pursuers; ++p)
    {
        int idx = num_evaders + p;
        if (!agents_states[idx].is_alive) continue;
        if ((agents_states[0].r_j2000 - agents_states[idx].r_j2000).norm() < capture_distance)
        {
            agents_states[0].is_alive = false;
            break;
        }
    }

    // Pursuer intercepted: any interceptor (blue[1+]) within intercept_distance of a pursuer
    for (int p = 0; p < num_pursuers; ++p)
    {
        int pidx = num_evaders + p;
        if (!agents_states[pidx].is_alive) continue;
        for (int e = 1; e < num_evaders; ++e)
        {
            if (!agents_states[e].is_alive) continue;
            if ((agents_states[e].r_j2000 - agents_states[pidx].r_j2000).norm() < intercept_distance)
            {
                agents_states[pidx].is_alive = false;
                break;
            }
        }
    }
}

bool MultiAgentOGE::isTerminal() const
{
    for (int i = 0; i < num_agents; ++i)
        if (!agents_states[i].is_alive) return true;
    return false;
}

bool MultiAgentOGE::isTruncated() const
{
    return current_time >= terminal_time;
}

bool MultiAgentOGE::isHVTCaptured() const
{
    return !agents_states[0].is_alive;
}

bool MultiAgentOGE::isPursuerIntercepted() const
{
    for (int p = 0; p < num_pursuers; ++p)
    {
        if (!agents_states[num_evaders + p].is_alive) return true;
    }
    return false;
}

// ── Sun position ─────────────────────────────────────────────────────────────

Eigen::Vector3d MultiAgentOGE::computeSunPosition() const
{
    Eigen::Vector3d pos_sun;
    solar_position(jd_epoch_ + current_time / 86400.0, pos_sun);
    return pos_sun;
}

// ── Observations ─────────────────────────────────────────────────────────────
//
// Layout per agent i (size = 13 + 7*(N-1)):
//   [0:3]                  own r_j2000 (km)
//   [3:6]                  own v_j2000 (km/s)
//   for each other agent j != i  [N-1 blocks, ordered 0..N-1 skipping i]:
//     [6+7*k : 6+7*k+3]    rel_pos of j in own LVLH (km)
//     [6+7*k+3 : 6+7*k+6]  rel_vel of j in own LVLH (m/s)
//     [6+7*k+6]             dist(i,j) / 20km
//   tail (7 scalars):
//     [base+0]  solar_angle w.r.t. HVT (rad)
//     [base+1]  dv_remain (km/s)
//     [base+2]  time_progress [0,1]
//     [base+3]  dv_ratio (dv_remain / dv_init)
//     [base+4:base+7]  sun_dir in HVT LVLH (unit vector)

void MultiAgentOGE::getObservations(std::vector<Eigen::VectorXd>& observations) const
{
    observations.resize(num_agents);

    const int obs_size = getObsSize();
    const int base     = 6 + 7 * (num_agents - 1);  // index of the 7 tail elements

    Eigen::Vector3d pos_sun = computeSunPosition();
    double time_progress = current_time / terminal_time;

    // Solar angle: Sun->HVT ^ HVT->self — computed per agent
    // Sun direction in HVT LVLH (for all agents)
    Eigen::Matrix3d dcm_hvt;
    DCM_J2000_to_LVLH(agents_states[0].r_j2000, agents_states[0].v_j2000, dcm_hvt);
    Eigen::Vector3d sun_dir_hvt_lvlh =
        dcm_hvt * (pos_sun - agents_states[0].r_j2000).normalized();

    for (int i = 0; i < num_agents; ++i)
    {
        observations[i].resize(obs_size);

        // Own state
        observations[i].segment<3>(0) = agents_states[i].r_j2000;
        observations[i].segment<3>(3) = agents_states[i].v_j2000;

        // Relative states to every other agent
        int k = 0;
        for (int j = 0; j < num_agents; ++j)
        {
            if (j == i) continue;

            Eigen::Vector3d r_j_lvlh, v_j_lvlh;
            RV_J20002LVLH(
                agents_states[i].r_j2000, agents_states[i].v_j2000,
                agents_states[j].r_j2000, agents_states[j].v_j2000,
                r_j_lvlh, v_j_lvlh
            );
            int blk = 6 + 7 * k;
            observations[i].segment<3>(blk)     = r_j_lvlh;
            observations[i].segment<3>(blk + 3) = v_j_lvlh * 1000.0; // km/s → m/s
            observations[i](blk + 6)             = r_j_lvlh.norm() / 20.0;
            ++k;
        }

        // Solar angle computation depends on agent role:
        //   evader[0]  (HVT)         : undefined — use 0
        //   evader[1+] (interceptor) : vertex = first pursuer (Blue),
        //                              angle between (Blue→Sun) and (Blue→self)
        //   pursuer    (Blue Recon)  : vertex = HVT,
        //                              angle between (HVT→Sun) and (HVT→self)
        double solar_angle;
        if (i == 0)
        {
            solar_angle = 0.0;
        }
        else if (i >= 1 && i < num_evaders)
        {
            // Interceptor/Escort: Blue (first pursuer) as vertex
            int blue_idx = num_evaders;  // first pursuer
            solar_angle = solar_illumination_angle(
                pos_sun,
                agents_states[blue_idx].r_j2000,  // Blue position (vertex)
                agents_states[i].r_j2000           // self position
            );
        }
        else
        {
            // Pursuer (Blue Recon): HVT as vertex
            solar_angle = solar_illumination_angle(
                pos_sun,
                agents_states[0].r_j2000,   // HVT position (vertex)
                agents_states[i].r_j2000    // self position
            );
        }

        bool is_evader = (i < num_evaders);
        double dv_init = is_evader ? dv_init_evader : dv_init_pursuer;

        observations[i](base + 0) = solar_angle;
        observations[i](base + 1) = agents_states[i].dv_remain;
        observations[i](base + 2) = time_progress;
        observations[i](base + 3) = agents_states[i].dv_remain / dv_init;
        observations[i].segment<3>(base + 4) = sun_dir_hvt_lvlh;
    }
}

// ── State access ─────────────────────────────────────────────────────────────

std::unordered_map<std::string, SatState> MultiAgentOGE::getSatStates() const
{
    std::unordered_map<std::string, SatState> result;
    result.reserve(num_agents);
    for (int i = 0; i < num_agents; ++i)
        result[agent_ids[i]] = agents_states[i];
    return result;
}

} // namespace oge
