#pragma once

#include <memory>
#include <sstream>
#include <cstring>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/eigen/dense.h>

#include "oge/environment/multi_agent_oge.h"
#include "oge/environment/oge_settings.h"
#include "oge/environment/oge_state.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace oge
{

/**
 * Thin nanobind wrapper around MultiAgentOGE.
 *
 * Owns its OGESettings internally and builds them from a Python dict,
 * avoiding cross-module nanobind type compatibility issues with _oge_py.
 *
 * Accepted dict keys: same as OGEEnvCfg (sma_base, ecc_base, dv_init_red, …)
 */
class MultiAgentPythonInterface
{
public:
    /**
     * @param cfg               Python dict with the same keys as OGEEnvCfg
     * @param num_evaders       Blue satellites: index 0 = HVT, 1+ = interceptors
     * @param num_pursuers      Red satellites
     * @param intercept_distance km — distance at which interceptor neutralises pursuer
     */
    MultiAgentPythonInterface(
        nb::dict    cfg,
        int         num_evaders        = 1,
        int         num_pursuers       = 1,
        double      intercept_distance = 30.0
    );

    void reset() { env->reset(); }

    void resetWithStates(
        const std::unordered_map<std::string, SatState>& states)
    { env->resetWithStates(states); }

    /** actions: numpy array shape (num_agents, 3), LVLH delta-v km/s */
    void act(const nb::ndarray<nb::numpy, const double>& actions);

    /** Returns numpy array shape (num_agents, obs_size) — full observations */
    nb::ndarray<nb::numpy, double> getObservations() const;

    /**
     * Set per-agent task assignments from Python.
     * @param assignments  list of dicts, each with keys:
     *   "task_type"  : int  (0=STRIKE, 1=RECON, 2=JAM, 3=OPERATE)
     *   "target_idx" : int  (global agent index of task target)
     *   "threat_idx" : int  (global agent index of main threat, -1 if none)
     */
    void setTaskAssignment(const nb::list& assignments);

    /** Returns numpy array shape (num_agents, 17) — task-specific observations */
    nb::ndarray<nb::numpy, double> getTaskObservations() const;

    bool isTerminal()          const { return env->isTerminal(); }
    bool isTruncated()         const { return env->isTruncated(); }
    bool isHVTCaptured()       const { return env->isHVTCaptured(); }
    bool isPursuerIntercepted()const { return env->isPursuerIntercepted(); }

    int    getObsSize()     const { return env->getObsSize(); }
    int    getNumAgents()   const { return env->getNumAgents(); }
    int    getNumEvaders()  const { return env->getNumEvaders(); }
    int    getNumPursuers() const { return env->getNumPursuers(); }
    double getCurrentTime() const { return env->getCurrentTime(); }

    std::unordered_map<std::string, SatState> getSatStates() const
    { return env->getSatStates(); }

private:
    OGESettings                 _settings; // owned copy
    std::unique_ptr<MultiAgentOGE> env;
};

} // namespace oge
