// nanobind module _oge_py_ma — multi-agent pursuit-evasion extension.
// Completely independent from _oge_py; does NOT modify any existing files.

#include "multi_agent_python_interface.h"

NB_MODULE(_oge_py_ma, m)
{
    m.doc() = "Multi-agent orbital pursuit-evasion environment (C++ backend)";

    nb::class_<oge::MultiAgentPythonInterface>(m, "MultiAgentOGEEnv")
        .def(nb::init<nb::dict, int, int, double>(),
             "cfg"_a,
             "num_evaders"_a        = 1,
             "num_pursuers"_a       = 1,
             "intercept_distance"_a = 30.0,
             "Create multi-agent environment.\n"
             "  cfg: dict with the same keys as OGEEnvCfg\n"
             "  num_evaders: blue team size (index 0 = HVT, 1+ = interceptors)\n"
             "  num_pursuers: red team size\n"
             "  intercept_distance: km")

        .def("reset",              &oge::MultiAgentPythonInterface::reset)
        .def("reset_with_states",  &oge::MultiAgentPythonInterface::resetWithStates,
             "states"_a)
        .def("act",                &oge::MultiAgentPythonInterface::act,
             "actions"_a)
        .def("get_observations",   &oge::MultiAgentPythonInterface::getObservations)

        .def("is_terminal",            &oge::MultiAgentPythonInterface::isTerminal)
        .def("is_truncated",           &oge::MultiAgentPythonInterface::isTruncated)
        .def("is_hvt_captured",        &oge::MultiAgentPythonInterface::isHVTCaptured)
        .def("is_pursuer_intercepted", &oge::MultiAgentPythonInterface::isPursuerIntercepted)

        .def("get_obs_size",      &oge::MultiAgentPythonInterface::getObsSize)
        .def("get_num_agents",    &oge::MultiAgentPythonInterface::getNumAgents)
        .def("get_num_evaders",   &oge::MultiAgentPythonInterface::getNumEvaders)
        .def("get_num_pursuers",  &oge::MultiAgentPythonInterface::getNumPursuers)
        .def("get_current_time",  &oge::MultiAgentPythonInterface::getCurrentTime)
        .def("get_sat_states",    &oge::MultiAgentPythonInterface::getSatStates);
}

namespace oge
{

// ── Constructor: populate OGESettings from Python dict ───────────────────────

MultiAgentPythonInterface::MultiAgentPythonInterface(
    nb::dict    cfg,
    int         num_evaders,
    int         num_pursuers,
    double      intercept_distance
)
{
    // Populate _settings from the dict
    for (auto [k, v] : cfg)
    {
        std::string key = nb::cast<std::string>(k);
        if (nb::isinstance<nb::bool_>(v))
        {
            _settings.setBool(key, nb::cast<bool>(v));
        }
        else if (nb::isinstance<nb::int_>(v))
        {
            _settings.setInt(key, nb::cast<int>(v));
        }
        else if (nb::isinstance<nb::float_>(v))
        {
            _settings.setFloat(key, static_cast<float>(nb::cast<double>(v)));
        }
        else if (nb::isinstance<nb::str>(v))
        {
            _settings.setString(key, nb::cast<std::string>(v));
        }
        // Ignore unknown types silently
    }

    env = std::make_unique<MultiAgentOGE>(
        _settings, num_evaders, num_pursuers, intercept_distance);
}

// ── act ──────────────────────────────────────────────────────────────────────

void MultiAgentPythonInterface::act(
    const nb::ndarray<nb::numpy, const double>& actions)
{
    if (actions.ndim() != 2
        || static_cast<int>(actions.shape(0)) != env->getNumAgents()
        || actions.shape(1) != 3)
    {
        throw std::runtime_error(
            "actions must have shape (num_agents, 3)");
    }

    auto view = actions.view<const double, nb::ndim<2>>();
    const int n = static_cast<int>(view.shape(0));
    std::vector<Eigen::Vector3d> acts(n);
    for (int i = 0; i < n; ++i)
        acts[i] = Eigen::Vector3d(view(i, 0), view(i, 1), view(i, 2));

    env->act(acts);
}

// ── get_observations ─────────────────────────────────────────────────────────

nb::ndarray<nb::numpy, double> MultiAgentPythonInterface::getObservations() const
{
    std::vector<Eigen::VectorXd> obs_vec;
    env->getObservations(obs_vec);

    const int n   = env->getNumAgents();
    const int obs = env->getObsSize();
    auto* data = new double[static_cast<size_t>(n) * static_cast<size_t>(obs)];

    for (int i = 0; i < n; ++i)
        std::memcpy(data + i * obs, obs_vec[i].data(), obs * sizeof(double));

    nb::capsule owner(data,
        [](void* p) noexcept { delete[] static_cast<double*>(p); });
    const size_t shape[2] = {static_cast<size_t>(n),
                              static_cast<size_t>(obs)};
    return {data, 2, shape, owner};
}

} // namespace oge
