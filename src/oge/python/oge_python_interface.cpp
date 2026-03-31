//
// Created by baoyicui on 2/21/26.
//
#include "oge/python/oge_python_interface.h"

NB_MODULE(_oge_py, m)
{
    m.attr("__version__") = OGE_VERSION;

    m.def("ma2ta", &oge::ma2ta,
          "ma"_a, "ecc"_a, "tol"_a = 1e-10, "max_iter"_a = 100,
          "Convert mean anomaly to true anomaly (rad)");

    m.def("coe2rv", [](double sma, double ecc, double incl, double raan, double argp, double ta) {
        Eigen::Matrix<double, 6, 1> coe;
        coe << sma, ecc, incl, raan, argp, ta;
        Eigen::Vector3d r, v;
        oge::coe2rv(coe, r, v);
        return nb::make_tuple(r, v);
    }, "sma"_a, "ecc"_a, "incl"_a, "raan"_a, "argp"_a, "ta"_a,
       "Convert orbital elements (km, rad) to position (km) and velocity (km/s) vectors");

    m.def("solar_illumination_angle", &oge::solar_illumination_angle,
          "pos_sun_j2000"_a, "pos_evader_j2000"_a, "pos_chaser_j2000"_a,
          "Calculate solar illumination angle between Sun->Evader and Evader->Chaser vectors");

    nb::class_<oge::SatState>(m, "SatState")
        .def(nb::init<>())
        .def_rw("r_j2000", &oge::SatState::r_j2000)
        .def_rw("v_j2000", &oge::SatState::v_j2000)
        .def_rw("dv_remain", &oge::SatState::dv_remain)
        .def_rw("is_alive", &oge::SatState::is_alive)
        .def("__repr__", [](const oge::SatState& s)
        {
            std::ostringstream oss;
            oss << s;
            return oss.str();
        });

    nb::class_<oge::OGESettings>(m, "OGESettings")
        .def(nb::init<>())
        .def("validate", &oge::OGESettings::validate)
        .def("set_int", &oge::OGESettings::setInt, "key"_a, "value"_a)
        .def("set_float", &oge::OGESettings::setFloat, "key"_a, "value"_a)
        .def("set_bool", &oge::OGESettings::setBool, "key"_a, "value"_a)
        .def("set_string", &oge::OGESettings::setString, "key"_a, "value"_a)
        .def("get_int", &oge::OGESettings::getInt, "key"_a, "strict"_a = false)
        .def("get_float", &oge::OGESettings::getFloat, "key"_a, "strict"_a = false)
        .def("get_bool", &oge::OGESettings::getBool, "key"_a, "strict"_a = false)
        .def("get_string", &oge::OGESettings::getString, "key"_a, "strict"_a = false);

    nb::class_<oge::OGEPythonInterface>(m, "OGEInterface")
        .def(nb::init<>())
        .def("init", &oge::OGEPythonInterface::init)
        .def("get_rewards", &oge::OGEPythonInterface::getRewards)
        .def("get_observations", &oge::OGEPythonInterface::getObservations)
        .def("is_terminal", &oge::OGEPythonInterface::isTerminal)
        .def("is_truncated", &oge::OGEPythonInterface::isTruncated)
        .def("get_obs_size", &oge::OGEPythonInterface::getObsSize)
        .def("get_current_time", &oge::OGEPythonInterface::getCurrentTime)
        .def("get_settings", [](oge::OGEPythonInterface& self)-> oge::OGESettings&
        {
            return *self.settings;
        }, nb::rv_policy::reference_internal)
        .def("act", &oge::OGEPythonInterface::act)
        .def("reset", &oge::OGEPythonInterface::reset)
        .def("get_sat_states", &oge::OGEInterface::getSatStates)
        .def("is_captured", &oge::OGEInterface::isCaptured)
        .def("reset_with_states", &oge::OGEInterface::resetWithStates, "states"_a)
        .def("init", &oge::OGEPythonInterface::init)
        .def("setInt", &oge::OGEPythonInterface::setInt)
        .def("setFloat", &oge::OGEPythonInterface::setFloat)
        .def("setBool", &oge::OGEPythonInterface::setBool)
        .def("setString", &oge::OGEPythonInterface::setString)
        .def("getInt", &oge::OGEPythonInterface::getInt, "key"_a, "strict"_a = false)
        .def("getFloat", &oge::OGEPythonInterface::getFloat, "key"_a, "strict"_a = false)
        .def("getBool", &oge::OGEPythonInterface::getBool, "key"_a, "strict"_a = false)
        .def("getString", &oge::OGEPythonInterface::getString, "key"_a, "strict"_a = false);
}

namespace oge
{
    nb::ndarray<nb::numpy, double> OGEPythonInterface::getRewards(
        const nb::ndarray<nb::numpy, const double>& actions) const
    {
        if (actions.ndim() != 2)
            throw std::runtime_error("Expected a numpy array with two dimensions.");
        if (actions.shape(1) != 3)
            throw std::runtime_error("Expected actions array with shape (num_agents, 3).");

        auto view = actions.view<const double, nb::ndim<2>>();
        const int n = static_cast<int>(view.shape(0));

        std::vector<Eigen::Vector3d> acts(n);
        for (int i = 0; i < n; i++)
            acts[i] = Eigen::Vector3d(view(i, 0), view(i, 1), view(i, 2));

        std::vector<double> rewards;
        OGEInterface::getRewards(acts, rewards);

        auto* data = new double[rewards.size()];
        std::copy(rewards.begin(), rewards.end(), data);

        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        size_t shape[1] = {rewards.size()};
        return {data, 1, shape, owner};
    }

    nb::ndarray<nb::numpy, double> OGEPythonInterface::getObservations() const
    {
        std::vector<Eigen::VectorXd> observations;
        OGEInterface::getObservations(observations);

        size_t num_agents = 2; // 1 blue_sat + 1 red_sat
        size_t obs_size = environment->getObsSize(0);
        size_t total_obs_size = num_agents * obs_size;

        if (observations.size() != num_agents)
        {
            throw std::runtime_error(
                "getObservations() returns wrong number of agent observations. Expected "
                + std::to_string(num_agents)
                + " but got " + std::to_string(observations.size())
            );
        }

        // allocate data arrays
        auto obs_raw = std::unique_ptr<double[]>(new double[total_obs_size]);
        for (auto i = 0; i < num_agents; ++i)
        {
            const auto& observation = observations[i];
            if (observation.size() != obs_size)
            {
                throw std::runtime_error(
                    "getObservations() returns wrong observation size. Expected "
                    + std::to_string(obs_size)
                    + " but got " + std::to_string(observation.size())
                );
            }
            std::memcpy(
                obs_raw.get() + i * obs_size,
                observation.data(),
                obs_size * sizeof(double)
            );
        }
        // Transfer ownership to capsules
        auto* obs_data = obs_raw.release();
        nb::capsule obs_owner(obs_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });

        // build numpy arrays
        const size_t obs_shape[2] = {num_agents, obs_size};

        auto obs = nb::ndarray<nb::numpy, double>(obs_data, 2, obs_shape, obs_owner);

        return obs;
    }

    int OGEPythonInterface::getObsSize() const
    {
        return environment->getObsSize(0);
    }

    bool OGEPythonInterface::isTerminal() const
    {
        return environment->isTerminal();
    }

    bool OGEPythonInterface::isTruncated() const
    {
        return environment->isTruncated();
    }

    void OGEPythonInterface::act(const nb::ndarray<nb::numpy, const double>& actions)
    {
        if (actions.ndim() != 2 || actions.shape(1) != 3)
            throw std::runtime_error("Expected actions array with shape (num_agents, 3).");

        auto view = actions.view<double, nb::ndim<2>>();
        const int n = static_cast<int>(view.shape(0));
        std::vector<Eigen::Vector3d> acts(n);
        for (int i = 0; i < n; i++)
            acts[i] = Eigen::Vector3d(view(i, 0), view(i, 1), view(i, 2));
        OGEInterface::act(acts);
    }
}
