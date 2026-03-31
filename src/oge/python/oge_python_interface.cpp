//
// Created by baoyicui on 2/21/26.
//
#include "oge/python/oge_python_interface.h"

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
