//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_OGE_PYTHON_INTERFACE_H
#define ORBITALGAMEENV_OGE_PYTHON_INTERFACE_H

#include <optional>
#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/eigen/dense.h>

#include "oge/oge_interface.h"
#include "oge/simcore/utils.h"
#include "version.h"

namespace nb = nanobind;
using namespace nb::literals;
void init_vector_module(nb::module_& m);

namespace oge
{
    class OGEPythonInterface : public OGEInterface
    {
    public:
        using OGEInterface::OGEInterface;

        nb::ndarray<nb::numpy, double> getRewards(const nb::ndarray<nb::numpy, const double>& actions) const;
        nb::ndarray<nb::numpy, double> getObservations() const;
        int getObsSize() const;
        bool isTerminal() const;
        bool isTruncated() const;
        void act(const nb::ndarray<nb::numpy, const double>& actions);
    };
}

NB_MODULE(_oge_py, m)
{
    m.attr("__version__") = OGE_VERSION;

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

#endif //ORBITALGAMEENV_OGE_PYTHON_INTERFACE_H
