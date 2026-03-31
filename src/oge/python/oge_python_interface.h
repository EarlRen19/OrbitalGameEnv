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

#endif //ORBITALGAMEENV_OGE_PYTHON_INTERFACE_H
