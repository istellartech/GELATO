//
// The MIT License
//
// Copyright (c) 2024 Interstellar Technologies Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <cmath>

#include "Air.hpp"

namespace py = pybind11;

#ifndef SRC_WRAPPER_AIR_HPP_
#define SRC_WRAPPER_AIR_HPP_

using vec3d = Eigen::Matrix<double, 3, 1>;
using vec4d = Eigen::Matrix<double, 4, 1>;
using vecXd = Eigen::Matrix<double, -1, 1>;
using matXd = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

double geopotential_altitude(double z) { return Air::geopotential_altitude(z); }

double airtemperature_at(double altitude_m) {
  return Air::temperature(altitude_m);
}

double airpressure_at(double altitude_m) { return Air::pressure(altitude_m); }

double airdensity_at(double altitude_m) { return Air::density(altitude_m); }

double speed_of_sound(double altitude_m) {
  return Air::speed_of_sound(altitude_m);
}

#endif  // SRC_WRAPPER_AIR_HPP_
