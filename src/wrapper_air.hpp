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
