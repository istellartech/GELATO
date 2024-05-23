#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <cmath>
#include "Air.hpp"


namespace py = pybind11;

using vec3d = Eigen::Matrix<double, 3, 1>;
using vec4d = Eigen::Matrix<double, 4, 1>;
using vecXd = Eigen::Matrix<double, -1, 1>;
using matXd = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

double geopotential_altitude(double z) {
    return Air::geopotential_altitude(z);
}

double airtemperature_at(double altitude_m) {
    return Air::temperature(altitude_m);
}

double airpressure_at(double altitude_m) {
    return Air::pressure(altitude_m);
}

double airdensity_at(double altitude_m) {
    return Air::density(altitude_m);
}

double speed_of_sound(double altitude_m) {
    return Air::speed_of_sound(altitude_m);
}

PYBIND11_MODULE(USStandardAtmosphere_c, m) {
    m.def("geopotential_altitude", &geopotential_altitude, "geopotential altitude");
    m.def("airtemperature_at", &airtemperature_at, "air temperature at altitude");
    m.def("airpressure_at", &airpressure_at, "air pressure at altitude");
    m.def("airdensity_at", &airdensity_at, "air density at altitude");
    m.def("speed_of_sound", &speed_of_sound, "speed of sound at altitude");
}
