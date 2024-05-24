#include "wrapper_utils.hpp"

PYBIND11_MODULE(utils_c, m) {
    m.def("haversine", &haversine, "Haversine formula");
    m.def("wind_ned", &wind_ned, "Wind in NED frame");
    m.def("angle_of_attack_all_rad", &angle_of_attack_all_rad, "Angle of attack in radians");
    m.def("angle_of_attack_ab_rad", &angle_of_attack_ab_rad, "Angle of attack in radians");
    m.def("dynamic_pressure_pa", &dynamic_pressure_pa, "Dynamic pressure in Pascals");
    m.def("q_alpha_pa_rad", &q_alpha_pa_rad, "Dynamic pressure times angle of attack in Pascals * radians");
}
