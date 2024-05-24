#include "wrapper_air.hpp"

PYBIND11_MODULE(USStandardAtmosphere_c, m) {
    m.def("geopotential_altitude", &geopotential_altitude, "geopotential altitude");
    m.def("airtemperature_at", &airtemperature_at, "air temperature at altitude");
    m.def("airpressure_at", &airpressure_at, "air pressure at altitude");
    m.def("airdensity_at", &airdensity_at, "air density at altitude");
    m.def("speed_of_sound", &speed_of_sound, "speed of sound at altitude");
}
