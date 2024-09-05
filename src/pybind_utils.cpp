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

#include "wrapper_utils.hpp"

PYBIND11_MODULE(utils_c, m) {
    m.def("haversine", &haversine, "Haversine formula");
    m.def("wind_ned", &wind_ned, "Wind in NED frame");
    m.def("angle_of_attack_all_rad", &angle_of_attack_all_rad, "Angle of attack in radians");
    m.def("angle_of_attack_ab_rad", &angle_of_attack_ab_rad, "Angle of attack in radians");
    m.def("dynamic_pressure_pa", &dynamic_pressure_pa, "Dynamic pressure in Pascals");
    m.def("q_alpha_pa_rad", &q_alpha_pa_rad, "Dynamic pressure times angle of attack in Pascals * radians");
}
