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

#include "wrapper_air.hpp"

PYBIND11_MODULE(USStandardAtmosphere_c, m) {
    m.def("geopotential_altitude", &geopotential_altitude, "geopotential altitude");
    m.def("airtemperature_at", &airtemperature_at, "air temperature at altitude");
    m.def("airpressure_at", &airpressure_at, "air pressure at altitude");
    m.def("airdensity_at", &airdensity_at, "air density at altitude");
    m.def("speed_of_sound", &speed_of_sound, "speed of sound at altitude");
}
