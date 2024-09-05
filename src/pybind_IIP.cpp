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

#include "iip.hpp"

namespace py = pybind11;

Eigen::Vector3d posLLH_IIP_FAA_deg(
    Eigen::Vector3d posECEF,
    Eigen::Vector3d velECEF,
    bool fill_na,
    int n_iter) {
  Eigen::Vector3d posLLH = posLLH_IIP_FAA(posECEF, velECEF);
    if (!fill_na) {
        if (posLLH[0] == 0.0 && posLLH[1] == 0.0 && posLLH[2] == 0.0) {
            posLLH[0] = std::numeric_limits<double>::quiet_NaN();
            posLLH[1] = std::numeric_limits<double>::quiet_NaN();
            posLLH[2] = std::numeric_limits<double>::quiet_NaN();
            return posLLH;
        }
    }

    posLLH[0] *= 180.0 / M_PI;
    posLLH[1] *= 180.0 / M_PI;

    return posLLH;
}


PYBIND11_MODULE(IIP_c, m) {
  m.def("posLLH_IIP_FAA", &posLLH_IIP_FAA_deg, "Calculate IIP",
        py::arg("posECEF"), py::arg("velECEF"),
        py::arg("fill_na") = true, py::arg("n_iter") = 5);
}
