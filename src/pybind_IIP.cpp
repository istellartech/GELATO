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
