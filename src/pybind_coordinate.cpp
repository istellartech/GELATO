#include "wrapper_coordinate.hpp"

PYBIND11_MODULE(coordinate_c, m) {
    m.def("quatmult", &quatmult, "quaternion multiplication");
    m.def("conj", &conj, "conjugate quaternion");
    m.def("normalize", &normalize, "normalize vector");
    m.def("quatrot", &quatrot, "rotate vector by quaternion");
    m.def("dcm_from_quat", &dcm_from_quat, "direction cosine matrix from quaternion");
    m.def("quat_from_dcm", &quat_from_dcm, "quaternion from direction cosine matrix");
    m.def("ecef2geodetic", &ecef2geodetic, "convert ECEF to geodetic");
    m.def("geodetic2ecef", &geodetic2ecef, "convert geodetic to ECEF");
    m.def("ecef2eci", &ecef2eci, "convert ECEF position to ECI");
    m.def("eci2ecef", &eci2ecef, "convert ECI position to ECEF");
    m.def("vel_ecef2eci", &vel_ecef2eci, "convert ECEF velocity to ECI");
    m.def("vel_eci2ecef", &vel_eci2ecef, "convert ECI velocity to ECEF");
    m.def("quat_eci2ecef", &quat_eci2ecef, "ECI to ECEF quaternion");
    m.def("quat_ecef2eci", &quat_ecef2eci, "ECEF to ECI quaternion");
    m.def("quat_ecef2nedg", &quat_ecef2nedg, "ECEF to NED quaternion");
    m.def("quat_nedg2ecef", &quat_nedg2ecef, "NED to ECEF quaternion");
    m.def("quat_eci2nedg", &quat_eci2nedg, "ECI to NED quaternion");
    m.def("quat_nedg2eci", &quat_nedg2eci, "NED to ECI quaternion");
    m.def("quat_from_euler", &quat_from_euler, "quaternion from Euler angles");
    m.def("gravity", &gravity, "gravity in ECI");
    m.def("quat_nedg2body", &quat_nedg2body, "NED to body quaternion");
    m.def("euler_from_quat", &euler_from_quat, "Euler angles from quaternion");
    m.def("euler_from_dcm", &euler_from_dcm, "Euler angles from direction cosine matrix");
    m.def("dcm_from_thrustvector", &dcm_from_thrustvector, "direction cosine matrix from thrust vector");
    m.def("eci2geodetic", &eci2geodetic, "convert ECI to geodetic");
    m.def("orbital_elements", &orbital_elements, "orbital elements from ECI position and velocity");
    m.def("distance_vincenty", &distance_vincenty, "Vincenty's formulae for geodesic distance");
    m.def("angular_momentum_vec", &angular_momentum_vec, "angular momentum vector from ECI position and velocity");
    m.def("angular_momentum", &angular_momentum, "angular momentum from ECI position and velocity");
    m.def("inclination_rad", &inclination_rad, "inclination from ECI position and velocity");
    m.def("inclination_cosine", &inclination_cosine, "inclination cosine from ECI position and velocity");
    m.def("laplace_vector", &laplace_vector, "Laplace vector from ECI position and velocity");
    m.def("orbit_energy", &orbit_energy, "orbital energy from ECI position and velocity");
    m.def("angular_momentum_from_altitude", &angular_momentum_from_altitude, "angular momentum from altitude");
    m.def("orbit_energy_from_altitude", &orbit_energy_from_altitude, "orbital energy from altitude");
}
