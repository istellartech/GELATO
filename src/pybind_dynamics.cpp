#include "wrapper_coordinate.hpp"

matXd dynamics_velocity_NoAir(vecXd mass_e, matXd pos_eci_e, matXd quat_eci2body, vecXd param, vecXd units) {

    vecXd mass = mass_e * units[0];
    matXd pos_eci = pos_eci_e * units[1];
    matXd acc_eci = matXd::Zero(pos_eci.rows(), 3);

    double thrust_vac = param[0];

    for (int i = 0; i < mass.rows(); i++) {
        double thrust = thrust_vac;
        vec3d thrustdir_eci = quatrot(conj(quat_eci2body.row(i)), vec3d(1.0, 0.0, 0.0));
        vec3d thrust_eci = thrust * thrustdir_eci;
        vec3d gravity_eci = gravity(pos_eci.row(i));
        vec3d acc_i = thrust_eci / mass[i] + gravity_eci;
        acc_eci.row(i) = acc_i;
    }

    return acc_eci / units[2];
}

matXd dynamics_quaternion(matXd quat_eci2body, matXd u_e, double unit_u) {

    matXd u = u_e * unit_u;
    matXd d_quat = matXd::Zero(quat_eci2body.rows(), 4);

    for (int i = 0; i < quat_eci2body.rows(); i++) {
        vec4d omega_rps_body = vec4d(0.0, u(i, 0), u(i, 1), u(i, 2));
        omega_rps_body = omega_rps_body * M_PI / 180.0;
        vec4d d_quat_i = 0.5 * quatmult(quat_eci2body.row(i), omega_rps_body);
        d_quat.row(i) = d_quat_i;
    }

    return d_quat;
}

PYBIND11_MODULE(dynamics_c, m) {
    m.def("dynamics_velocity_NoAir", &dynamics_velocity_NoAir, "velocity without air resistance");
    m.def("dynamics_quaternion", &dynamics_quaternion, "quaternion dynamics");
}