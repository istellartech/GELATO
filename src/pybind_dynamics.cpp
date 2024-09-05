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
#include "wrapper_coordinate.hpp"
#include "wrapper_air.hpp"


matXd dynamics_velocity(
    vecXd mass_e,
    matXd pos_eci_e,
    matXd vel_eci_e,
    matXd quat_eci2body,
    vecXd t,
    vecXd param,
    matXd wind_table,
    matXd CA_table,
    vecXd units) {

    vecXd mass = mass_e * units[0];
    matXd pos_eci = pos_eci_e * units[1];
    matXd vel_eci = vel_eci_e * units[2];
    matXd acc_eci = matXd::Zero(pos_eci.rows(), 3);

    double thrust_vac = param[0];
    double air_area = param[2];
    double nozzle_area = param[4];

    for (int i = 0; i < mass.rows(); i++) {
        vec3d pos_llh = ecef2geodetic(pos_eci(i, 0), pos_eci(i, 1), pos_eci(i, 2));
        double altitude = geopotential_altitude(pos_llh[2]);
        double rho = airdensity_at(altitude);
        double p = airpressure_at(altitude);

        vec3d vel_ecef = vel_eci2ecef(vel_eci.row(i), pos_eci.row(i), t[i]);
        vec3d vel_wind_ned = wind_ned(altitude, wind_table);

        vec3d vel_wind_eci = quatrot(quat_nedg2eci(pos_eci.row(i), t[i]), vel_wind_ned);
        vec3d vel_air_eci = ecef2eci(vel_ecef, t[i]) - vel_wind_eci;
        double mach_number = vel_air_eci.norm() / speed_of_sound(altitude);

        double ca = interp(mach_number, CA_table.col(0), CA_table.col(1));

        vec3d aeroforce_eci = 0.5 * rho * air_area * ca * vel_air_eci.norm() * -vel_air_eci;

        double thrust = thrust_vac - nozzle_area * p;
        vec3d thrustdir_eci = quatrot(conj(quat_eci2body.row(i)), vec3d(1.0, 0.0, 0.0));
        vec3d thrust_eci = thrust * thrustdir_eci;
        vec3d gravity_eci = gravity(pos_eci.row(i));
        vec3d acc_i = (thrust_eci + aeroforce_eci) / mass[i] + gravity_eci;
        acc_eci.row(i) = acc_i;
    }

    return acc_eci / units[2];
}

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
    m.def("dynamics_velocity", &dynamics_velocity, "velocity with aerodynamic forces");
    m.def("dynamics_velocity_NoAir", &dynamics_velocity_NoAir, "velocity without aerodynamic forces");
    m.def("dynamics_quaternion", &dynamics_quaternion, "quaternion dynamics");
}