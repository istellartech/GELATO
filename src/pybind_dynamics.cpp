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

vec3d dynamics_velocity_single(
    double mass_e,
    vec3d pos_eci_e,
    vec3d vel_eci_e,
    vec4d quat_eci2body,
    double t,
    vecXd param,
    matXd wind_table,
    matXd CA_table,
    vecXd units) {

    double mass = mass_e * units[0];
    vec3d pos_eci = pos_eci_e * units[1];
    vec3d vel_eci = vel_eci_e * units[2];

    double thrust_vac = param[0];
    double air_area = param[2];
    double nozzle_area = param[4];

    vec3d pos_llh = ecef2geodetic(pos_eci(0), pos_eci(1), pos_eci(2));
    double altitude = geopotential_altitude(pos_llh[2]);
    double rho = airdensity_at(altitude);
    double p = airpressure_at(altitude);

    vec3d vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t);
    vec3d vel_wind_ned = wind_ned(altitude, wind_table);

    vec3d vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned);
    vec3d vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci;
    double mach_number = vel_air_eci.norm() / speed_of_sound(altitude);

    double ca = interp(mach_number, CA_table.col(0), CA_table.col(1));

    vec3d aeroforce_eci = 0.5 * rho * air_area * ca * vel_air_eci.norm() * -vel_air_eci;

    double thrust = thrust_vac - nozzle_area * p;
    vec3d thrustdir_eci = quatrot(conj(quat_eci2body), vec3d(1.0, 0.0, 0.0));
    vec3d thrust_eci = thrust * thrustdir_eci;
    vec3d gravity_eci = gravity(pos_eci);
    vec3d acc_eci = (thrust_eci + aeroforce_eci) / mass + gravity_eci;

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

vec3d dynamics_velocity_NoAir_single(double mass_e, vec3d pos_eci_e, vec4d quat_eci2body, vecXd param, vecXd units) {

    double mass = mass_e * units[0];
    vec3d pos_eci = pos_eci_e * units[1];

    double thrust_vac = param[0];

    double thrust = thrust_vac;
    vec3d thrustdir_eci = quatrot(conj(quat_eci2body), vec3d(1.0, 0.0, 0.0));
    vec3d thrust_eci = thrust * thrustdir_eci;
    vec3d gravity_eci = gravity(pos_eci);
    vec3d acc_eci = thrust_eci / mass + gravity_eci;

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

vecXd dynamics_quaternion_single(vecXd quat_eci2body, vecXd u_e, double unit_u) {

    vecXd u = u_e * unit_u;
    vecXd d_quat = vecXd::Zero(4);

    vec4d omega_rps_body = vec4d(0.0, u(0), u(1), u(2));
    omega_rps_body = omega_rps_body * M_PI / 180.0;
    d_quat = 0.5 * quatmult(quat_eci2body, omega_rps_body);

    return d_quat;
}

PYBIND11_MODULE(dynamics_c, m) {
    m.def("dynamics_velocity", &dynamics_velocity, "velocity with aerodynamic forces");
    m.def("dynamics_velocity_NoAir", &dynamics_velocity_NoAir, "velocity without aerodynamic forces");
    m.def("dynamics_quaternion", &dynamics_quaternion, "quaternion dynamics");
    m.def("dynamics_velocity_single", &dynamics_velocity_single, "velocity with aerodynamic forces single");
    m.def("dynamics_velocity_NoAir_single", &dynamics_velocity_NoAir_single, "velocity without aerodynamic forces single");
    m.def("dynamics_quaternion_single", &dynamics_quaternion_single, "quaternion dynamics single");
}