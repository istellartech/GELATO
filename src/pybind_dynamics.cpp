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
#include "wrapper_coordinate.hpp"
#include "wrapper_utils.hpp"

vec3d dynamics_velocity(double mass_e, vec3d pos_eci_e, vec3d vel_eci_e,
                        vec4d quat_eci2body, double t, vecXd param,
                        matXd wind_table, matXd CA_table, vecXd units) {
  double mass = mass_e * units[0];
  vec3d pos_eci = pos_eci_e * units[1];
  vec3d vel_eci = vel_eci_e * units[2];

  double thrust_vac = param[0];
  double air_area = param[2];
  double nozzle_area = param[4];

  vec3d aeroforce_eci(0.0, 0.0, 0.0);
  double thrust = thrust_vac;
  bool use_air = (air_area > 0.0);


  if (use_air) {
    vec3d pos_llh = ecef2geodetic(pos_eci[0], pos_eci[1], pos_eci[2]);
    double altitude = geopotential_altitude(pos_llh[2]);
    double rho = airdensity_at(altitude);
    double p = airpressure_at(altitude);

    vec3d vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t);
    vec3d vel_wind_ned = wind_ned(altitude, wind_table);

    vec3d vel_wind_eci =
        quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned);
    vec3d vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci;
    double mach_number = vel_air_eci.norm() / speed_of_sound(altitude);

    double ca = interp(mach_number, CA_table.col(0), CA_table.col(1));

    aeroforce_eci =
        0.5 * rho * air_area * ca * vel_air_eci.norm() * -vel_air_eci;

    thrust = thrust_vac - nozzle_area * p;
  } else {
    aeroforce_eci.setZero();
    thrust = thrust_vac;
  }
  vec3d thrustdir_eci =
      quatrot(conj(quat_eci2body), vec3d(1.0, 0.0, 0.0));
  vec3d thrust_eci = thrust * thrustdir_eci;
  vec3d gravity_eci = gravity(pos_eci);
  vec3d acc_eci = (thrust_eci + aeroforce_eci) / mass + gravity_eci;

  return acc_eci / units[2];
}

matXd dynamics_velocity_array(vecXd mass_e, matXd pos_eci_e, matXd vel_eci_e,
                        matXd quat_eci2body, vecXd t, vecXd param,
                        matXd wind_table, matXd CA_table, vecXd units) {

  matXd acc_eci_e = matXd::Zero(pos_eci_e.rows(), 3);
  for (int i = 0; i < pos_eci_e.rows(); i++) {
    acc_eci_e.row(i) = dynamics_velocity(mass_e(i), pos_eci_e.row(i), vel_eci_e.row(i),
                        quat_eci2body.row(i), t(i), param,
                        wind_table, CA_table, units);
  }

  return acc_eci_e;
}

vec4d dynamics_quaternion(vec4d quat_eci2body, vec3d u_e, double unit_u) {

  vec3d omega_rps_body = u_e * unit_u * M_PI / 180.0;

  vec4d d_quat = vec4d::Zero();
  d_quat[0] = 0.5 * (-quat_eci2body[1] * omega_rps_body[0] -
                      quat_eci2body[2] * omega_rps_body[1] -
                      quat_eci2body[3] * omega_rps_body[2]);
  d_quat[1] = 0.5 * ( quat_eci2body[0] * omega_rps_body[0] -
                      quat_eci2body[3] * omega_rps_body[1] +
                      quat_eci2body[2] * omega_rps_body[2]);
  d_quat[2] = 0.5 * ( quat_eci2body[3] * omega_rps_body[0] +
                      quat_eci2body[0] * omega_rps_body[1] -
                      quat_eci2body[1] * omega_rps_body[2]);
  d_quat[3] = 0.5 * (-quat_eci2body[2] * omega_rps_body[0] +
                      quat_eci2body[1] * omega_rps_body[1] +
                      quat_eci2body[0] * omega_rps_body[2]);

  return d_quat;
}


matXd dynamics_quaternion_array(matXd quat_eci2body, matXd u_e, double unit_u) {

  matXd d_quat = matXd::Zero(quat_eci2body.rows(), 4);

  for (int i = 0; i < quat_eci2body.rows(); i++) {
    d_quat.row(i) = dynamics_quaternion(quat_eci2body.row(i), u_e.row(i), unit_u);
  }

  return d_quat;
}

py::dict dynamics_velocity_rh_gradient(
  double mass, vec3d pos, vec3d vel, vec4d quat, double t,
  vecXd param, matXd wind_table, matXd CA_table, vecXd units,
  double to, double tf, double unit_time, double dx
) {


  vec3d f_c = dynamics_velocity(
    mass, pos, vel, quat, t, param, wind_table, CA_table, units
  );

  // mass
  vec3d grad_mass = vec3d::Zero();
  mass += dx;
  vec3d f_p_mass = dynamics_velocity(
    mass, pos, vel, quat, t, param, wind_table, CA_table, units
  );
  mass -= dx;
  grad_mass = -(f_p_mass - f_c) / dx * (tf - to) * unit_time / 2.0;

  // position
  mat3d grad_position = mat3d::Zero();
  for (int k = 0; k < 3; k++) {
    pos(k) += dx;
    vec3d f_p_pos = dynamics_velocity(
      mass, pos, vel, quat, t, param, wind_table, CA_table, units
    );
    pos(k) -= dx;
    grad_position.col(k) = -(f_p_pos - f_c) / dx * (tf - to) * unit_time / 2.0;
  }

  // velocity: changes only affect aerodynamic forces
  mat3d grad_velocity = mat3d::Zero();
  if (param[2] > 0.0) {
    for (int k = 0; k < 3; k++) {
      vel(k) += dx;
      vec3d f_p_vel = dynamics_velocity(
        mass, pos, vel, quat, t, param, wind_table, CA_table, units
      );
      vel(k) -= dx;
      grad_velocity.col(k) = -(f_p_vel - f_c) / dx * (tf - to) * unit_time / 2.0;
    }
  }

  // quaternion
  matXd grad_quaternion = matXd::Zero(3, 4);
  for (int k = 0; k < 4; k++) {
    quat(k) += dx;
    vec3d f_p_quat = dynamics_velocity(
      mass, pos, vel, quat, t, param, wind_table, CA_table, units
    );
    quat(k) -= dx;
    grad_quaternion.col(k) = -(f_p_quat - f_c) / dx * (tf - to) * unit_time / 2.0;
  }

  // to, tf: changes only affect aerodynamic forces
  vec3d grad_to = vec3d::Zero();
  vec3d grad_tf = vec3d::Zero();
  if (param[2] > 0.0) {
    double to_p = to + dx;
    double t_p1 = to_p + t / (tf - to) * (tf - to_p);
    vec3d f_p_to = dynamics_velocity(
      mass, pos, vel, quat, t_p1, param, wind_table, CA_table, units
    );
    grad_to = -(f_p_to * (tf - to_p) - f_c * (tf - to)) / dx * unit_time / 2.0;

    double tf_p = tf + dx;
    double t_p2 = to + t / (tf - to) * (tf_p - to);
    vec3d f_p_tf = dynamics_velocity(
      mass, pos, vel, quat, t_p2, param, wind_table, CA_table, units
    );
    grad_tf = -(f_p_tf * (tf_p - to) - f_c * (tf - to)) / dx * unit_time / 2.0;
  } else {
    grad_to = f_c * unit_time / 2.0;
    grad_tf = -grad_to;
  }

  py::dict grad;
  grad["mass"] = grad_mass;
  grad["position"] = grad_position;
  grad["velocity"] = grad_velocity;
  grad["quaternion"] = grad_quaternion;
  grad["to"] = grad_to;
  grad["tf"] = grad_tf;

  return grad;
}

py::dict dynamics_quaternion_rh_gradient(
  vec4d quat, vec3d u, double unit_u,
  double to, double tf, double unit_time, double dx
) {

  vec4d f_c = dynamics_quaternion(quat, u, unit_u);

  vec3d omega_rps_body = u * unit_u * M_PI / 180.0;

  // quaternion
  matXd grad_quaternion = matXd::Zero(4, 4);
  grad_quaternion << 0.0, -0.5 * omega_rps_body[0], -0.5 * omega_rps_body[1], -0.5 * omega_rps_body[2],
                     0.5 * omega_rps_body[0], 0.0, 0.5 * omega_rps_body[2], -0.5 * omega_rps_body[1],
                     0.5 * omega_rps_body[1], -0.5 * omega_rps_body[2], 0.0, 0.5 * omega_rps_body[0],
                     0.5 * omega_rps_body[2], 0.5 * omega_rps_body[1], -0.5 * omega_rps_body[0], 0.0;
  grad_quaternion = -grad_quaternion * (tf - to) * unit_time / 2.0;

  // u (angular velocity)
  matXd grad_u = matXd::Zero(4, 3);
  grad_u << -0.5 * quat[1], -0.5 * quat[2], -0.5 * quat[3],
             0.5 * quat[0], -0.5 * quat[3], 0.5 * quat[2],
             0.5 * quat[3], 0.5 * quat[0], -0.5 * quat[1],
             -0.5 * quat[2], 0.5 * quat[1], 0.5 * quat[0];
  grad_u = -grad_u * unit_u * M_PI / 180.0 * (tf - to) * unit_time / 2.0;

  //to, tf
  vec4d grad_to = f_c * unit_time / 2.0;
  vec4d grad_tf = -grad_to;

  py::dict grad;
  grad["quaternion"] = grad_quaternion;
  grad["u"] = grad_u;
  grad["to"] = grad_to;
  grad["tf"] = grad_tf;

  return grad;
}

// =============================================================================
// Batch Jacobian: velocity dynamics
// Computes all COO entries for one section's velocity Jacobian in C++,
// eliminating the Python per-node loop and list.extend overhead.
// =============================================================================
py::dict jac_dynamics_velocity_section(
  int n, int ua, int xa, int section_idx,
  vecXd mass_i, matXd pos_i, matXd vel_i, matXd quat_i,
  vecXd t_nodes, vecXd param,
  matXd wind_table, matXd CA_table, vecXd units,
  double to, double tf, double unit_t, double dx,
  matXd Di
) {
  // Pre-calculate sizes for COO arrays
  // mass: n entries of 3 each = 3*n
  // position: n entries of 3*3 each = 9*n
  // velocity: n entries of (n+1) sub-blocks:
  //   - one 3x3 dense block (j+1==jcol): 9 entries
  //   - n diagonal entries (j+1!=jcol): 3 entries each
  //   total per node: 9 + 3*n => total: n*(9 + 3*n) = 9n + 3n^2
  // quaternion: n entries of 3*4 = 12*n
  // t: n entries of 2*3 = 6*n (to and tf)

  const int sz_mass = 3 * n;
  const int sz_pos  = 9 * n;
  const int sz_vel  = n * (9 + 3 * n);
  const int sz_quat = 12 * n;
  const int sz_t    = 6 * n;

  // Allocate COO arrays
  Eigen::VectorXi mass_row(sz_mass), mass_col(sz_mass);
  vecXd mass_data(sz_mass);
  Eigen::VectorXi pos_row(sz_pos), pos_col(sz_pos);
  vecXd pos_data(sz_pos);
  Eigen::VectorXi vel_row(sz_vel), vel_col(sz_vel);
  vecXd vel_data(sz_vel);
  Eigen::VectorXi quat_row(sz_quat), quat_col(sz_quat);
  vecXd quat_data(sz_quat);
  Eigen::VectorXi t_row(sz_t), t_col(sz_t);
  vecXd t_data(sz_t);

  int im = 0, ip = 0, iv = 0, iq = 0, it = 0;

  for (int j = 0; j < n; j++) {
    // Compute gradient for node j+1 (collocation node)
    double m_j   = mass_i(j + 1);
    vec3d  p_j   = pos_i.row(j + 1);
    vec3d  v_j   = vel_i.row(j + 1);
    vec4d  q_j   = quat_i.row(j + 1);
    double t_j   = t_nodes(j + 1);

    // --- Compute f_c ---
    vec3d f_c = dynamics_velocity(m_j, p_j, v_j, q_j, t_j, param,
                                  wind_table, CA_table, units);

    // --- mass gradient ---
    vec3d grad_mass;
    {
      double m_p = m_j + dx;
      vec3d f_p = dynamics_velocity(m_p, p_j, v_j, q_j, t_j, param,
                                    wind_table, CA_table, units);
      grad_mass = -(f_p - f_c) / dx * (tf - to) * unit_t / 2.0;
    }

    int row_base = (ua + j) * 3;
    for (int k = 0; k < 3; k++) {
      mass_row(im) = row_base + k;
      mass_col(im) = xa + 1 + j;
      mass_data(im) = grad_mass(k);
      im++;
    }

    // --- position gradient (3x3) ---
    mat3d grad_position = mat3d::Zero();
    for (int k = 0; k < 3; k++) {
      vec3d p_pert = p_j;
      p_pert(k) += dx;
      vec3d f_p = dynamics_velocity(m_j, p_pert, v_j, q_j, t_j, param,
                                    wind_table, CA_table, units);
      grad_position.col(k) = -(f_p - f_c) / dx * (tf - to) * unit_t / 2.0;
    }

    // COO: row repeats 3 times for each of 3 columns
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        pos_row(ip) = row_base + r;
        pos_col(ip) = (xa + 1 + j) * 3 + c;
        pos_data(ip) = grad_position(r, c);
        ip++;
      }
    }

    // --- velocity gradient (3x3), only if aero ---
    mat3d grad_velocity = mat3d::Zero();
    if (param[2] > 0.0) {
      for (int k = 0; k < 3; k++) {
        vec3d v_pert = v_j;
        v_pert(k) += dx;
        vec3d f_p = dynamics_velocity(m_j, p_j, v_pert, q_j, t_j, param,
                                      wind_table, CA_table, units);
        grad_velocity.col(k) = -(f_p - f_c) / dx * (tf - to) * unit_t / 2.0;
      }
    }

    // COO for velocity: iterate over all columns of Di
    for (int jcol = 0; jcol < n + 1; jcol++) {
      if (jcol == j + 1) {
        // Dense 3x3 block: D[j,jcol]*I + grad_velocity
        for (int r = 0; r < 3; r++) {
          for (int c = 0; c < 3; c++) {
            vel_row(iv) = row_base + r;
            vel_col(iv) = (xa + jcol) * 3 + c;
            double val = grad_velocity(r, c);
            if (r == c) val += Di(j, jcol);
            vel_data(iv) = val;
            iv++;
          }
        }
      } else {
        // Diagonal entries: D[j,jcol] on the diagonal
        for (int k = 0; k < 3; k++) {
          vel_row(iv) = row_base + k;
          vel_col(iv) = (xa + jcol) * 3 + k;
          vel_data(iv) = Di(j, jcol);
          iv++;
        }
      }
    }

    // --- quaternion gradient (3x4) ---
    matXd grad_quaternion = matXd::Zero(3, 4);
    for (int k = 0; k < 4; k++) {
      vec4d q_pert = q_j;
      q_pert(k) += dx;
      vec3d f_p = dynamics_velocity(m_j, p_j, v_j, q_pert, t_j, param,
                                    wind_table, CA_table, units);
      grad_quaternion.col(k) = -(f_p - f_c) / dx * (tf - to) * unit_t / 2.0;
    }

    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 4; c++) {
        quat_row(iq) = row_base + r;
        quat_col(iq) = (xa + 1 + j) * 4 + c;
        quat_data(iq) = grad_quaternion(r, c);
        iq++;
      }
    }

    // --- t gradients ---
    vec3d grad_to_vec = vec3d::Zero();
    vec3d grad_tf_vec = vec3d::Zero();
    if (param[2] > 0.0) {
      double to_p = to + dx;
      double t_p1 = to_p + t_j / (tf - to) * (tf - to_p);
      vec3d f_p_to = dynamics_velocity(m_j, p_j, v_j, q_j, t_p1, param,
                                       wind_table, CA_table, units);
      grad_to_vec = -(f_p_to * (tf - to_p) - f_c * (tf - to)) / dx * unit_t / 2.0;

      double tf_p = tf + dx;
      double t_p2 = to + t_j / (tf - to) * (tf_p - to);
      vec3d f_p_tf = dynamics_velocity(m_j, p_j, v_j, q_j, t_p2, param,
                                       wind_table, CA_table, units);
      grad_tf_vec = -(f_p_tf * (tf_p - to) - f_c * (tf - to)) / dx * unit_t / 2.0;
    } else {
      grad_to_vec = f_c * unit_t / 2.0;
      grad_tf_vec = -grad_to_vec;
    }

    for (int k = 0; k < 3; k++) {
      t_row(it) = row_base + k;
      t_col(it) = section_idx;
      t_data(it) = grad_to_vec(k);
      it++;
    }
    for (int k = 0; k < 3; k++) {
      t_row(it) = row_base + k;
      t_col(it) = section_idx + 1;
      t_data(it) = grad_tf_vec(k);
      it++;
    }
  }

  py::dict result;
  py::dict mass_coo, pos_coo, vel_coo, quat_coo, t_coo;
  mass_coo["row"] = mass_row;
  mass_coo["col"] = mass_col;
  mass_coo["data"] = mass_data;
  pos_coo["row"] = pos_row;
  pos_coo["col"] = pos_col;
  pos_coo["data"] = pos_data;
  vel_coo["row"] = vel_row;
  vel_coo["col"] = vel_col;
  vel_coo["data"] = vel_data;
  quat_coo["row"] = quat_row;
  quat_coo["col"] = quat_col;
  quat_coo["data"] = quat_data;
  t_coo["row"] = t_row;
  t_coo["col"] = t_col;
  t_coo["data"] = t_data;

  result["mass"] = mass_coo;
  result["position"] = pos_coo;
  result["velocity"] = vel_coo;
  result["quaternion"] = quat_coo;
  result["t"] = t_coo;

  return result;
}

// =============================================================================
// Batch Jacobian: quaternion dynamics
// Computes all COO entries for one section's quaternion Jacobian in C++.
// =============================================================================
py::dict jac_dynamics_quaternion_section(
  int n, int ua, int xa, int section_idx,
  matXd quat_i, matXd u_i,
  double unit_u, double to, double tf, double unit_t, double dx,
  matXd Di, bool is_hold
) {
  if (is_hold) {
    // Hold/vertical attitude: quat_i[1:] - quat_i[0] = 0
    // Jacobian: -I for xa*4..(xa+1)*4, +I for (xa+1)*4..xb*4
    const int sz_quat = 8 * n;  // 4*n for -1 entries + 4*n for +1 entries
    Eigen::VectorXi quat_row(sz_quat), quat_col(sz_quat);
    vecXd quat_data(sz_quat);

    int iq = 0;
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < 4; k++) {
        quat_row(iq) = (ua + j) * 4 + k;
        quat_col(iq) = xa * 4 + k;
        quat_data(iq) = -1.0;
        iq++;
      }
    }
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < 4; k++) {
        quat_row(iq) = (ua + j) * 4 + k;
        quat_col(iq) = (xa + 1 + j) * 4 + k;
        quat_data(iq) = 1.0;
        iq++;
      }
    }

    // Empty u and t
    Eigen::VectorXi empty_i(0);
    vecXd empty_d(0);

    py::dict result;
    py::dict q_coo, u_coo, t_coo;
    q_coo["row"] = quat_row;
    q_coo["col"] = quat_col;
    q_coo["data"] = quat_data;
    u_coo["row"] = empty_i;
    u_coo["col"] = empty_i;
    u_coo["data"] = empty_d;
    t_coo["row"] = empty_i;
    t_coo["col"] = empty_i;
    t_coo["data"] = empty_d;

    result["quaternion"] = q_coo;
    result["u"] = u_coo;
    result["t"] = t_coo;
    return result;
  }

  // Normal attitude control mode
  // quaternion COO: n * ((n+1) sub-blocks):
  //   diagonal (jcol==j+1): 4*4=16 entries
  //   off-diagonal: 4 entries each, n of them
  //   total per node: 16 + 4*n => total: n*(16 + 4*n) = 16n + 4n^2
  // u COO: n * 4*3 = 12*n
  // t COO: n * 2*4 = 8*n

  const int sz_quat = n * (16 + 4 * n);
  const int sz_u    = 12 * n;
  const int sz_t    = 8 * n;

  Eigen::VectorXi quat_row(sz_quat), quat_col(sz_quat);
  vecXd quat_data(sz_quat);
  Eigen::VectorXi u_row(sz_u), u_col(sz_u);
  vecXd u_data(sz_u);
  Eigen::VectorXi t_row(sz_t), t_col(sz_t);
  vecXd t_data(sz_t);

  int iq = 0, iu = 0, it = 0;

  for (int j = 0; j < n; j++) {
    vec4d q_j = quat_i.row(j + 1);
    vec3d u_j = u_i.row(j);

    // Analytical gradient
    vec3d omega = u_j * unit_u * M_PI / 180.0;
    double scale = (tf - to) * unit_t / 2.0;

    // grad_quaternion (4x4)
    matXd grad_quat = matXd::Zero(4, 4);
    grad_quat << 0.0, -0.5*omega[0], -0.5*omega[1], -0.5*omega[2],
                 0.5*omega[0], 0.0, 0.5*omega[2], -0.5*omega[1],
                 0.5*omega[1], -0.5*omega[2], 0.0, 0.5*omega[0],
                 0.5*omega[2], 0.5*omega[1], -0.5*omega[0], 0.0;
    grad_quat = -grad_quat * scale;

    // grad_u (4x3)
    matXd grad_u = matXd::Zero(4, 3);
    grad_u << -0.5*q_j[1], -0.5*q_j[2], -0.5*q_j[3],
               0.5*q_j[0], -0.5*q_j[3],  0.5*q_j[2],
               0.5*q_j[3],  0.5*q_j[0], -0.5*q_j[1],
              -0.5*q_j[2],  0.5*q_j[1],  0.5*q_j[0];
    grad_u = -grad_u * unit_u * M_PI / 180.0 * scale;

    // grad_to, grad_tf (4-vectors)
    vec4d f_c = dynamics_quaternion(q_j, u_j, unit_u);
    vec4d grad_to_vec = f_c * unit_t / 2.0;
    vec4d grad_tf_vec = -grad_to_vec;

    int row_base = (ua + j) * 4;

    // --- quaternion COO ---
    for (int jcol = 0; jcol < n + 1; jcol++) {
      if (jcol == j + 1) {
        // Dense 4x4 block: D[j,jcol]*I + grad_quat
        for (int r = 0; r < 4; r++) {
          for (int c = 0; c < 4; c++) {
            quat_row(iq) = row_base + r;
            quat_col(iq) = (xa + jcol) * 4 + c;
            double val = grad_quat(r, c);
            if (r == c) val += Di(j, jcol);
            quat_data(iq) = val;
            iq++;
          }
        }
      } else {
        // Diagonal entries: D[j,jcol]
        for (int k = 0; k < 4; k++) {
          quat_row(iq) = row_base + k;
          quat_col(iq) = (xa + jcol) * 4 + k;
          quat_data(iq) = Di(j, jcol);
          iq++;
        }
      }
    }

    // --- u COO ---
    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 3; c++) {
        u_row(iu) = row_base + r;
        u_col(iu) = (ua + j) * 3 + c;
        u_data(iu) = grad_u(r, c);
        iu++;
      }
    }

    // --- t COO ---
    for (int k = 0; k < 4; k++) {
      t_row(it) = row_base + k;
      t_col(it) = section_idx;
      t_data(it) = grad_to_vec(k);
      it++;
    }
    for (int k = 0; k < 4; k++) {
      t_row(it) = row_base + k;
      t_col(it) = section_idx + 1;
      t_data(it) = grad_tf_vec(k);
      it++;
    }
  }

  py::dict result;
  py::dict q_coo, u_coo_d, t_coo;
  q_coo["row"] = quat_row;
  q_coo["col"] = quat_col;
  q_coo["data"] = quat_data;
  u_coo_d["row"] = u_row;
  u_coo_d["col"] = u_col;
  u_coo_d["data"] = u_data;
  t_coo["row"] = t_row;
  t_coo["col"] = t_col;
  t_coo["data"] = t_data;

  result["quaternion"] = q_coo;
  result["u"] = u_coo_d;
  result["t"] = t_coo;
  return result;
}

// =============================================================================
// Batch Jacobian: mass dynamics
// Computes all COO entries for one section's mass Jacobian in C++.
// =============================================================================
py::dict jac_dynamics_mass_section(
  int n, int ua, int xa, int section_idx,
  bool engine_on, double massflow_coeff,  // massflow / unit_mass * unit_t / 2.0
  matXd Di  // n x (n+1)
) {
  if (engine_on) {
    const int sz_mass = n * (n + 1);
    const int sz_t    = 2 * n;

    Eigen::VectorXi mass_row(sz_mass), mass_col(sz_mass);
    vecXd mass_data(sz_mass);
    Eigen::VectorXi t_row(sz_t), t_col(sz_t);
    vecXd t_data(sz_t);

    int im = 0;
    for (int j = 0; j < n; j++) {
      for (int jcol = 0; jcol < n + 1; jcol++) {
        mass_row(im) = ua + j;
        mass_col(im) = xa + jcol;
        mass_data(im) = Di(j, jcol);
        im++;
      }
    }

    int it = 0;
    for (int j = 0; j < n; j++) {
      t_row(it) = ua + j;
      t_col(it) = section_idx;
      t_data(it) = -massflow_coeff;
      it++;
    }
    for (int j = 0; j < n; j++) {
      t_row(it) = ua + j;
      t_col(it) = section_idx + 1;
      t_data(it) = massflow_coeff;
      it++;
    }

    py::dict result, m_coo, t_coo;
    m_coo["row"] = mass_row;
    m_coo["col"] = mass_col;
    m_coo["data"] = mass_data;
    t_coo["row"] = t_row;
    t_coo["col"] = t_col;
    t_coo["data"] = t_data;
    result["mass"] = m_coo;
    result["t"] = t_coo;
    return result;

  } else {
    // Hold mass: jac is identity-like (quat_i[1:] - quat_i[0] = 0)
    const int sz_mass = 2 * n;

    Eigen::VectorXi mass_row(sz_mass), mass_col(sz_mass);
    vecXd mass_data(sz_mass);

    for (int j = 0; j < n; j++) {
      mass_row(j)     = ua + j;
      mass_col(j)     = xa;
      mass_data(j)    = -1.0;
      mass_row(n + j) = ua + j;
      mass_col(n + j) = xa + 1 + j;
      mass_data(n + j)= 1.0;
    }

    Eigen::VectorXi empty_i(0);
    vecXd empty_d(0);

    py::dict result, m_coo, t_coo;
    m_coo["row"] = mass_row;
    m_coo["col"] = mass_col;
    m_coo["data"] = mass_data;
    t_coo["row"] = empty_i;
    t_coo["col"] = empty_i;
    t_coo["data"] = empty_d;
    result["mass"] = m_coo;
    result["t"] = t_coo;
    return result;
  }
}

// =============================================================================
// Batch Jacobian: position dynamics
// Computes all COO entries for one section's position Jacobian in C++,
// eliminating the Python per-node loop and list.extend overhead.
// =============================================================================
py::dict jac_dynamics_position_section(
  int n, int ua, int xa, int section_idx,
  matXd vel_i,     // shape (n+1, 3)
  double to, double tf,
  double unit_vel, double unit_pos, double unit_t,
  matXd Di         // shape (n, n+1)
) {
  const int sz_vel = 3 * n;
  const int sz_t   = 6 * n;
  const int sz_pos = 3 * n * (n + 1);

  Eigen::VectorXi vel_row(sz_vel), vel_col(sz_vel);
  vecXd vel_data(sz_vel);
  Eigen::VectorXi t_row(sz_t), t_col(sz_t);
  vecXd t_data(sz_t);
  Eigen::VectorXi pos_row(sz_pos), pos_col(sz_pos);
  vecXd pos_data(sz_pos);

  double rh_vel = -unit_vel * (tf - to) * unit_t / 2.0 / unit_pos;

  // velocity: diagonal identity-scaled entries for collocation nodes
  for (int j = 0; j < n; j++) {
    for (int k = 0; k < 3; k++) {
      int idx = j * 3 + k;
      vel_row(idx) = (ua + j) * 3 + k;
      vel_col(idx) = (xa + 1 + j) * 3 + k;
      vel_data(idx) = rh_vel;
    }
  }

  // t entries
  int it = 0;
  for (int j = 0; j < n; j++) {
    for (int k = 0; k < 3; k++) {
      double val = vel_i(j + 1, k) * unit_vel * unit_t / 2.0 / unit_pos;
      t_row(it) = (ua + j) * 3 + k;
      t_col(it) = section_idx;
      t_data(it) = val;
      it++;
    }
  }
  for (int j = 0; j < n; j++) {
    for (int k = 0; k < 3; k++) {
      double val = -vel_i(j + 1, k) * unit_vel * unit_t / 2.0 / unit_pos;
      t_row(it) = (ua + j) * 3 + k;
      t_col(it) = section_idx + 1;
      t_data(it) = val;
      it++;
    }
  }

  // position: D matrix blocks (three independent coordinate dimensions)
  int ip = 0;
  for (int ki = 0; ki < 3; ki++) {
    for (int j = 0; j < n; j++) {
      for (int jcol = 0; jcol < n + 1; jcol++) {
        pos_row(ip) = (ua + j) * 3 + ki;
        pos_col(ip) = (xa + jcol) * 3 + ki;
        pos_data(ip) = Di(j, jcol);
        ip++;
      }
    }
  }

  py::dict result;
  py::dict vel_coo, t_coo, pos_coo;
  vel_coo["row"] = vel_row;
  vel_coo["col"] = vel_col;
  vel_coo["data"] = vel_data;
  t_coo["row"] = t_row;
  t_coo["col"] = t_col;
  t_coo["data"] = t_data;
  pos_coo["row"] = pos_row;
  pos_coo["col"] = pos_col;
  pos_coo["data"] = pos_data;
  result["velocity"] = vel_coo;
  result["t"] = t_coo;
  result["position"] = pos_coo;
  return result;
}

PYBIND11_MODULE(dynamics_c, m) {
  m.def("dynamics_velocity_array", &dynamics_velocity_array,
        "velocity with aerodynamic forces (array)");
  m.def("dynamics_quaternion_array", &dynamics_quaternion_array,
        "quaternion dynamics array");
  m.def("dynamics_velocity_rh_gradient", &dynamics_velocity_rh_gradient,
        "gradient of equality_dynamics_velocity RHS components");
  m.def("dynamics_quaternion_rh_gradient", &dynamics_quaternion_rh_gradient,
        "gradient of equality_dynamics_quaternion RHS components");
  m.def("jac_dynamics_velocity_section", &jac_dynamics_velocity_section,
        "Batch Jacobian for velocity dynamics (one section, returns COO arrays)");
  m.def("jac_dynamics_quaternion_section", &jac_dynamics_quaternion_section,
        "Batch Jacobian for quaternion dynamics (one section, returns COO arrays)");
  m.def("jac_dynamics_position_section", &jac_dynamics_position_section,
        "Batch Jacobian for position dynamics (one section, returns COO arrays)");
  m.def("jac_dynamics_mass_section", &jac_dynamics_mass_section,
        "Batch Jacobian for mass dynamics (one section, returns COO arrays)");
}
