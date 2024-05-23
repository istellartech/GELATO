// Copyright (c) 2022 Interstellar Technologies Inc.
// All rights reserved.

#include "Coordinate.hpp"

#include <cmath>
#include <iostream>

using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using std::acos;
using std::atan2;
using std::cos;
using std::sin;
using std::sqrt;

Vector3d Coordinate::ecef2eci(Vector3d xyz_ecef, double t) {
  Vector3d out;
  out(0) = xyz_ecef(0) * cos(Earth::omega_earth_rps * t) -
           xyz_ecef(1) * sin(Earth::omega_earth_rps * t);
  out(1) = xyz_ecef(0) * sin(Earth::omega_earth_rps * t) +
           xyz_ecef(1) * cos(Earth::omega_earth_rps * t);
  out(2) = xyz_ecef(2);
  return out;
}

Vector3d Coordinate::eci2ecef(Vector3d xyz_eci, double t) {
  Vector3d out;
  out(0) = xyz_eci(0) * cos(Earth::omega_earth_rps * t) +
           xyz_eci(1) * sin(Earth::omega_earth_rps * t);
  out(1) = -xyz_eci(0) * sin(Earth::omega_earth_rps * t) +
           xyz_eci(1) * cos(Earth::omega_earth_rps * t);
  out(2) = xyz_eci(2);
  return out;
}

Vector3d Coordinate::vel_ecef2eci(Vector3d vel_ecef, Vector3d pos_ecef,
                                  double t) {
  Vector3d pos_eci = ecef2eci(pos_ecef, t);
  Vector3d vel_ground_eci = ecef2eci(vel_ecef, t);
  Vector3d omega_vec(0.0, 0.0, Earth::omega_earth_rps);
  return vel_ground_eci + omega_vec.cross(pos_eci);
}

Vector3d Coordinate::vel_eci2ecef(Vector3d vel_eci, Vector3d pos_eci,
                                  double t) {
  Vector3d omega_vec(0.0, 0.0, Earth::omega_earth_rps);
  return eci2ecef(vel_eci - omega_vec.cross(pos_eci), t);
}

Quaterniond Coordinate::quat_eci2ecef(double t) {
  Quaterniond out(cos(Earth::omega_earth_rps * t / 2.0), 0.0, 0.0,
                  sin(Earth::omega_earth_rps * t / 2.0));
  return out;
}

Quaterniond Coordinate::quat_ecef2eci(double t) {
  return quat_eci2ecef(t).conjugate();
}

Quaterniond Coordinate::quat_ecef2ned(Vector3d pos_ecef) {
  Vector3d geodetic = Earth::ecef2geodetic(pos_ecef);
  double c_hl = cos(geodetic(1) / 2.0);
  double s_hl = sin(geodetic(1) / 2.0);
  double c_hp = cos(geodetic(0) / 2.0);
  double s_hp = sin(geodetic(0) / 2.0);

  double q0 = c_hl * (c_hp - s_hp) / sqrt(2.0);
  double q1 = s_hl * (c_hp + s_hp) / sqrt(2.0);
  double q2 = -c_hl * (c_hp + s_hp) / sqrt(2.0);
  double q3 = s_hl * (c_hp - s_hp) / sqrt(2.0);
  Quaterniond q(q0, q1, q2, q3);
  return q;
}

Quaterniond Coordinate::quat_ned2ecef(Vector3d pos_ecef) {
  return quat_ecef2ned(pos_ecef).conjugate();
}

Quaterniond Coordinate::quat_eci2ned(Vector3d pos_eci, double t) {
  return quat_eci2ecef(t) * quat_ecef2ned(eci2ecef(pos_eci, t));
}

Quaterniond Coordinate::quat_ned2eci(Vector3d pos_eci, double t) {
  return quat_eci2ned(pos_eci, t).conjugate();
}

Quaterniond Coordinate::quat_ned2body(Quaterniond quat_eci2body,
                                      Vector3d pos_eci, double t) {
  return quat_ned2eci(pos_eci, t).conjugate() * quat_eci2body;
}

Quaterniond Coordinate::quat_from_euler_deg(double az_deg, double el_deg,
                                            double ro_deg) {
  double az = az_deg * M_PI / 180.0;
  double el = el_deg * M_PI / 180.0;
  double ro = ro_deg * M_PI / 180.0;
  Quaterniond q = Eigen::AngleAxisd(az, Vector3d::UnitZ()) *
                  Eigen::AngleAxisd(el, Vector3d::UnitY()) *
                  Eigen::AngleAxisd(ro, Vector3d::UnitX());
  return q;
}

Vector3d Coordinate::euler_from_quat(Quaterniond q) {
  Vector3d out = q.toRotationMatrix().eulerAngles(2, 1, 0);
  // EigenのeulerAnglesは値域が[0:pi]x[-pi:pi]x[-pi:pi]なので修正し、
  // [0,2pi]x[-pi/2,pi/2]x[-pi,pi]の範囲に収まるようにする
  if (std::fabs(out[1]) > M_PI / 2.0) {
    if (out[1] > 0.0) {
      out[1] = M_PI - out[1];
    } else {
      out[1] = -M_PI - out[1];
    }
    out[0] = M_PI + out[0];
    out[2] = M_PI + out[2];
  }
  out[0] = std::fmod(out[0], 2.0 * M_PI);
  if (out[0] < 0.0)
    out[0] += 2.0 * M_PI;
  out[2] = std::fmod(out[2] + M_PI, 2.0 * M_PI) - M_PI;
  return out;
}

Vector3d Coordinate::euler_from_dcm(Matrix3d C) {
  Vector3d out = C.transpose().eulerAngles(2, 1, 0);
  // EigenのeulerAnglesは値域が[0:pi]x[-pi:pi]x[-pi:pi]なので修正し、
  // [0,2pi]x[-pi/2,pi/2]x[-pi,pi]の範囲に収まるようにする
  if (std::fabs(out[1]) > M_PI / 2.0) {
    if (out[1] > 0.0) {
      out[1] = M_PI - out[1];
    } else {
      out[1] = -M_PI - out[1];
    }
    out[0] = M_PI + out[0];
    out[2] = M_PI + out[2];
  }
  out[0] = std::fmod(out[0], 2.0 * M_PI);
  if (out[0] < 0.0)
    out[0] += 2.0 * M_PI;
  out[2] = std::fmod(out[2] + M_PI, 2.0 * M_PI) - M_PI;
  return out;
}

Matrix3d Coordinate::dcm_from_quat(Quaterniond q) {
  Matrix3d C = q.toRotationMatrix().transpose();
  return C;
}

Quaterniond Coordinate::quat_from_dcm(Matrix3d C) {
  Quaterniond q(C.transpose());
  return q;
}

Matrix3d Coordinate::dcm_from_thrustvector(Vector3d thrustvec_eci,
                                           Vector3d pos_eci) {
  Vector3d xb_dir = thrustvec_eci.normalized();
  Vector3d yb_dir;
  if (1.0 - xb_dir.dot(pos_eci.normalized()) < 1.0e-10) {
    Vector3d z_eci(0.0, 0.0, 1.0);
    yb_dir = z_eci.cross(xb_dir).normalized();
  } else {
    yb_dir = xb_dir.cross(pos_eci.normalized()).normalized();
  }
  Vector3d zb_dir = xb_dir.cross(yb_dir);
  Matrix3d out;
  out << xb_dir, yb_dir, zb_dir;  // 列ベクトルを行方向に並べる
  return out.transpose();
}

Quaterniond Coordinate::quat_from_thrustvector(Vector3d thrustvec_eci,
                                               Vector3d pos_eci) {
  return quat_from_dcm(dcm_from_thrustvector(thrustvec_eci, pos_eci));
}

Matrix<double, 6, 1> Coordinate::orbital_elements(Vector3d pos_eci,
                                                  Vector3d vel_eci) {
  Vector3d nr = pos_eci.normalized();

  Vector3d c_eci = pos_eci.cross(vel_eci);                 // orbit plane vector
  Vector3d f_eci = vel_eci.cross(c_eci) - Earth::mu * nr;  // Laplace vector

  Vector3d c1_eci = c_eci.normalized();
  Vector3d f1_eci = f_eci.normalized();

  double inclination_rad = acos(c1_eci(2));
  double ascending_node_rad = 0.0;
  double argument_perigee_rad = 0.0;

  if (inclination_rad > 1.0e-10) {
    ascending_node_rad = atan2(c1_eci(0), -c1_eci(1));
    Vector3d n_eci(cos(ascending_node_rad), sin(ascending_node_rad),
                   0.0);  // direction of ascending node
    argument_perigee_rad = acos(n_eci.dot(f1_eci));
    if (f_eci(2) < 0.0) {
      argument_perigee_rad *= -1.0;
    }
  } else {
    ascending_node_rad = 0.0;
    if (f_eci.norm() > 1.0e-10) {
      argument_perigee_rad = atan2(f_eci(1), f_eci(0));
    } else {
      argument_perigee_rad = 0.0;
    }
  }

  double p = c_eci.dot(c_eci) / Earth::mu;
  double e = f_eci.norm() / Earth::mu;
  double a = p / (1.0 - e * e);

  double true_anomaly_rad = acos(f1_eci.dot(nr));
  if (vel_eci.dot(pos_eci) < 0.0) {
    true_anomaly_rad = 2.0 * M_PI - true_anomaly_rad;
  }

  if (ascending_node_rad < 0.0)
    ascending_node_rad += 2.0 * M_PI;
  if (argument_perigee_rad < 0.0)
    argument_perigee_rad += 2.0 * M_PI;
  if (true_anomaly_rad < 0.0)
    true_anomaly_rad += 2.0 * M_PI;

  Matrix<double, 6, 1> out;
  out << a, e, inclination_rad, ascending_node_rad, argument_perigee_rad,
      true_anomaly_rad;
  return out;
}

Vector3d Coordinate::pos_from_orbital_elements(Matrix<double, 6, 1> elem) {
  double a = elem[0];
  double e = elem[1];
  double i = elem[2];
  double O = elem[3];
  double w = elem[4];
  double theta = elem[5];

  double p = a * (1.0 - e * e);
  Vector3d c1_eci(sin(i) * sin(O), -sin(i) * cos(O), cos(i));
  Vector3d f1_eci(cos(O) * cos(w) - sin(O) * cos(i) * sin(w),
                  sin(O) * cos(w) + cos(O) * cos(i) * sin(w), sin(i) * sin(w));
  Vector3d y1_eci = c1_eci.cross(f1_eci);

  return p / (1.0 + e * cos(theta)) *
         (cos(theta) * f1_eci + sin(theta) * y1_eci);
}

Vector3d Coordinate::vel_from_orbital_elements(Matrix<double, 6, 1> elem) {
  double a = elem[0];
  double e = elem[1];
  double i = elem[2];
  double O = elem[3];
  double w = elem[4];
  double theta = elem[5];

  double p = a * (1.0 - e * e);
  Vector3d c1_eci(sin(i) * sin(O), -sin(i) * cos(O), cos(i));
  Vector3d f1_eci(cos(O) * cos(w) - sin(O) * cos(i) * sin(w),
                  sin(O) * cos(w) + cos(O) * cos(i) * sin(w), sin(i) * sin(w));
  Vector3d y1_eci = c1_eci.cross(f1_eci);

  return sqrt(Earth::mu / p) *
         (-sin(theta) * f1_eci + (e + cos(theta)) * y1_eci);
}
