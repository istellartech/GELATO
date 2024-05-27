#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#include "Coordinate.hpp"
#include "Earth.hpp"
#include "gravity.hpp"

namespace py = pybind11;

#ifndef SRC_WRAPPER_COORDINATE_HPP_
#define SRC_WRAPPER_COORDINATE_HPP_


using vec3d = Eigen::Matrix<double, 3, 1>;
using vec4d = Eigen::Matrix<double, 4, 1>;
using vecXd = Eigen::Matrix<double, -1, 1>;
using matXd = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
using mat3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

vec4d quatmult(vec4d q, vec4d p) {
  vec4d qp;
  qp[0] = q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3];
  qp[1] = q[0] * p[1] + q[1] * p[0] + q[2] * p[3] - q[3] * p[2];
  qp[2] = q[0] * p[2] - q[1] * p[3] + q[2] * p[0] + q[3] * p[1];
  qp[3] = q[0] * p[3] + q[1] * p[2] - q[2] * p[1] + q[3] * p[0];
  return qp;
}

vec4d conj(vec4d q) {
  vec4d q_conj;
  q_conj[0] = q[0];
  q_conj[1] = -q[1];
  q_conj[2] = -q[2];
  q_conj[3] = -q[3];
  return q_conj;
}

vecXd normalize(vecXd v) { return v / v.norm(); }

vec3d quatrot(vec4d q, vec3d v) {
  vec4d vq;
  vq[0] = 0;
  vq[1] = v[0];
  vq[2] = v[1];
  vq[3] = v[2];
  vec4d rvq = quatmult(conj(q), quatmult(vq, q));
  return rvq.tail(3);
}

Eigen::Matrix3d dcm_from_quat(vec4d q) {
  Eigen::Matrix3d C;
  C(0, 0) = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3];
  C(0, 1) = 2 * (q[1] * q[2] + q[0] * q[3]);
  C(0, 2) = 2 * (q[1] * q[3] - q[0] * q[2]);

  C(1, 0) = 2 * (q[1] * q[2] - q[0] * q[3]);
  C(1, 1) = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3];
  C(1, 2) = 2 * (q[2] * q[3] + q[0] * q[1]);

  C(2, 0) = 2 * (q[1] * q[3] + q[0] * q[2]);
  C(2, 1) = 2 * (q[2] * q[3] - q[0] * q[1]);
  C(2, 2) = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3];
  return C;
}

vec4d quat_from_dcm(Eigen::Matrix3d C) {
  vec4d q;
  q[0] = 0.5 * sqrt(1 + C(0, 0) + C(1, 1) + C(2, 2));
  q[1] = (C(1, 2) - C(2, 1)) / (4 * q[0]);
  q[2] = (C(2, 0) - C(0, 2)) / (4 * q[0]);
  q[3] = (C(0, 1) - C(1, 0)) / (4 * q[0]);
  return q;
}

vec3d ecef2geodetic(double x, double y, double z) {
  vec3d pos_ecef(x, y, z);
  vec3d geodetic = Earth::ecef2geodetic(pos_ecef);
  geodetic[0] = geodetic[0] * 180.0 / M_PI;
  geodetic[1] = geodetic[1] * 180.0 / M_PI;
  return geodetic;
}

vec3d geodetic2ecef(double lat, double lon, double alt) {
  vec3d geodetic(lat * M_PI / 180.0, lon * M_PI / 180.0, alt);
  return Earth::geodetic2ecef(geodetic);
}

vec3d ecef2eci(vec3d xyz_in, double t) {
  return Coordinate::ecef2eci(xyz_in, t);
}

vec3d eci2ecef(vec3d xyz_in, double t) {
  return Coordinate::eci2ecef(xyz_in, t);
}

vec3d vel_ecef2eci(vec3d vel_ecef, vec3d pos_ecef, double t) {
  return Coordinate::vel_ecef2eci(vel_ecef, pos_ecef, t);
}

vec3d vel_eci2ecef(vec3d vel_eci, vec3d pos_eci, double t) {
  return Coordinate::vel_eci2ecef(vel_eci, pos_eci, t);
}

vec4d quat_eci2ecef(double t) {
  Eigen::Quaterniond q = Coordinate::quat_eci2ecef(t);
  return vec4d(q.w(), q.x(), q.y(), q.z());
}

vec4d quat_ecef2eci(double t) {
  Eigen::Quaterniond q = Coordinate::quat_ecef2eci(t);
  return vec4d(q.w(), q.x(), q.y(), q.z());
}

vec4d quat_ecef2nedg(vec3d pos_ecef) {
  Eigen::Quaterniond q = Coordinate::quat_ecef2ned(pos_ecef);
  return vec4d(q.w(), q.x(), q.y(), q.z());
}

vec4d quat_nedg2ecef(vec3d pos_ecef) {
  Eigen::Quaterniond q = Coordinate::quat_ned2ecef(pos_ecef);
  return vec4d(q.w(), q.x(), q.y(), q.z());
}

vec4d quat_eci2nedg(vec3d pos_eci, double t) {
  Eigen::Quaterniond q = Coordinate::quat_eci2ned(pos_eci, t);
  return vec4d(q.w(), q.x(), q.y(), q.z());
}

vec4d quat_nedg2eci(vec3d pos_eci, double t) {
  Eigen::Quaterniond q = Coordinate::quat_ned2eci(pos_eci, t);
  return vec4d(q.w(), q.x(), q.y(), q.z());
}

vec4d quat_from_euler(double az, double el, double ro) {
  Eigen::Quaterniond q = Coordinate::quat_from_euler_deg(az, el, ro);
  return vec4d(q.w(), q.x(), q.y(), q.z());
}

vec3d gravity(vec3d pos) { return gravityECI(pos); }

vec4d quat_nedg2body(vec4d quat_eci2body, vec3d pos_eci, double t) {
  vec4d q = quat_eci2nedg(pos_eci, t);
  return quatmult(conj(q), quat_eci2body);
}

vec3d euler_from_quat(vec4d q) {
  Eigen::Quaterniond q_(q[0], q[1], q[2], q[3]);
  vec3d euler = Coordinate::euler_from_quat(q_);
  return euler * 180.0 / M_PI;
}

vec3d euler_from_dcm(Eigen::Matrix3d C) {
  Eigen::Matrix3d C_(C.data());
  vec3d euler = Coordinate::euler_from_dcm(C_);
  return euler * 180.0 / M_PI;
}

Eigen::Matrix3d dcm_from_thrustvector(vec3d pos_eci, vec3d thrustvec_eci) {
  Eigen::Matrix3d C = Coordinate::dcm_from_thrustvector(thrustvec_eci, pos_eci);
  return C;
}

vec3d eci2geodetic(vec3d pos_eci, double t) {
  vec3d pos_ecef = Coordinate::eci2ecef(pos_eci, t);
  vec3d geodetic = Earth::ecef2geodetic(pos_ecef);
  geodetic[0] = geodetic[0] * 180.0 / M_PI;
  geodetic[1] = geodetic[1] * 180.0 / M_PI;
  return geodetic;
}

Eigen::Matrix<double, 6, 1> orbital_elements(vec3d pos_eci, vec3d vel_eci) {
  Eigen::Matrix<double, 6, 1> elem =
      Coordinate::orbital_elements(pos_eci, vel_eci);
  elem[2] = elem[2] * 180.0 / M_PI;
  elem[3] = elem[3] * 180.0 / M_PI;
  elem[4] = elem[4] * 180.0 / M_PI;
  elem[5] = elem[5] * 180.0 / M_PI;
  return elem;
}

double distance_vincenty(double lat_origin, double lon_origin, 
                         double lat_target, double lon_target) {
  
  std::pair <double, double>dist_azimuth = Earth::distance_vincenty(
      Eigen::Vector3d(lat_origin * M_PI / 180.0, lon_origin * M_PI / 180.0, 0.0),
      Eigen::Vector3d(lat_target * M_PI / 180.0, lon_target * M_PI / 180.0, 0.0));

  return dist_azimuth.first;
}

#endif  // SRC_WRAPPER_COORDINATE_HPP_
