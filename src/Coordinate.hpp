// Copyright (c) 2022 Interstellar Technologies Inc.
// All rights reserved.

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Earth.hpp"

#ifndef SRC_COORDINATE_HPP_
#define SRC_COORDINATE_HPP_

class Coordinate {
 public:
  static Eigen::Vector3d ecef2eci(Eigen::Vector3d xyz_ecef, double t);
  static Eigen::Vector3d eci2ecef(Eigen::Vector3d xyz_eci, double t);
  static Eigen::Vector3d vel_ecef2eci(Eigen::Vector3d vel_ecef,
                                      Eigen::Vector3d pos_ecef, double t);
  static Eigen::Vector3d vel_eci2ecef(Eigen::Vector3d vel_eci,
                                      Eigen::Vector3d pos_eci, double t);
  static Eigen::Quaterniond quat_eci2ecef(double t);
  static Eigen::Quaterniond quat_ecef2eci(double t);
  static Eigen::Quaterniond quat_ecef2ned(Eigen::Vector3d pos_ecef);
  static Eigen::Quaterniond quat_ned2ecef(Eigen::Vector3d pos_ecef);
  static Eigen::Quaterniond quat_eci2ned(Eigen::Vector3d pos_eci, double t);
  static Eigen::Quaterniond quat_ned2eci(Eigen::Vector3d pos_eci, double t);
  static Eigen::Quaterniond quat_ned2body(Eigen::Quaterniond quat_eci2body,
                                          Eigen::Vector3d pos_eci, double t);
  static Eigen::Quaterniond quat_from_euler_deg(double az_deg, double el_deg,
                                                double ro_deg);
  static Eigen::Vector3d euler_from_quat(Eigen::Quaterniond q);
  static Eigen::Vector3d euler_from_dcm(Eigen::Matrix3d C);
  static Eigen::Matrix3d dcm_from_quat(Eigen::Quaterniond q);
  static Eigen::Quaterniond quat_from_dcm(Eigen::Matrix3d C);
  static Eigen::Matrix3d dcm_from_thrustvector(Eigen::Vector3d thrustvec_eci,
                                               Eigen::Vector3d pos_eci);
  static Eigen::Quaterniond quat_from_thrustvector(
      Eigen::Vector3d thrustvec_eci, Eigen::Vector3d pos_eci);
  static Eigen::Matrix<double, 6, 1> orbital_elements(Eigen::Vector3d pos_eci,
                                                      Eigen::Vector3d vel_eci);
  static Eigen::Vector3d pos_from_orbital_elements(
      Eigen::Matrix<double, 6, 1> elem);
  static Eigen::Vector3d vel_from_orbital_elements(
      Eigen::Matrix<double, 6, 1> elem);
};

#endif  // SRC_COORDINATE_HPP_
