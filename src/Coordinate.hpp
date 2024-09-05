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
