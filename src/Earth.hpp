// Copyright (c) 2022 Interstellar Technologies Inc.
// All rights reserved.

#include <Eigen/Core>
#include <cmath>
#include <utility>

#ifndef SRC_EARTH_HPP_
#define SRC_EARTH_HPP_

class Earth {
 public:
  static const double mu;
  static const double omega_earth_rps;
  static const double Ra;
  static const double f;
  static const double Rb;
  static const double e2;
  static const double ep2;

  static inline double pow2(double x) { return x * x; }

  static Eigen::Vector3d ecef2geodetic(Eigen::Vector3d pos_ecef);
  static Eigen::Vector3d geodetic2ecef(Eigen::Vector3d geodetic);
  static std::pair<double, double> distance_vincenty(
      Eigen::Vector3d observer_LLH, Eigen::Vector3d target_LLH);
};

#endif  // SRC_EARTH_HPP_
