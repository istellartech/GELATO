// Copyright (c) 2022 Interstellar Technologies Inc.
// All rights reserved.

#include <Eigen/Core>
#include <cmath>
#include <vector>

#ifndef SRC_AIR_HPP_
#define SRC_AIR_HPP_

struct AirParams {
  double Hb, Lmb, Tmb, Pb, R;
};

class Air {
  static const double Rstar, g0, r0;
  static const std::vector<double> hb, lmb, tmb, pb, mb;

 public:
  static double geopotential_altitude(double geometric_altitude);
  static AirParams us76_params(double altitude);
  static double temperature(double altitude);
  static double pressure(double altitude);
  static double density(double altitude);
  static double speed_of_sound(double altitude);
};

#endif  // SRC_AIR_HPP_
