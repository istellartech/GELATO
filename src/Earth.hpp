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
