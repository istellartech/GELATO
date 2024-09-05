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
