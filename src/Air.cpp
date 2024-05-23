// Copyright (c) 2022 Interstellar Technologies Inc.
// All rights reserved.

#include "Air.hpp"

const double Air::Rstar = 8314.32;
const double Air::g0 = 9.80665;
const double Air::r0 = 6356766.0;
const std::vector<double> Air::hb = {0.0,     11000.0,  20000.0, 32000.0,
                                     47000.0, 51000.0,  71000.0, 86000.0,
                                     91000.0, 110000.0, 120000.0};
const std::vector<double> Air::lmb = {-0.0065, 0.0,     0.001,  0.0028,
                                      0.0,     -0.0028, -0.002, 0.0,
                                      0.0025,  0.012,   0.012};
const std::vector<double> Air::tmb = {288.15,   216.65, 216.65, 228.65,
                                      270.65,   270.65, 214.65, 186.8673,
                                      186.8673, 240.0,  360.0};
const std::vector<double> Air::pb = {101325.0, 22632.0,   5474.9,   868.02,
                                     110.91,   66.939,    3.9564,   0.37338,
                                     0.15381,  7.1042e-3, 2.5382e-3};
const std::vector<double> Air::mb = {28.9644, 28.9644, 28.9644, 28.9644,
                                     28.9644, 28.9644, 28.9644, 28.9522,
                                     28.89,   27.27,   26.20};

double Air::geopotential_altitude(double geometric_altitude) {
  double h;
  if (geometric_altitude < 86000.0)
    h = 1.0 * (r0 * geometric_altitude) / (r0 + geometric_altitude);
  else
    h = geometric_altitude;
  return h;
}

AirParams Air::us76_params(double altitude) {
  int k = 0;
  for (int i = 0; i < hb.size(); i++) {
    if (altitude >= hb[i])
      k = i;
  }

  AirParams p;
  p.Hb = hb[k];
  p.Lmb = lmb[k];
  p.Tmb = tmb[k];
  p.Pb = pb[k];
  p.R = Rstar / mb[k];
  return p;
}

double Air::temperature(double h) {
  AirParams p = us76_params(h);

  if (h <= 91000.0) {
    return p.Tmb + p.Lmb * (h - p.Hb);
  } else if (h <= 110000.0) {
    double Tc = 263.1905;
    double A = -76.3232;
    double a = -19942.9;
    return Tc + A * sqrt(1.0 - (h - 91000.0) * (h - 91000.0) / a / a);
  } else if (h <= 120000.0) {
    return p.Tmb + p.Lmb * (h - p.Hb);
  } else {
    double Tinf = 1000.0;
    double xi = (h - p.Hb) * (r0 + p.Hb) / (r0 + h);
    return Tinf - (Tinf - p.Tmb) * exp(-0.01875e-3 * xi);
  }
}

double Air::pressure(double h) {
  AirParams p = us76_params(h);

  if (std::abs(p.Lmb) > 1.0e-6) {
    return p.Pb * pow((p.Tmb + p.Lmb * (h - p.Hb)) / p.Tmb, -g0 / p.Lmb / p.R);
  } else {
    return p.Pb * exp(g0 / p.R * (p.Hb - h) / p.Tmb);
  }
}

double Air::density(double h) {
  AirParams p = us76_params(h);
  double T = temperature(h);
  double P = pressure(h);
  return P / p.R / T;
}

double Air::speed_of_sound(double h) {
  AirParams p = us76_params(h);
  double T = temperature(h);
  return sqrt(1.4 * p.R * T);
}
