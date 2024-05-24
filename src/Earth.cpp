// Copyright (c) 2022 Interstellar Technologies Inc.
// All rights reserved.

#include "Earth.hpp"

#include <Eigen/Core>
#include <iostream>
#include <utility>

using Eigen::Vector3d;
using std::abs;
using std::atan;
using std::atan2;
using std::cos;
using std::sin;
using std::sqrt;
using std::tan;

const double Earth::mu = 3.986004418e14;
const double Earth::omega_earth_rps = 7.2921151467e-5;
const double Earth::Ra = 6378137.0;
const double Earth::f = 1.0 / 298.257223563;
const double Earth::Rb = Ra * (1.0 - f);
const double Earth::e2 = (Ra * Ra - Rb * Rb) / Ra / Ra;
const double Earth::ep2 = (Ra * Ra - Rb * Rb) / Rb / Rb;

Vector3d Earth::ecef2geodetic(Vector3d pos_ecef) {
  double p = sqrt(pos_ecef(0) * pos_ecef(0) + pos_ecef(1) * pos_ecef(1));
  double theta = atan2(pos_ecef(2) * Ra, p * Rb);

  double lat_rad =
      atan2(pos_ecef(2) + ep2 * Rb * (sin(theta) * sin(theta) * sin(theta)),
            p - e2 * Ra * (cos(theta) * cos(theta) * cos(theta)));
  double lon_rad = atan2(pos_ecef(1), pos_ecef(0));
  double N = Ra / sqrt(1.0 - e2 * sin(lat_rad) * sin(lat_rad));
  double alt = p / cos(lat_rad) - N;
  Vector3d out(lat_rad, lon_rad, alt);
  return out;
}

Vector3d Earth::geodetic2ecef(Vector3d geodetic) {
  double N = Ra / sqrt(1.0 - e2 * sin(geodetic(0)) * sin(geodetic(0)));

  double x = (N + geodetic(2)) * cos(geodetic(0)) * cos(geodetic(1));
  double y = (N + geodetic(2)) * cos(geodetic(0)) * sin(geodetic(1));
  double z = (N * (1.0 - e2) + geodetic(2)) * sin(geodetic(0));
  Vector3d out(x, y, z);
  return out;
}

// calculate geodesic distance using Vincenty's formulae
// from : https://github.com/sus304/ForRocket
std::pair<double, double> Earth::distance_vincenty(Vector3d observer_LLH,
                                                   Vector3d target_LLH) {
  // Input: [lat, lon, alt], [lat, lon, alt] ([rad, m])
  // Output: downrange [m], Azimuth start->end [rad]

  int itr_limit = 100;

  if (observer_LLH(0) == target_LLH(0) && observer_LLH(1) == target_LLH(1)) {
    return std::make_pair(0.0, 0.0);
  }

  double lat1 = observer_LLH(0);
  double lon1 = observer_LLH(1);
  double lat2 = target_LLH(0);
  double lon2 = target_LLH(1);

  double U1 = atan((1.0 - f) * tan(lat1));
  double U2 = atan((1.0 - f) * tan(lat2));
  double diff_lon = lon2 - lon1;

  double sin_sigma = 0.0;
  double cos_sigma = 0.0;
  double sigma = 0.0;
  double sin_alpha = 0.0;
  double cos_alpha = 0.0;
  double cos_2sigma_m = 0.0;
  double coeff = 0.0;

  double lamda = diff_lon;
  for (int i = 0; i < itr_limit; ++i) {
    sin_sigma = pow2(cos(U2) * sin(lamda)) +
                pow2(cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(lamda));
    sin_sigma = sqrt(sin_sigma);
    cos_sigma = sin(U1) * sin(U2) + cos(U1) * cos(U2) * cos(lamda);
    sigma = atan2(sin_sigma, cos_sigma);

    sin_alpha = cos(U1) * cos(U2) * sin(lamda) / sin_sigma;
    cos_alpha = sqrt(1.0 - pow2(sin_alpha));

    cos_2sigma_m = cos_sigma - 2.0 * sin(U1) * sin(U2) / pow2(cos_alpha);

    coeff =
        f / 16.0 * pow2(cos_alpha) * (4.0 + f * (4.0 - 3.0 * pow2(cos_alpha)));
    double lamda_itr = lamda;
    lamda = diff_lon +
            (1.0 - coeff) * f * sin_alpha *
                (sigma + coeff * sin_sigma *
                             (cos_2sigma_m +
                              coeff * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m)));

    if (abs(lamda - lamda_itr) < 1e-12) {
      break;
    }  // TODO( ): 収束しなかった時の処理追加
  }

  double u_squr = pow2(cos_alpha) * (pow2(Ra) - pow2(Rb)) / pow2(Rb);
  double A = 1.0 + u_squr / 16384.0 *
                       (4096.0 +
                        u_squr * (-768.0 + u_squr * (320.0 - 175.0 * u_squr)));
  double B = u_squr / 1024.0 *
             (256.0 + u_squr * (-128.0 + u_squr * (74.0 - 47.0 * u_squr)));
  double delta_sigma =
      B * sin_sigma *
      (cos_2sigma_m +
       0.25 * B *
           (cos_sigma * (-1.0 + 2.0 * pow2(cos_2sigma_m)) -
            (1.0 / 6.0) * B * cos_2sigma_m * (-3.0 + 4.0 * pow2(sin_sigma)) *
                (-3.0 + 4.0 * pow2(cos_2sigma_m))));

  double downrange = Rb * A * (sigma - delta_sigma);
  double alpha1 =
      atan2(cos(U2) * sin(lamda),
            (cos(U1) * sin(U2) -
             sin(U1) * cos(U2) * cos(lamda)));  // observer to target azimuth
  double alpha2 =
      atan2(cos(U1) * sin(lamda),
            (-sin(U1) * cos(U2) +
             cos(U1) * sin(U2) * cos(lamda)));  // target to observer azimuth
  return std::make_pair(downrange, alpha1);
}
