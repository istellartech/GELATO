// Copyright (c) 2022 Interstellar Technologies Inc.
// All rights reserved.

#include "iip.hpp"

#include "Coordinate.hpp"

using Eigen::Vector3d;
using std::asin;
using std::atan2;
using std::sqrt;
using std::tan;

Vector3d posLLH_IIP_FAA(Vector3d posECEF_, Vector3d velECEF_) {
  int n_iter = 5;

  // (v)-(A): The distance frome the center of the Earth ellipsoid to the launch
  // point (the initial approximation of r_k1, k=1)
  double r_k1 = Earth::Rb;

  // (v)-(B): The radial distance from the geocenter to the launch vehicle
  // position
  Vector3d posECI_init_ = Coordinate::ecef2eci(posECEF_, 0.0);
  double r0 = posECI_init_.norm();
  if (r0 < r_k1)  // then tha launch vehicle position is below the Earth's
                  // surface and an impact point cannot be computed
    return Vector3d::Zero();  // no solution

  // (v)-(C): The inertial velocity compoents
  Vector3d velECI_init_ = Coordinate::vel_ecef2eci(velECEF_, posECEF_, 0.0);
  // (v)-(D): The magnitude of the inertial velocity vector
  double v0 = velECI_init_.norm();

  // (v)-(E): The eccentricity of the trajectory ellipse multiplied by the
  // cosine of the eccentric anomaly at epoch
  double eps_cos = (r0 * v0 * v0 / Earth::mu) - 1.0;
  if (eps_cos >=
      1.0)  // then the trajectory orbit is not elliptical, but is hyperbolic or
            // parabolic, and an impact point cannot be computed
    return Vector3d::Zero();  // no solution

  // (v)-(F): The semi-major axis of the trajectory ellipse
  double a_t = r0 / (1 - eps_cos);

  // (v)-(G): The eccentricity of the trajectory ellipse multiplied by the sine
  // of the eccentric anomaly at epoch
  double eps_sin = posECI_init_.dot(velECI_init_) / sqrt(Earth::mu * a_t);

  // (v)-(H): The eccentricity of the trajectory ellipse squared
  double eps2 = eps_cos * eps_cos + eps_sin * eps_sin;
  if (sqrt(eps2) <= 1.0 &&
      a_t * (1 - sqrt(eps2)) - Earth::Ra >=
          0.0)  // then the trajectory perigee height is positive and an impact
                // point cannot be computed
    return Vector3d::Zero();  // no solution

  double eps_k_cos, eps_k_sin, delta_eps_k_cos, delta_eps_k_sin;
  double fseries_2, gseries_2, Ek, Fk, Gk, r_k2, r_k1_tmp;

  for (int i = 0; i < n_iter; i++) {
    // (v)-(I): The eccentricity of the trajectory ellipse multiplied by the
    // cosine of the eccentric anomaly at impact
    eps_k_cos = (a_t - r_k1) / a_t;

    // (v)-(J): The eccentricity of the trajectory ellipse multiplied by the
    // sine of the eccentric anomaly at impact
    if ((eps2 - eps_k_cos * eps_k_cos) <
        0)  // then the trajectory orbit does not intersect the Earth's surface
            // and an impact point cannot be computed
      return Vector3d::Zero();  // no solution
    eps_k_sin = -sqrt(eps2 - eps_k_cos * eps_k_cos);

    // (v)-(K): The cosine of the difference between the eccentric anomaly at
    // impact and epoch
    delta_eps_k_cos = (eps_k_cos * eps_cos + eps_k_sin * eps_sin) / eps2;

    // (v)-(L): The sine of the difference between the eccentric anomaly at
    // impact and epoch
    delta_eps_k_sin = (eps_k_sin * eps_cos - eps_k_cos * eps_sin) / eps2;

    // (v)-(M): The f-series expansion of Kepler's equations
    fseries_2 = (delta_eps_k_cos - eps_cos) / (1 - eps_cos);
    // (v)-(N): The g-series expansion of Kepler's equations
    gseries_2 = (delta_eps_k_sin + eps_sin - eps_k_sin) *
                sqrt(a_t * a_t * a_t / Earth::mu);

    // (v)-(O): The E,F,G coordinates at impact
    Ek = fseries_2 * posECI_init_[0] + gseries_2 * velECI_init_[0];
    Fk = fseries_2 * posECI_init_[1] + gseries_2 * velECI_init_[1];
    Gk = fseries_2 * posECI_init_[2] + gseries_2 * velECI_init_[2];

    // (v)-(P): The approximated distance from the geocenter to the launch
    // vehicle position at impact
    r_k2 = Earth::Ra /
           sqrt((Earth::e2 / (1 - Earth::e2)) * (Gk / r_k1) * (Gk / r_k1) + 1);

    // (v)-(Q): Substituting and repeating
    r_k1_tmp = r_k1;
    r_k1 = r_k2;
  }

  // (v)-(Q): check convergence
  if (abs(r_k1_tmp - r_k2) >
      1.0)  // then the iterative solution does not converge and an impact point
            // does not meet the accuracy tolerance
    return Vector3d::Zero();  // no solution

  // (v)-(R): The difference between the eccentric anomaly at impact and epoch
  double delta_eps = atan2(delta_eps_k_sin, delta_eps_k_cos);

  // (v)-(S): The time of flight from epoch to impact
  double time_sec =
      (delta_eps + eps_sin - eps_k_sin) * sqrt(a_t * a_t * a_t / Earth::mu);

  // (v)-(T): The geocentric latitude at impact
  double phi_impact_tmp = asin(Gk / r_k2);
  // (v)-(U): The geodetic latitude at impact
  double phi_impact = atan2(tan(phi_impact_tmp), 1.0 - Earth::e2);
  // (v)-(V): The East longitude at impact
  double lambda_impact = atan2(Fk, Ek) - Earth::omega_earth_rps * time_sec;

  // finish: convert to posECEF_IIP_
  // posECEF_IIP_ = coord_utils.CoordUtils([rad2deg(phi_impact),
  // rad2deg(lammda_impact), 0]).ecef_origin

  Vector3d posIIP(phi_impact, lambda_impact, 0.0);
  return posIIP;
}
