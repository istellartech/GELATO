#
# The MIT License
#
# Copyright (c) 2022 Interstellar Technologies Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import numpy as np
from numpy import tan, arcsin, arctan2, sqrt


def posLLH_IIP_FAA(posECEF_, velECEF_, fill_na=True, n_iter=5):
    a = 6378137
    mu = 3.986004418e14
    f = 1.0 / 298.257223563
    omega_earth = 7.2921151467e-5
    b = a * (1.0 - f)
    e2 = 2.0 * f - f * f
    omegaVec = np.array([0.0, 0.0, omega_earth])
    if fill_na:
        no_solution = np.zeros(3)
    else:
        no_solution = np.full(3, np.nan)

    # (v)-(A): The distance from the center of the Earth ellipsoid to the launch point
    # (the initial approximation of r_k1, k=1)
    r_k1 = b

    # (v)-(B): The radial distance from the geocenter to the launch vehicle position
    posECI_init_ = posECEF_
    r0 = np.linalg.norm(posECI_init_)
    if r0 < r_k1:
        # then the launch vehicle position is below the Earth's surface
        return no_solution  # no solution

    # (v)-(C): The inertial velocity components
    velECI_init_ = velECEF_ + np.cross(omegaVec, posECEF_)
    # (v)-(D): The magnitude of the inertial velocity vector
    v0 = np.linalg.norm(velECI_init_)

    # (v)-(E): The eccentricity of the trajectory ellipse multiplied by the cosine of
    # the eccentric anomaly at epoch
    eps_cos = (r0 * v0**2 / mu) - 1
    if eps_cos >= 1:
        # then the trajectory orbit is not elliptical, but is hyperbolic or parabolic
        return no_solution  # no solution

    # (v)-(F): The semi-major axis of the trajectory ellipse
    a_t = r0 / (1 - eps_cos)

    # (v)-(G): The eccentricity of the trajectory ellipse multiplied by the sine of the
    # eccentric anomaly at epoch
    eps_sin = np.dot(posECI_init_, velECI_init_) / sqrt(mu * a_t)

    # (v)-(H): The eccentricity of the trajectory ellipse squared
    eps2 = eps_cos**2 + eps_sin**2
    if (sqrt(eps2) <= 1) and (a_t * (1 - sqrt(eps2)) - a >= 0):
        # then the trajectory perigee height is positive
        return no_solution  # no solution

    for i in range(n_iter):
        # (v)-(I): The eccentricity of the trajectory ellipse multiplied by the cosine
        # of the eccentric anomaly at impact
        eps_k_cos = (a_t - r_k1) / a_t

        # (v)-(J): The eccentricity of the trajectory ellipse multiplied by the sine of
        # the eccentric anomaly at impact
        if eps2 - eps_k_cos**2 < 0:
            # then the trajectory orbit does not intersect the Earth's surface
            return no_solution  # no solution
        eps_k_sin = -sqrt(eps2 - eps_k_cos**2)

        # (v)-(K): The cosine of the difference between the eccentric anomaly at impact
        # and epoch
        delta_eps_k_cos = (eps_k_cos * eps_cos + eps_k_sin * eps_sin) / eps2

        # (v)-(L): The sine of the difference between the eccentric anomaly at impact
        # and epoch
        delta_eps_k_sin = (eps_k_sin * eps_cos - eps_k_cos * eps_sin) / eps2

        # (v)-(M): The f-series expansion of Kepler's equations
        fseries_2 = (delta_eps_k_cos - eps_cos) / (1 - eps_cos)
        # (v)-(N): The g-series expansion of Kepler's equations
        gseries_2 = (delta_eps_k_sin + eps_sin - eps_k_sin) * sqrt(a_t**3 / mu)

        # (v)-(O): The E,F,G coordinates at impact
        Ek = fseries_2 * posECI_init_[0] + gseries_2 * velECI_init_[0]
        Fk = fseries_2 * posECI_init_[1] + gseries_2 * velECI_init_[1]
        Gk = fseries_2 * posECI_init_[2] + gseries_2 * velECI_init_[2]

        # (v)-(P): The approximated distance from the geocenter to the launch vehicle
        # position at impact
        r_k2 = a / sqrt((e2 / (1 - e2)) * (Gk / r_k1) ** 2 + 1)

        # (v)-(Q): Substituting and repeating
        r_k1_tmp = r_k1
        r_k1 = r_k2

    # (v)-(Q): check convergence
    if np.abs(r_k1_tmp - r_k2) > 1:
        # then the iterative solution does not converge and an impact point does not
        # meet the accuracy tolerance
        return no_solution  # no solution

    # (v)-(R): The difference between the eccentric anomaly at impact and epoch
    delta_eps = arctan2(delta_eps_k_sin, delta_eps_k_cos)

    # (v)-(S): The time of flight from epoch to impact
    time_sec = (delta_eps + eps_sin - eps_k_sin) * sqrt(a_t**3 / mu)

    # (v)-(T): The geocentric latitude at impact
    phi_impact_tmp = arcsin(Gk / r_k2)
    # (v)-(U): The geodetic latitude at impact
    phi_impact = arctan2(tan(phi_impact_tmp), 1 - e2)
    # (v)-(V): The East longitude at impact
    lambda_impact = arctan2(Fk, Ek) - omega_earth * time_sec

    return np.array([phi_impact, lambda_impact, 0.0]) * 180.0 / np.pi
