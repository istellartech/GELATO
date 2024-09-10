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

import math
from math import sqrt
import numpy as np

# U.S. Standard Atmosphere 1976


def geopotential_altitude(z):
    """Calculates geopotential altitude.
    Refer to US1976 Equation 18.
    Args:
        z (float64) : geometric altitude [m]
    Returns:
        float64 : geopotential altitude (z <= 86000),
        geopotential altitude (z > 86000) [m]
    """

    r0 = 6356766
    g0 = 9.80665

    if z < 86000:
        return 1.0 * (r0 * z) / (r0 + z)
    else:
        return z


def us_standard_atmosphere_params_at(altitude_m):
    """Returns parameters at each reference levels.
    The values of temperature gradient for the layer from 91 to 110 km
    and above 120km are dummy values for calculation of the approximate
    pressure.
    Args:
        altitude_m (float64) : geopotential altitude when Z <= 86000,
        geometric altitude when Z > 86000 [m]
    Returns:
        ndarray: parameters (altitude[m], temperature gradient[K/m],
        temperature[K], pressure[Pa], gas constant[J/kg-K],
        gravity acceleration[m/s2])
    """

    GRAVITY_ACC_CONST = 9.80665
    Rstar = 8314.32

    #  b <  7:[Hb, Lmb, Tmb, Pb, R]
    #  b >= 7:[Zb, Lb,  Tb,  Pb, R]
    PARAMS = [
        [0.0, -0.0065, 288.15, 101325.0, Rstar / 28.9644],
        [11000.0, 0.0, 216.65, 22632.0, Rstar / 28.9644],
        [20000.0, 0.001, 216.65, 5474.9, Rstar / 28.9644],
        [32000.0, 0.0028, 228.65, 868.02, Rstar / 28.9644],
        [47000.0, 0.0, 270.65, 110.91, Rstar / 28.9644],
        [51000.0, -0.0028, 270.65, 66.939, Rstar / 28.9644],
        [71000.0, -0.002, 214.65, 3.9564, Rstar / 28.9644],
        [86000.0, 0.0, 186.8673, 0.37338, Rstar / 28.9522],
        [91000.0, 0.0025, 186.8673, 0.15381, Rstar / 28.89],  # Lb is dummy
        [110000.0, 0.012, 240.0, 7.1042e-3, Rstar / 27.27],
        [120000.0, 0.012, 360.0, 2.5382e-3, Rstar / 26.20],  # Lb is dummy
    ]

    k = 0
    for i in range(len(PARAMS)):
        if altitude_m >= PARAMS[i][0]:
            k = i

    return np.append(np.array(PARAMS[k]), GRAVITY_ACC_CONST)


def airtemperature_at(altitude_m):
    """Air temperature at the given altitude.
    Refer to US1976 Section 1.2.5.
    Args:
        altitude_m (float64) : geopotential altitude when Z <= 86000,
        geometric altitude when Z > 86000 [m]
    Returns:
        float64: temperature[K]
    """

    air_params = us_standard_atmosphere_params_at(altitude_m)
    HAL = air_params[0]
    LR = air_params[1]
    T0 = air_params[2]
    P0 = air_params[3]
    R = air_params[4]
    GRAVITY = air_params[5]

    if altitude_m <= 91000:
        return T0 + LR * (altitude_m - HAL)
    elif altitude_m <= 110000:
        Tc = 263.1905
        A = -76.3232
        a = -19942.9
        return Tc + A * sqrt(1.0 - (altitude_m - 91000) ** 2 / a**2)
    elif altitude_m <= 120000:
        return T0 + LR * (altitude_m - HAL)
    else:
        Tinf = 1000.0
        r0 = 6356766
        xi = (altitude_m - HAL) * (r0 + HAL) / (r0 + altitude_m)
        return Tinf - (Tinf - T0) * np.exp(-0.01875 * 1e-3 * xi)


def airpressure_at(altitude_m):
    """Air pressure at the given altitude.
    Refer to US1976 Section 1.3.1.
    The value above 86000m is an approximation by fitting.
    Args:
        altitude_m (float64) : geopotential altitude when Z <= 86000,
        geometric altitude when Z > 86000 [m]
    Returns:
        float64: pressure[Pa]
    """

    air_params = us_standard_atmosphere_params_at(altitude_m)
    HAL = air_params[0]
    LR = air_params[1]
    T0 = air_params[2]
    P0 = air_params[3]
    R = air_params[4]
    GRAVITY = air_params[5]

    air_temperature = airtemperature_at(altitude_m)

    if math.fabs(LR) > 1.0e-10:
        air_pressure = P0 * ((T0 + LR * (altitude_m - HAL)) / T0) ** (GRAVITY / -LR / R)
    else:
        air_pressure = P0 * math.exp(GRAVITY / R * (HAL - altitude_m) / T0)

    return air_pressure


def airdensity_at(altitude_m):
    """Air density at the given altitude.
    Refer to US1976 Section 1.3.4.
    The value above 86000m is an approximation by fitting.
    Args:
        altitude_m (float64) : geopotential altitude when Z <= 86000,
        geometric altitude when Z > 86000 [m]
    Returns:
        float64: mass density[kg/m3]
    """

    air_params = us_standard_atmosphere_params_at(altitude_m)
    HAL = air_params[0]
    LR = air_params[1]
    T0 = air_params[2]
    P0 = air_params[3]
    R = air_params[4]
    GRAVITY = air_params[5]

    air_temperature = airtemperature_at(altitude_m)

    if math.fabs(LR) > 1.0e-10:
        air_pressure = P0 * ((T0 + LR * (altitude_m - HAL)) / T0) ** (GRAVITY / -LR / R)
    else:
        air_pressure = P0 * math.exp(GRAVITY / R * (HAL - altitude_m) / T0)

    air_density = air_pressure / R / air_temperature
    return air_density


def speed_of_sound(altitude_m):
    """Speed of sound at the given altitude.
    Refer to US1976 Section 1.3.10.
    Args:
        altitude_m (float64) : geopotential altitude when Z <= 86000,
        geometric altitude when Z > 86000 [m]
    Returns:
        float64: speed of sound[m/s]
    """

    air_params = us_standard_atmosphere_params_at(altitude_m)
    HAL = air_params[0]
    LR = air_params[1]
    T0 = air_params[2]
    P0 = air_params[3]
    R = air_params[4]
    GRAVITY = air_params[5]

    air_temperature = airtemperature_at(altitude_m)

    return sqrt(1.4 * R * air_temperature)
