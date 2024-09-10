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

from math import sin, cos, asin, atan2, sqrt, radians, degrees
import numpy as np
from numpy.linalg import norm
from .USStandardAtmosphere import *
from .coordinate import *


def haversine(lon1, lat1, lon2, lat2, r):
    """Calculates surface distance with haversine formula.
    This function is DEPRECATED; use Vincenty formula instead.
    Args:
        lon1 (float64) : longitude of start point [deg]
        lat1 (float64) : latitude of start point [deg]
        lon2 (float64) : longitude of end point [deg]
        lat2 (float64) : latitude of end point [deg]
        r (float64) : radius of the sphere
    Returns:
        float64 : surface distance between start and end points
    """

    # convert decimal degrees to radians
    lon1 = radians(lon1)
    lat1 = radians(lat1)
    lon2 = radians(lon2)
    lat2 = radians(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def distance_on_earth(x_km, y_km, z_km, lon0, lat0, time):
    """Calculates surface distance of two points on Earth with
    haversine formula.
    This function is DEPRECATED; use Vincenty formula instead.
    Args:
        x_km (float64) : x-coordinates of target point in ECEF frame [km]
        y_km (float64) : y-coordinates of target point in ECEF frame [km]
        z_km (float64) : z-coordinates of target point in ECEF frame [km]
        lon0 (float64) : longitude of reference point [deg]
        lat0 (float64) : latitude of reference point [deg]
        time (float64) : time [s]
    Returns:
        float64 : surface distance between reference and target points[km]
    """
    radius_km = 6378.142
    angular_velocity_rps = 0.729211586e-4

    r_km = norm(np.array([x_km, y_km, z_km]))
    latitude = asin(z_km / r_km)
    longitude = atan2(y_km, x_km) - angular_velocity_rps * time

    return haversine(lon0, lat0, degrees(longitude), degrees(latitude), radius_km)


def wind_ned(altitude_m, wind_data):
    """Get wind speed in NED frame by interpolation."""
    wind = np.zeros(3)
    wind[0] = np.interp(altitude_m, wind_data[:, 0], wind_data[:, 1])
    wind[1] = np.interp(altitude_m, wind_data[:, 0], wind_data[:, 2])
    wind[2] = 0.0
    return wind


def angle_of_attack_all_rad(pos_eci, vel_eci, quat, t, wind):
    """Calculates total angle of attack.
    Args:
        pos_eci (ndarray) : position in ECI frame [m]
        vel_eci (ndarray) : inertial velocity in ECI frame [m/s]
        quat (ndarray) : coordinate transformation quaternion from ECI
          to body frame
        t (float64) : time [s]
        wind (ndarray) : wind table
    Returns:
        float64 : angle of attack [rad]
    """

    thrust_dir_eci = quatrot(conj(quat), np.array([1.0, 0.0, 0.0]))

    pos_llh = ecef2geodetic(pos_eci[0], pos_eci[1], pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])

    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)

    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci

    c_alpha = normalize(vel_air_eci).dot(normalize(thrust_dir_eci))

    if c_alpha >= 1.0 or norm(vel_air_eci) < 0.001:
        return 0.0
    else:
        return acos(c_alpha)


def angle_of_attack_all_array_rad(pos_eci, vel_eci, quat, t, wind):
    """Array version of angle_of_attack_all_rad."""
    alpha = np.zeros(pos_eci.shape[0])
    for i in range(pos_eci.shape[0]):
        alpha[i] = angle_of_attack_all_rad(pos_eci[i], vel_eci[i], quat[i], t[i], wind)
    return alpha


def angle_of_attack_ab_rad(pos_eci, vel_eci, quat, t, wind):
    """Calculates pitch and yaw angles of attack.
    Args:
        pos_eci (ndarray) : position in ECI frame [m]
        vel_eci (ndarray) : inertial velocity in ECI frame [m/s]
        quat (ndarray) : coordinate transformation quaternion from ECI
          to body frame
        t (float64) : time [s]
        wind (ndarray) : wind table
    Returns:
        ndarray : pitch and yaw angles of attack [rad]
    """

    pos_llh = ecef2geodetic(pos_eci[0], pos_eci[1], pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])

    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)

    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci

    vel_air_body = quatrot(quat, vel_air_eci)

    if vel_air_body[0] < 0.001:
        return np.zeros(2)
    else:
        alpha_z = atan2(vel_air_body[2], vel_air_body[0])
        alpha_y = atan2(vel_air_body[1], vel_air_body[0])
        return np.array((alpha_z, alpha_y))


def dynamic_pressure_pa(pos_eci, vel_eci, t, wind):
    """Calculates dynamic pressure.
    Args:
        pos_eci (ndarray) : position in ECI frame [m]
        vel_eci (ndarray) : inertial velocity in ECI frame [m/s]
        t (float64) : time [s]
        wind (ndarray) : wind table
    Returns:
        float64 : dynamic pressure [Pa]
    """

    pos_llh = ecef2geodetic(pos_eci[0], pos_eci[1], pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])
    rho = airdensity_at(altitude_m)

    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)
    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci

    return 0.5 * vel_air_eci.dot(vel_air_eci) * rho


def dynamic_pressure_array_pa(pos_eci, vel_eci, t, wind):
    """Array version of dynamic_pressure_pa."""
    q = np.zeros(pos_eci.shape[0])
    for i in range(pos_eci.shape[0]):
        q[i] = dynamic_pressure_pa(pos_eci[i], vel_eci[i], t[i], wind)
    return q


def q_alpha_pa_rad(pos_eci, vel_eci, quat, t, wind):
    """Calculates Q_alpha."""
    alpha = angle_of_attack_all_rad(pos_eci, vel_eci, quat, t, wind)
    q = dynamic_pressure_pa(pos_eci, vel_eci, t, wind)
    return q * alpha


def q_alpha_array_pa_rad(pos_eci, vel_eci, quat, t, wind):
    """Array version of q_alpha_pa_rad."""
    qa = np.zeros(pos_eci.shape[0])
    for i in range(pos_eci.shape[0]):
        qa[i] = q_alpha_pa_rad(pos_eci[i], vel_eci[i], quat[i], t[i], wind)
    return qa
