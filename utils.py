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
from numpy.linalg import norm
from math import sin,cos,asin,atan2,sqrt,radians,degrees
from copy import deepcopy
from USStandardAtmosphere import *
from coordinate import *
from numba import jit

@jit('f8(f8,f8,f8,f8,f8)',nopython=True)
def haversine(lon1, lat1, lon2, lat2, r):

    # convert decimal degrees to radians 
    lon1 = radians(lon1)
    lat1 = radians(lat1)
    lon2 = radians(lon2)
    lat2 = radians(lat2)

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * r * asin(sqrt(a)) 

@jit('f8(f8,f8,f8,f8,f8,f8)',nopython=True)
def distance_on_earth(x_km, y_km, z_km, lon0, lat0, time):
    radius_km = 6378.142
    angular_velocity_rps = 0.729211586e-4

    r_km = norm(np.array([x_km, y_km, z_km]))
    latitude = asin(z_km/r_km)
    longitude = atan2(y_km,x_km) - angular_velocity_rps*time
    
    return haversine(lon0, lat0, degrees(longitude), degrees(latitude), radius_km)

@jit('f8[:](f8,f8[:,:])',nopython=True)
def wind_ned(altitude_m, wind_data):
    wind = np.zeros(3)
    wind[0] = np.interp(altitude_m, wind_data[:,0], wind_data[:,1])
    wind[1] = np.interp(altitude_m, wind_data[:,0], wind_data[:,2])
    wind[2] = 0.0
    return wind

@jit(nopython=True)
def angle_of_attack_all_rad(pos_eci, vel_eci, quat, t, wind):

    thrust_dir_eci = quatrot(conj(quat), np.array([1.0, 0.0, 0.0]))
    
    pos_llh = ecef2geodetic(pos_eci[0],pos_eci[1],pos_eci[2])
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

@jit(nopython=True)
def angle_of_attack_ab_rad(pos_eci, vel_eci, quat, t, wind):

    pos_llh = ecef2geodetic(pos_eci[0],pos_eci[1],pos_eci[2])
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

@jit(nopython=True)
def dynamic_pressure_pa(pos_eci, vel_eci, t, wind):

    pos_llh = ecef2geodetic(pos_eci[0],pos_eci[1],pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])
    rho = airdensity_at(altitude_m)
        
    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)
    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci
    
    return 0.5 * vel_air_eci.dot(vel_air_eci) * rho



def jac_fd(con, xdict, pdict, unitdict, condition):
    """
    Calculate jacobian by finite-difference method(forward difference).

    Note that this function is slow, because this function do not use sparse matrix.

    Args:
        con(function) : object function
        xdict : variable arg for con
        pdict : parameter arg for con
        unitdict : unit arg for con
        conditiondict : condition arg for con

    Returns:
        jac(dict(ndarray)) : dict of jacobian matrix

    """

    jac = {}
    dx = 1.0e-8
    g_base = con(xdict, pdict, unitdict, condition)
    if hasattr(g_base,"__len__"):
        nRows = len(g_base)
    else:
        nRows = 1
    for key,val in xdict.items():
        jac[key] = np.zeros((nRows, val.size))
        for i in range(val.size):
            xdict_p = deepcopy(xdict)
            xdict_p[key][i] += dx
            g_p = con(xdict_p, pdict, unitdict, condition)
            jac[key][:,i] = (g_p - g_base) / dx

    return jac