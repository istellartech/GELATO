import numpy as np
from numpy.linalg import norm
from scipy import sparse
from math import sin,cos,asin,atan2,sqrt,radians,degrees
from copy import deepcopy
from numba import jit

@jit('f8[:](f8,f8,f8)',nopython=True)
def ecef2geodetic(x,y,z):
    
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = (a**2 - b**2) / a**2
    ep2 = (a**2 - b**2) / b**2
    
    p = sqrt(x**2 + y**2)
    theta = atan2(z*a, p*b)
    
    lat = atan2(z + ep2 * b * sin(theta)**3, p - e2 * a * cos(theta)**3 )
    lon = atan2(y, x)
    N = a / sqrt(1.0 - e2 * sin(lat)**2)
    alt = p / cos(lat) - N
    
    return np.array((degrees(lat), degrees(lon), alt))

@jit('f8[:](f8,f8,f8)',nopython=True)
def geodetic2ecef(lat, lon, alt):
    
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = (a**2 - b**2) / a**2
    
    N = a / sqrt(1.0 - e2 * sin(radians(lat))**2)

    x = (N + alt) * cos(radians(lat)) * cos(radians(lon))
    y = (N + alt) * cos(radians(lat)) * sin(radians(lon))
    z = (N * (1 - e2) + alt) * sin(radians(lat))
    
    return np.array((x, y, z))

@jit('f8[:](f8,f8,f8)',nopython=True)
def ecef2geodetic_sphere(x, y, z):
    r_Earth = 6378137.0
    lat = degrees(atan2(z,sqrt(x**2+y**2)))
    lon = degrees(atan2(y,x))
    alt = sqrt(x**2+y**2+z**2) - r_Earth
    return np.array((lat, lon, alt))

@jit('f8[:](f8,f8,f8)',nopython=True)
def geodetic2ecef_sphere(lat, lon, alt):
    r_Earth = 6378137.0
    z = (alt + r_Earth) * sin(radians(lat))
    y = (alt + r_Earth) * cos(radians(lat)) * sin(radians(lon))
    x = (alt + r_Earth) * cos(radians(lat)) * cos(radians(lon))
    return np.array((x, y, z))


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