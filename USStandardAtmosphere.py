import math
import numpy as np
from numba import jit
from math import sqrt

#U.S. Standard Atmosphere 1976

# altitude_m : geopotential altitude

@jit(nopython=True)
def geopotential_altitude(z):
    
    r0 = 6356766
    g0 = 9.80665
    
    if z < 86000:
        return 1.0 * (r0 * z) / (r0 + z)
    else:
        return z


@jit(nopython=True)
def us_standard_atmosphere_params_at(altitude_m):
    # ref: https://github.com/istellartech/OpenTsiolkovsky/blob/master/src/air.cpp
    # ref: https://github.com/istellartech/OpenTsiolkovsky/blob/master/src/air.hpp

    
    GRAVITY_ACC_CONST = 9.80665
    R = 287.0531 # specific gas constant for dry air
    Rstar = 8314.32
    
    #  b <  7:[Hb, Lmb, Tmb, Pb, R]
    #  b >= 7:[Zb, Lb,  Tb,  Pb, R]
    PARAMS = [[     0.0, -0.0065, 288.15, 101325.0,  Rstar/28.9644],
              [ 11000.0,  0.0,    216.65, 22632.0,   Rstar/28.9644],
              [ 20000.0,  0.001,  216.65, 5474.9,    Rstar/28.9644],
              [ 32000.0,  0.0028, 228.65, 868.02,    Rstar/28.9644],
              [ 47000.0,  0.0,    270.65, 110.91,    Rstar/28.9644],
              [ 51000.0, -0.0028, 270.65, 66.939,    Rstar/28.9644],
              [ 71000.0, -0.002,  214.65, 3.9564,    Rstar/28.9644],
              [ 86000.0,  0.0,    186.8673, 0.37338, Rstar/28.9522],
              [ 91000.0,  0.0025, 186.8673, 0.15381, Rstar/28.89], # elliptic region, LR is dummy
              [110000.0,  0.012,  240.0,  7.1042e-3, Rstar/27.27],
              [120000.0,  0.012,  360.0,  2.5382e-3, Rstar/26.20]] # exponential region, LR is dummy

    k = 0
    for i in range(len(PARAMS)):
        if altitude_m >= PARAMS[i][0]:
            k = i



    return np.append(np.array(PARAMS[k]),GRAVITY_ACC_CONST)

@jit(nopython=True)
def airtemperature_at(altitude_m):
    
    # ref: https://github.com/istellartech/OpenTsiolkovsky/blob/master/src/air.cpp
    # ref: https://github.com/istellartech/OpenTsiolkovsky/blob/master/src/air.hpp

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
        return Tc + A * sqrt(1.0 - (altitude_m - 91000)**2 / a**2)
    elif altitude_m <= 120000:
        return T0 + LR * (altitude_m - HAL)
    else:
        Tinf = 1000.0
        r0 = 6356766
        xi = (altitude_m - HAL) * (r0 + HAL) / (r0 + altitude_m)
        return Tinf - (Tinf - T0) * np.exp(-0.01875*1e-3*xi)
        

    
@jit(nopython=True)
def airpressure_at(altitude_m):
    
    # ref: https://github.com/istellartech/OpenTsiolkovsky/blob/master/src/air.cpp
    # ref: https://github.com/istellartech/OpenTsiolkovsky/blob/master/src/air.hpp
    
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


@jit(nopython=True)
def airdensity_at(altitude_m):
    
    # ref: https://github.com/istellartech/OpenTsiolkovsky/blob/master/src/air.cpp
    # ref: https://github.com/istellartech/OpenTsiolkovsky/blob/master/src/air.hpp
    
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



@jit(nopython=True)
def speed_of_sound(altitude_m):
    
    # ref: https://github.com/istellartech/OpenTsiolkovsky/blob/master/src/air.cpp
    # ref: https://github.com/istellartech/OpenTsiolkovsky/blob/master/src/air.hpp

    air_params = us_standard_atmosphere_params_at(altitude_m)
    HAL = air_params[0]
    LR = air_params[1]
    T0 = air_params[2]
    P0 = air_params[3]
    R = air_params[4]
    GRAVITY = air_params[5]

    air_temperature = airtemperature_at(altitude_m)

    return sqrt(1.4 * R * air_temperature)
