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
from math import atan2, asin, degrees
import pandas as pd
from utils_c import *
from PSfunctions import *
from USStandardAtmosphere_c import *
from coordinate_c import *
from tools.IIP import posLLH_IIP_FAA
from tools.downrange import distance_vincenty

def output_result(xdict, unitdict, tx_res, tu_res, pdict):
    """Returns DataFrame that contains optimization results.

    Args:
        xdict (dict) : dict of calculation result
        unitdict (dict) : dict of units of the state vector
        tx_res (ndarray) : time of nodes including initial points
        tu_res (ndarray) : time of LGR nodes
        pdict (dict) : dict of parameters

    Returns:
        DataFrame : time history of state vectors and other values

    """

    N = len(tx_res)

    unit_mass = unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_u = unitdict["u"]
    unit_t = unitdict["t"]

    mass_ = xdict["mass"] * unit_mass
    pos_ = xdict["position"].reshape(-1, 3) * unit_pos
    vel_ = xdict["velocity"].reshape(-1, 3) * unit_vel
    quat_ = xdict["quaternion"].reshape(-1, 4)

    u_ = xdict["u"].reshape(-1, 3) * unit_u

    out = {
        "event": [""] * N,
        "time": tx_res.round(6),
        "stage": [""] * N,
        "section": np.zeros(N, dtype="i4"),
        "thrust": np.zeros(N),
        "mass": mass_,
        "lat": np.zeros(N),
        "lon": np.zeros(N),
        "lat_IIP": np.zeros(N),
        "lon_IIP": np.zeros(N),
        "downrange": np.zeros(N),
        "altitude": np.zeros(N),
        "altitude_apogee": np.zeros(N),
        "altitude_perigee": np.zeros(N),
        "inclination": np.zeros(N),
        "argument_perigee": np.zeros(N),
        "lon_ascending_node": np.zeros(N),
        "true_anomaly": np.zeros(N),
        "pos_ECI_X": pos_[:, 0],
        "pos_ECI_Y": pos_[:, 1],
        "pos_ECI_Z": pos_[:, 2],
        "vel_ECI_X": vel_[:, 0],
        "vel_ECI_Y": vel_[:, 1],
        "vel_ECI_Z": vel_[:, 2],
        "vel_ground_NED_X": np.zeros(N),
        "vel_ground_NED_Y": np.zeros(N),
        "vel_ground_NED_Z": np.zeros(N),
        "quat_ECI2BODY_0": quat_[:, 0],
        "quat_ECI2BODY_1": quat_[:, 1],
        "quat_ECI2BODY_2": quat_[:, 2],
        "quat_ECI2BODY_3": quat_[:, 3],
        "accel_BODY_X": np.zeros(N),
        "aero_BODY_X": np.zeros(N),
        "heading_NED2BODY": np.zeros(N),
        "pitch_NED2BODY": np.zeros(N),
        "roll_NED2BODY": np.zeros(N),
        "vel_inertial": norm(vel_, axis=1),
        "flightpath_vel_inertial_geocentric": np.zeros(N),
        "azimuth_vel_inertial_geocentric": np.zeros(N),
        "thrust_direction_ECI_X": np.zeros(N),
        "thrust_direction_ECI_Y": np.zeros(N),
        "thrust_direction_ECI_Z": np.zeros(N),
        "rate_BODY_X": np.interp(tx_res, tu_res, u_[:, 0]),
        "rate_BODY_Y": np.interp(tx_res, tu_res, u_[:, 1]),
        "rate_BODY_Z": np.interp(tx_res, tu_res, u_[:, 2]),
        "vel_ground": np.zeros(N),
        "vel_air": np.zeros(N),
        "AOA_total": np.zeros(N),
        "AOA_pitch": np.zeros(N),
        "AOA_yaw": np.zeros(N),
        "dynamic_pressure": np.zeros(N),
        "Q_alpha": np.zeros(N),
        "M": np.zeros(N),
    }

    section = 0
    out["event"][0] = pdict["params"][0]["name"]

    for i in range(N):

        mass = mass_[i]
        pos = pos_[i]
        vel = vel_[i]
        quat = normalize(quat_[i])
        t = tx_res[i]

        out["section"][i] = section
        out["stage"][i] = pdict["params"][section]["rocketStage"]
        thrust_vac_n = pdict["params"][section]["thrust"]
        massflow = pdict["params"][section]["massflow"]
        airArea_m2 = pdict["params"][section]["reference_area"]
        nozzleArea_m2 = pdict["params"][section]["nozzle_area"]
        if (
            i
            >= pdict["ps_params"].index_start_u(section)
            + pdict["ps_params"].nodes(section)
            + section
        ):
            out["event"][i] = pdict["params"][section + 1]["name"]
            section += 1

        pos_llh = eci2geodetic(pos, t)
        altitude_m = geopotential_altitude(pos_llh[2])
        out["lat"][i], out["lon"][i], out["altitude"][i] = pos_llh
        out["downrange"][i] = distance_vincenty(
            pdict["LaunchCondition"]["lat"],
            pdict["LaunchCondition"]["lon"],
            pos_llh[0],
            pos_llh[1],
        )

        elem = orbital_elements(pos, vel)
        out["altitude_apogee"][i] = elem[0] * (1.0 + elem[1]) - 6378137
        out["altitude_perigee"][i] = elem[0] * (1.0 - elem[1]) - 6378137
        (
            out["inclination"][i],
            out["lon_ascending_node"][i],
            out["argument_perigee"][i],
            out["true_anomaly"][i],
        ) = elem[2:6]

        vel_ground_ecef = vel_eci2ecef(vel, pos, t)
        vel_ground_ned = quatrot(quat_ecef2nedg(eci2ecef(pos, t)), vel_ground_ecef)
        (
            out["vel_ground_NED_X"][i],
            out["vel_ground_NED_Y"][i],
            out["vel_ground_NED_Z"][i],
        ) = vel_ground_ned
        vel_ned = quatrot(quat_eci2nedg(pos, t), vel)
        vel_air_ned = vel_ground_ned - wind_ned(altitude_m, pdict["wind_table"])
        out["vel_ground"][i] = norm(vel_ground_ecef)

        out["azimuth_vel_inertial_geocentric"][i] = degrees(
            atan2(vel_ned[1], vel_ned[0])
        )
        out["flightpath_vel_inertial_geocentric"][i] = degrees(
            asin(-vel_ned[2] / norm(vel_ned))
        )

        q = 0.5 * norm(vel_air_ned) ** 2 * airdensity_at(altitude_m)
        out["dynamic_pressure"][i] = q

        aoa_all_deg = (
            angle_of_attack_all_rad(pos, vel, quat, t, pdict["wind_table"])
            * 180.0
            / np.pi
        )
        aoa_ab_deg = (
            angle_of_attack_ab_rad(pos, vel, quat, t, pdict["wind_table"])
            * 180.0
            / np.pi
        )

        out["AOA_total"][i] = aoa_all_deg
        out["Q_alpha"][i] = aoa_all_deg * q
        out["AOA_pitch"][i], out["AOA_yaw"][i] = aoa_ab_deg

        thrustdir_eci = quatrot(conj(quat), np.array([1.0, 0.0, 0.0]))
        (
            out["thrust_direction_ECI_X"][i],
            out["thrust_direction_ECI_Y"][i],
            out["thrust_direction_ECI_Z"][i],
        ) = thrustdir_eci
        euler = euler_from_quat(quat_nedg2body(quat, pos, t))
        out["heading_NED2BODY"][i] = euler[0]
        out["pitch_NED2BODY"][i] = euler[1]
        out["roll_NED2BODY"][i] = euler[2]

        #####
        rho = airdensity_at(altitude_m)
        p = airpressure_at(altitude_m)

        # 対気速度

        pos_ecef = eci2ecef(pos, t)
        vel_ecef = vel_eci2ecef(vel, pos, t)
        vel_wind_ned = wind_ned(altitude_m, pdict["wind_table"])

        vel_wind_eci = quatrot(quat_nedg2eci(pos, t), vel_wind_ned)
        vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci
        mach_number = norm(vel_air_eci) / speed_of_sound(altitude_m)
        out["M"][i] = mach_number
        airAxialForce_coeff = np.interp(
            mach_number, pdict["ca_table"][:, 0], pdict["ca_table"][:, 1]
        )
        out["vel_air"][i] = norm(vel_air_eci)

        ret = np.zeros(11)

        aero_n_eci = (
            0.5
            * rho
            * norm(vel_air_eci)
            * -vel_air_eci
            * airArea_m2
            * airAxialForce_coeff
        )
        aero_n_body = quatrot(quat, aero_n_eci)

        thrust_n = thrust_vac_n - nozzleArea_m2 * p
        out["thrust"][i] = thrust_n
        thrustdir_eci = quatrot(conj(quat), np.array([1.0, 0.0, 0.0]))
        thrust_n_eci = thrustdir_eci * thrust_n
        gravity_eci = gravity(pos)
        out["aero_BODY_X"][i] = aero_n_body[0]
        out["accel_BODY_X"][i] = (thrust_n + aero_n_body[0]) / mass

        out["lat_IIP"][i], out["lon_IIP"][i], _ = posLLH_IIP_FAA(
            pos_ecef, vel_ecef, False
        )

        acc_eci = gravity_eci + (thrust_n_eci + aero_n_eci) / mass

        #####

    return pd.DataFrame(out)
