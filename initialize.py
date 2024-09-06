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

import sys
import numpy as np
from scipy.interpolate import interp1d
from lib.utils import *
from lib.PSfunctions import *
from lib.USStandardAtmosphere import *
from lib.coordinate import *
from tools.plot_output import display_6DoF
from output_result import output_result


def dynamics_init(x, u, t, param, zlt, wind, ca):
    """Full equation of motion.

    Args:
        x (ndarray) : state vector
        (mass[kg], posion in ECI frame[m],
          inertial velocity in ECI frame[m/s],
          quaternion from ECI to body frame)
        u (ndarray) : rate in body frame[deg/s]
        t (float64) : time[s]
        param (ndarray) : parameters
        (thrust at vacuum[N], massflow rate[kg/s],
          reference area[m2], (unused), nozzle area[m2])
        zlt (boolean) : True when zero-lift turn mode
        wind (ndarray) : wind table
        ca (ndarray) : CA table

    Returns:
        differential of the state vector

    """

    mass = x[0]
    pos_eci = x[1:4]
    vel_eci = x[4:7]
    quat_eci2body = x[7:11]
    d_roll = u[0]
    d_pitch = u[1]
    d_yaw = u[2]

    pos_llh = ecef2geodetic(pos_eci[0], pos_eci[1], pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])
    rho = airdensity_at(altitude_m)
    p = airpressure_at(altitude_m)

    # 対気速度

    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)

    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci
    mach_number = norm(vel_air_eci) / speed_of_sound(altitude_m)

    thrust_vac_n = param[0]
    massflow = param[1]
    airArea_m2 = param[2]
    airAxialForce_coeff = np.interp(mach_number, ca[:, 0], ca[:, 1])
    nozzleArea_m2 = param[4]

    ret = np.zeros(11)

    aero_n_eci = (
        0.5 * rho * norm(vel_air_eci) * -vel_air_eci * airArea_m2 * airAxialForce_coeff
    )

    thrust_n = thrust_vac_n - nozzleArea_m2 * p
    if zlt:
        thrustdir_eci = normalize(vel_air_eci)
    else:
        thrustdir_eci = quatrot(conj(quat_eci2body), np.array([1.0, 0.0, 0.0]))
    thrust_n_eci = thrustdir_eci * thrust_n
    gravity_eci = gravity(pos_eci)

    acc_eci = gravity_eci + (thrust_n_eci + aero_n_eci) / mass

    omega_rps_body = np.deg2rad(np.array([0.0, d_roll, d_pitch, d_yaw]))
    d_quat = 0.5 * quatmult(quat_eci2body, omega_rps_body)

    ret[0] = -massflow
    ret[1:4] = vel_eci
    ret[4:7] = acc_eci
    ret[7:11] = d_quat

    return ret


def rocket_simulation(x_init, u_table, pdict, t_init, t_out, dt=0.1):
    """Simulates the motion of the rocket and output time history.

    Args:
        x_init (ndarray) : initial values of the state vector
        u_table (ndarray) : time history of the rate
        pdict (dict) : dict of parameters
        t_init (float64) : initial time
        t_out (ndarray or float64) : time(s) used for output
        dt (float64) : integration interval

    Returns:
        ndarray : time history of the state vector
    """

    x_map = [x_init]
    t_map = [t_init]
    u_map = [[0.0, 0.0, 0.0]]

    x = x_init
    t = t_init

    if hasattr(t_out, "__iter__"):
        t_final = t_out[-1]
    else:
        t_final = t_out
    event_index = -1
    zlt = False
    param = np.zeros(5)

    wind = pdict["wind_table"]
    ca = pdict["ca_table"]

    while t < t_final:

        tn = t + dt

        if event_index < len(pdict["params"]) - 1:
            if tn > pdict["params"][event_index + 1]["time"]:
                event_index += 1
                param[0] = pdict["params"][event_index]["thrust"]
                param[1] = pdict["params"][event_index]["massflow"]
                param[2] = pdict["params"][event_index]["reference_area"]
                param[4] = pdict["params"][event_index]["nozzle_area"]
                x[0] -= pdict["params"][event_index]["mass_jettison"]

        u = np.array([np.interp(t, u_table[:, 0], u_table[:, i + 1]) for i in range(3)])
        x = integrate_runge_kutta_4d(
            lambda xa, ta: dynamics_init(xa, u, ta, param, zlt, wind, ca), x, t, dt
        )
        t = t + dt

        if pdict["params"][event_index]["attitude"] == "zero-lift-turn":
            x[7:11] = zerolift_turn_correct(x, t, wind)
        x[7:11] = normalize(x[7:11])

        t_map.append(t)
        x_map.append(x)
        u_map.append(u)

    x_map = np.array(x_map)
    u_map = np.array(u_map)
    x_out = np.vstack([np.interp(t_out, t_map, x_map[:, i]) for i in range(len(x))]).T
    u_out = np.vstack([np.interp(t_out, t_map, u_map[:, i]) for i in range(len(u))]).T

    return x_out, u_out


def zerolift_turn_correct(x, t, wind=np.zeros((2, 3))):
    """Corrects attitude quaternion during zero-lift turn.

    The body axis is corrected to be perpendicular to the air velocity.
    The roll angle is corrected to be zero.

    Args:
        x (ndarray) : state vector
        t (float64) : time
        wind (ndarray) : wind table

    Returns:
        ndarray : corrected quaternion

    """

    # mass = x[0]
    pos_eci = x[1:4]
    vel_eci = x[4:7]
    # quat_eci2body = x[7:11]

    pos_llh = ecef2geodetic(pos_eci[0], pos_eci[1], pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])

    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)

    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci

    xb_dir = normalize(vel_air_eci)
    yb_dir = normalize(np.cross(vel_air_eci, pos_eci))
    zb_dir = np.cross(xb_dir, yb_dir)

    q0 = 0.5 * sqrt(1.0 + xb_dir[0] + yb_dir[1] + zb_dir[2])
    q1 = 0.25 / q0 * (yb_dir[2] - zb_dir[1])
    q2 = 0.25 / q0 * (zb_dir[0] - xb_dir[2])
    q3 = 0.25 / q0 * (xb_dir[1] - yb_dir[0])

    return normalize(np.array((q0, q1, q2, q3)))


def integrate_euler(function, x, t, dt):
    """Integration of function by first-order Euler method."""
    return x + function(x, t) * dt


def integrate_runge_kutta_4d(function, x, t, dt):
    """Integration of function by fourth-order Runge-Kutta method."""
    k1 = function(x, t)
    k2 = function(x + dt / 2.0 * k1, t + dt / 2.0)
    k3 = function(x + dt / 2.0 * k2, t + dt / 2.0)
    k4 = function(x + dt * k3, t + dt)
    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0 * dt


def initialize_xdict_6DoF_2(
    x_init, pdict, condition, unitdict, mode="LGR", dt=0.005, flag_display=True
):
    """
    Initialize and set xdict by solving equation of motion.

    Args:
        x_init (dict) : initial values of state(mass, position, velocity, quaternion)
        pdict (dict) : calculation parameters
        condition (dict) : flight condition parameters
        unitdict (dict) : unit of the state (use for normalizing)
        mode (str) : calculation mode (LG, LGR or LGL)
        dt (double) : time step for integration
        flag_display (bool) : plot and display initial state if true

    Returns:
        xdict (dict) : initial values of variables for NLP

    """
    xdict = {}
    num_sections = pdict["num_sections"]

    time_nodes = np.array([])
    time_x_nodes = np.array([])

    for i in range(num_sections):
        to = pdict["params"][i]["time"]
        tf = pdict["params"][i]["timeFinishAt"]
        tau = pdict["ps_params"].tau(i)

        if mode == "LG" or mode == "LGR":
            tau_x = np.hstack((-1.0, tau))
        else:
            tau_x = tau

        time_nodes = np.hstack((time_nodes, tau * (tf - to) / 2.0 + (tf + to) / 2.0))
        time_x_nodes = np.hstack(
            (time_x_nodes, tau_x * (tf - to) / 2.0 + (tf + to) / 2.0)
        )

    time_knots = np.array([e["time"] for e in pdict["params"]])
    xdict["t"] = (time_knots / unitdict["t"]).ravel()

    # 現在位置と目標軌道から、適当に目標位置・速度を決める

    if condition["rf_m"] is not None:
        r_final = condition["rf_m"]
    else:
        if condition["hp_m"] is None or condition["ha_m"] is None:
            print("DESTINATION ORBIT NOT DETERMINED!!")
            sys.exit()
    print(r_final)

    u_nodes = np.vstack(
        [
            [
                [
                    0.0,
                    pdict["params"][i]["pitchrate_init"],
                    pdict["params"][i]["yawrate_init"],
                ]
            ]
            * pdict["ps_params"].nodes(i)
            for i in range(num_sections)
        ]
    )
    xdict["u"] = (u_nodes / unitdict["u"]).ravel()

    u_table = np.hstack((time_nodes.reshape(-1, 1), u_nodes))

    x_nodes, _ = rocket_simulation(
        x_init, u_table, pdict, time_nodes[0], time_x_nodes, dt
    )

    xdict["mass"] = x_nodes[:, 0] / unitdict["mass"]
    xdict["position"] = (x_nodes[:, 1:4] / unitdict["position"]).ravel()
    xdict["velocity"] = (x_nodes[:, 4:7] / unitdict["velocity"]).ravel()
    xdict["quaternion"] = (x_nodes[:, 7:11]).ravel()

    if flag_display:
        display_6DoF(output_result(xdict, unitdict, time_x_nodes, time_nodes, pdict))
    return xdict


def initialize_xdict_6DoF_from_file(
    x_ref, pdict, condition, unitdict, mode="LGL", flag_display=True
):
    """
    Initialize and set xdict by interpolating reference values.

    Args:
        x_ref (DataFrame) : time history of state(mass, position, velocity, quaternion)
        pdict (dict) : calculation parameters
        condition (dict) : flight condition parameters
        unitdict (dict) : unit of the state (use for normalizing)
        mode (str) : calculation mode (LG, LGR or LGL)
        flag_display (bool) : plot and display initial state if true

    Returns:
        xdict (dict) : initial values of variables for NLP

    """
    xdict = {}
    num_sections = pdict["num_sections"]

    time_nodes = np.array([])
    time_x_nodes = np.array([])

    for i in range(num_sections):
        to = pdict["params"][i]["time"]
        tf = pdict["params"][i]["timeFinishAt"]
        tau = pdict["ps_params"].tau(i)

        if mode == "LG" or mode == "LGR":
            tau_x = np.hstack((-1.0, tau))
        else:
            tau_x = tau

        time_nodes = np.hstack((time_nodes, tau * (tf - to) / 2.0 + (tf + to) / 2.0))
        time_x_nodes = np.hstack(
            (time_x_nodes, tau_x * (tf - to) / 2.0 + (tf + to) / 2.0)
        )

    time_knots = np.array([e["time"] for e in pdict["params"]])
    xdict["t"] = (time_knots / unitdict["t"]).ravel()

    xdict["mass"] = (
        interp1d(x_ref["time"], x_ref["mass"], fill_value="extrapolate")(time_x_nodes)
        / unitdict["mass"]
    ).ravel()
    xdict["position"] = (
        interp1d(
            x_ref["time"],
            x_ref[["pos_ECI_X", "pos_ECI_Y", "pos_ECI_Z"]],
            axis=0,
            fill_value="extrapolate",
        )(time_x_nodes)
        / unitdict["position"]
    ).ravel()
    xdict["velocity"] = (
        interp1d(
            x_ref["time"],
            x_ref[["vel_ECI_X", "vel_ECI_Y", "vel_ECI_Z"]],
            axis=0,
            fill_value="extrapolate",
        )(time_x_nodes)
        / unitdict["velocity"]
    ).ravel()
    xdict["quaternion"] = (
        interp1d(
            x_ref["time"],
            x_ref[
                [
                    "quat_ECI2BODY_0",
                    "quat_ECI2BODY_1",
                    "quat_ECI2BODY_2",
                    "quat_ECI2BODY_3",
                ]
            ],
            axis=0,
            fill_value="extrapolate",
        )(time_x_nodes)
    ).ravel()
    xdict["u"] = (
        interp1d(
            x_ref["time"],
            x_ref[["rate_BODY_X", "rate_BODY_Y", "rate_BODY_Z"]],
            axis=0,
            fill_value="extrapolate",
        )(time_nodes)
        / unitdict["u"]
    ).ravel()

    if flag_display:
        display_6DoF(output_result(xdict, unitdict, time_x_nodes, time_nodes, pdict))
    return xdict
