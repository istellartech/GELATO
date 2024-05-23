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
from numpy.linalg import norm
from numba import jit
from utils import *
from USStandardAtmosphere import *
from coordinate import *
from user_constraints import *
from tools.IIP import posLLH_IIP_FAA
from tools.downrange import distance_vincenty


def equality_init(xdict, pdict, unitdict, condition):
    """Equality constraint about initial conditions."""

    con = []
    mass_ = xdict["mass"]
    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)

    # initial condition
    if condition["OptimizationMode"] != "Payload":
        con.append(mass_[0] - condition["init"]["mass"] / unitdict["mass"])
    con.append(pos_[0] - condition["init"]["position"] / unitdict["position"])
    con.append(vel_[0] - condition["init"]["velocity"] / unitdict["velocity"])
    con.append(quat_[0] - condition["init"]["quaternion"])

    return np.concatenate(con, axis=None)


def equality_jac_init(xdict, pdict, unitdict, condition):
    """Jacobian of equality_init."""

    jac = {}

    if condition["OptimizationMode"] == "Payload":
        jac["position"] = {
            "coo": [
                np.arange(0, 3, dtype="i4"),
                np.arange(0, 3, dtype="i4"),
                np.ones(3),
            ],
            "shape": (10, pdict["M"] * 3),
        }
        jac["velocity"] = {
            "coo": [
                np.arange(3, 6, dtype="i4"),
                np.arange(0, 3, dtype="i4"),
                np.ones(3),
            ],
            "shape": (10, pdict["M"] * 3),
        }
        jac["quaternion"] = {
            "coo": [
                np.arange(6, 10, dtype="i4"),
                np.arange(0, 4, dtype="i4"),
                np.ones(4),
            ],
            "shape": (10, pdict["M"] * 4),
        }

    else:
        jac["mass"] = {
            "coo": [np.zeros(1, dtype="i4"), np.zeros(1, dtype="i4"), np.ones(1)],
            "shape": (11, pdict["M"]),
        }
        jac["position"] = {
            "coo": [
                np.arange(1, 4, dtype="i4"),
                np.arange(0, 3, dtype="i4"),
                np.ones(3),
            ],
            "shape": (11, pdict["M"] * 3),
        }
        jac["velocity"] = {
            "coo": [
                np.arange(4, 7, dtype="i4"),
                np.arange(0, 3, dtype="i4"),
                np.ones(3),
            ],
            "shape": (11, pdict["M"] * 3),
        }
        jac["quaternion"] = {
            "coo": [
                np.arange(7, 11, dtype="i4"),
                np.arange(0, 4, dtype="i4"),
                np.ones(4),
            ],
            "shape": (11, pdict["M"] * 4),
        }

    return jac


def equality_time(xdict, pdict, unitdict, condition):
    """Equality constraint about time of knots."""

    con = []
    unit_t = unitdict["t"]

    t_ = xdict["t"]

    num_sections = pdict["num_sections"]

    # force to fix initial time
    con.append(t_[0] - pdict["params"][0]["time"] / unit_t)
    for i in range(1, num_sections + 1):
        if pdict["params"][i]["time_ref"] in pdict["event_index"].keys():
            i_ref = pdict["event_index"][pdict["params"][i]["time_ref"]]
            con.append(
                t_[i]
                - t_[i_ref]
                - (pdict["params"][i]["time"] - pdict["params"][i_ref]["time"]) / unit_t
            )

    return np.concatenate(con, axis=None)


def equality_jac_time(xdict, pdict, unitdict, condition):
    """Jacobian of equality_time."""

    jac = {}

    data = [1.0]
    row = [0]
    col = [0]

    iRow = 1
    for i in range(1, pdict["num_sections"] + 1):
        if pdict["params"][i]["time_ref"] in pdict["event_index"].keys():
            i_ref = pdict["event_index"][pdict["params"][i]["time_ref"]]
            data.extend([1.0, -1.0])
            row.extend([iRow, iRow])
            col.extend([i, i_ref])
            iRow += 1

    jac["t"] = {
        "coo": [np.array(row, dtype="i4"), np.array(col, dtype="i4"), np.array(data)],
        "shape": (iRow, len(xdict["t"])),
    }

    return jac


def equality_knot_LGR(xdict, pdict, unitdict, condition):
    """Equality constraint about knotting conditions."""

    con = []

    mass_ = xdict["mass"]
    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)

    num_sections = pdict["num_sections"]

    param = np.zeros(5)

    section_sep_list = []
    for key, stage in pdict["RocketStage"].items():
        if stage["separation_at"] is not None:
            section_ig = [
                i
                for i, value in enumerate(pdict["params"])
                if value["name"] == stage["ignition_at"]
            ][0]
            section_sep = [
                i
                for i, value in enumerate(pdict["params"])
                if value["name"] == stage["separation_at"]
            ][0]
            section_sep_list.append(section_sep)

            # mass after separation
            mass_stage = (
                stage["mass_dry"]
                + stage["mass_propellant"]
                + sum([item["mass"] for item in stage["dropMass"].values()])
            )
            index_ig = pdict["ps_params"][section_ig]["index_start"] + section_ig
            index_sep = pdict["ps_params"][section_sep]["index_start"] + section_sep
            con.append(
                mass_[index_ig] - mass_[index_sep] - mass_stage / unitdict["mass"]
            )

    for i in range(1, num_sections):
        a = pdict["ps_params"][i]["index_start"]

        param[0] = pdict["params"][i]["thrust"]
        param[1] = pdict["params"][i]["massflow"]
        param[2] = pdict["params"][i]["reference_area"]
        param[4] = pdict["params"][i]["nozzle_area"]

        # knotting constraints
        mass_init_ = mass_[a + i]
        mass_prev_ = mass_[a + i - 1]
        if not (i in section_sep_list):
            con.append(
                mass_init_
                - mass_prev_
                + pdict["params"][i]["mass_jettison"] / unitdict["mass"]
            )

        pos_init_ = pos_[a + i]
        pos_prev_ = pos_[a + i - 1]
        con.append(pos_init_ - pos_prev_)

        vel_init_ = vel_[a + i]
        vel_prev_ = vel_[a + i - 1]
        con.append(vel_init_ - vel_prev_)

        quat_init_ = quat_[a + i]
        quat_prev_ = quat_[a + i - 1]
        con.append(quat_init_ - quat_prev_)

    return np.concatenate(con, axis=None)


def equality_jac_knot_LGR(xdict, pdict, unitdict, condition):
    """Jacobian of equality_knot."""

    jac = {}

    num_sections = pdict["num_sections"]

    f_center = equality_knot_LGR(xdict, pdict, unitdict, condition)
    nRow = len(f_center)

    jac["mass"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"])}
    jac["position"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["velocity"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["quaternion"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 4)}

    iRow = 0

    section_sep_list = []
    for key, stage in pdict["RocketStage"].items():
        if stage["separation_at"] is not None:
            section_ig = [
                i
                for i, value in enumerate(pdict["params"])
                if value["name"] == stage["ignition_at"]
            ][0]
            section_sep = [
                i
                for i, value in enumerate(pdict["params"])
                if value["name"] == stage["separation_at"]
            ][0]
            section_sep_list.append(section_sep)

            # mass after separation
            index_ig = pdict["ps_params"][section_ig]["index_start"] + section_ig
            index_sep = pdict["ps_params"][section_sep]["index_start"] + section_sep
            jac["mass"]["coo"][0].extend([iRow, iRow])
            jac["mass"]["coo"][1].extend([index_ig, index_sep])
            jac["mass"]["coo"][2].extend([1.0, -1.0])
            iRow += 1

    for i in range(1, num_sections):
        a = pdict["ps_params"][i]["index_start"]

        if not (i in section_sep_list):
            jac["mass"]["coo"][0].extend([iRow, iRow])
            jac["mass"]["coo"][1].extend([a + i - 1, a + i])
            jac["mass"]["coo"][2].extend([-1.0, 1.0])
            iRow += 1

        jac["position"]["coo"][0].extend(list(range(iRow, iRow + 3)))
        jac["position"]["coo"][1].extend(list(range((a + i - 1) * 3, (a + i) * 3)))
        jac["position"]["coo"][2].extend([-1.0] * 3)
        jac["position"]["coo"][0].extend(list(range(iRow, iRow + 3)))
        jac["position"]["coo"][1].extend(list(range((a + i) * 3, (a + i + 1) * 3)))
        jac["position"]["coo"][2].extend([1.0] * 3)
        iRow += 3

        jac["velocity"]["coo"][0].extend(list(range(iRow, iRow + 3)))
        jac["velocity"]["coo"][1].extend(list(range((a + i - 1) * 3, (a + i) * 3)))
        jac["velocity"]["coo"][2].extend([-1.0] * 3)
        jac["velocity"]["coo"][0].extend(list(range(iRow, iRow + 3)))
        jac["velocity"]["coo"][1].extend(list(range((a + i) * 3, (a + i + 1) * 3)))
        jac["velocity"]["coo"][2].extend([1.0] * 3)
        iRow += 3

        jac["quaternion"]["coo"][0].extend(list(range(iRow, iRow + 4)))
        jac["quaternion"]["coo"][1].extend(list(range((a + i - 1) * 4, (a + i) * 4)))
        jac["quaternion"]["coo"][2].extend([-1.0] * 4)
        jac["quaternion"]["coo"][0].extend(list(range(iRow, iRow + 4)))
        jac["quaternion"]["coo"][1].extend(list(range((a + i) * 4, (a + i + 1) * 4)))
        jac["quaternion"]["coo"][2].extend([1.0] * 4)
        iRow += 4

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def equality_6DoF_LGR_terminal(xdict, pdict, unitdict, condition):
    """Equality constraint about terminal condition."""

    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)

    # terminal conditions

    pos_f = pos_[-1] * unit_pos
    vel_f = vel_[-1] * unit_vel

    GMe = 3.986004418e14
    if (
        condition["altitude_perigee"] is not None
        and condition["altitude_apogee"] is not None
    ):
        a = (
            condition["altitude_perigee"] + condition["altitude_apogee"]
        ) / 2.0 + 6378137.0
        rp = condition["altitude_perigee"] + 6378137.0
        vp2 = GMe * (2.0 / rp - 1.0 / a)
        c2_target = rp * rp * vp2  # squared target angular momentum
        e_target = -GMe / 2.0 / a  # target orbit energy
    else:
        c2_target = (condition["radius"] * condition["vel_tangential_geocentric"]) ** 2
        vf_target = condition["vel_tangential_geocentric"] / cos(
            radians(condition["flightpath_vel_inertial_geocentric"])
        )
        e_target = vf_target**2 / 2.0 - GMe / condition["radius"]

    c = np.cross(pos_f, vel_f)
    vf2 = vel_f[0] ** 2 + vel_f[1] ** 2 + vel_f[2] ** 2
    c2 = c[0] ** 2 + c[1] ** 2 + c[2] ** 2
    con.append(
        (vf2 / 2.0 - GMe / norm(pos_f) - e_target) / unit_vel**2
    )  # orbit energy
    con.append((c2 - c2_target) / unit_vel**2 / unit_pos**2)  # angular momentum

    if condition["inclination"] is not None:
        con.append(
            (c[2] - sqrt(c2_target) * cos(radians(condition["inclination"])))
            / unit_vel
            / unit_pos
        )

    return np.concatenate(con, axis=None)


def equality_jac_6DoF_LGR_terminal(xdict, pdict, unitdict, condition):
    """Jacobian of equality_terminal."""

    jac = {}
    dx = 1.0e-8

    f_center = equality_6DoF_LGR_terminal(xdict, pdict, unitdict, condition)

    if hasattr(f_center, "__len__"):
        nRow = len(f_center)
    else:
        nRow = 1

    nCol = pdict["M"] * 3
    jac["position"] = {"coo": [[], [], []], "shape": (nRow, nCol)}
    jac["velocity"] = {"coo": [[], [], []], "shape": (nRow, nCol)}

    for key in ["position", "velocity"]:

        for j in range(nCol - 3, nCol):
            xdict[key][j] += dx
            f_p = equality_6DoF_LGR_terminal(xdict, pdict, unitdict, condition)
            xdict[key][j] -= dx
            jac[key]["coo"][0].extend(list(range(nRow)))
            jac[key]["coo"][1].extend([j] * nRow)
            jac[key]["coo"][2].extend(((f_p - f_center) / dx).tolist())

        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def equality_6DoF_rate(xdict, pdict, unitdict, condition):
    """Equality constraint about angular rate."""

    con = []

    unit_pos = unitdict["position"]

    pos_ = xdict["position"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)

    u_ = xdict["u"].reshape(-1, 3)

    num_sections = pdict["num_sections"]

    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        pos_i_ = pos_[a + i : b + i + 1]
        quat_i_ = quat_[a + i : b + i + 1]
        u_i_ = u_[a:b]

        # rate constraint

        att = pdict["params"][i]["attitude"]

        # attitude hold : angular velocity is zero
        if att in ["hold", "vertical"]:
            con.append(u_i_)

        # kick-turn : pitch rate constant, roll/yaw rate is zero
        elif att == "kick-turn" or att == "pitch":
            con.append(u_i_[:, 0])
            con.append(u_i_[:, 2])
            con.append(u_i_[1:, 1] - u_i_[0, 1])

        # pitch-yaw : pitch/yaw constant, roll ANGLE is zero
        elif att == "pitch-yaw":
            con.append(u_i_[1:, 1] - u_i_[0, 1])
            con.append(u_i_[1:, 2] - u_i_[0, 2])
            con.append(roll_direction_array(pos_i_[1:] * unit_pos, quat_i_[1:]))

        # same-rate : pitch/yaw is the same as previous section, roll ANGLE is zero
        elif att == "same-rate":
            uf_prev = u_[a - 1]
            con.append(u_i_[:, 1] - uf_prev[1])
            con.append(u_i_[:, 2] - uf_prev[2])
            con.append(roll_direction_array(pos_i_[1:] * unit_pos, quat_i_[1:]))

        # zero-lift-turn or free : roll hold
        elif att == "zero-lift-turn" or att == "free":
            con.append(u_i_[:, 0])

        else:
            print("ERROR: UNKNOWN ATTITUDE OPTION! ({})".format(att))
            sys.exit()

    return np.concatenate(con, axis=None)


def equality_jac_6DoF_rate(xdict, pdict, unitdict, condition):
    """Jacobian of equality_rate."""

    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]

    pos_ = xdict["position"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)

    num_sections = pdict["num_sections"]

    f_center = equality_6DoF_rate(xdict, pdict, unitdict, condition)
    nRow = len(f_center)
    jac["position"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["quaternion"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 4)}
    jac["u"] = {"coo": [[], [], []], "shape": (nRow, pdict["N"] * 3)}

    iRow = 0

    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        pos_i_ = pos_[a + i : b + i + 1]
        quat_i_ = quat_[a + i : b + i + 1]

        # rate constraint

        att = pdict["params"][i]["attitude"]
        # attitude hold : angular velocity is zero
        if att in ["hold", "vertical"]:
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n * 3)))
            jac["u"]["coo"][1].extend(list(range(a * 3, (a + n) * 3)))
            jac["u"]["coo"][2].extend([1.0] * (n * 3))
            iRow += n * 3

        # kick-turn : pitch rate constant, roll/yaw rate is zero
        elif att == "kick-turn" or att == "pitch":
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(a * 3, (a + n) * 3, 3)))
            jac["u"]["coo"][2].extend([1.0] * n)
            iRow += n
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(a * 3 + 2, (a + n) * 3 + 2, 3)))
            jac["u"]["coo"][2].extend([1.0] * n)
            iRow += n
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend([a * 3 + 1] * (n - 1))
            jac["u"]["coo"][2].extend([-1.0] * (n - 1))
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend(list(range((a + 1) * 3 + 1, (a + n) * 3 + 1, 3)))
            jac["u"]["coo"][2].extend([1.0] * (n - 1))
            iRow += n - 1

        # pitch-yaw : pitch/yaw constant, roll ANGLE is zero
        elif att == "pitch-yaw":
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend([a * 3 + 1] * (n - 1))
            jac["u"]["coo"][2].extend([-1.0] * (n - 1))
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend(list(range((a + 1) * 3 + 1, (a + n) * 3 + 1, 3)))
            jac["u"]["coo"][2].extend([1.0] * (n - 1))
            iRow += n - 1
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend([a * 3 + 2] * (n - 1))
            jac["u"]["coo"][2].extend([-1.0] * (n - 1))
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend(list(range((a + 1) * 3 + 2, (a + n) * 3 + 2, 3)))
            jac["u"]["coo"][2].extend([1.0] * (n - 1))
            iRow += n - 1
            for k in range(n):
                f_c = yb_r_dot(pos_i_[k + 1] * unit_pos, quat_i_[k + 1])
                for j in range(3):
                    pos_i_[k + 1, j] += dx
                    f_p = yb_r_dot(pos_i_[k + 1] * unit_pos, quat_i_[k + 1])
                    pos_i_[k + 1, j] -= dx
                    jac["position"]["coo"][0].append(iRow + k)
                    jac["position"]["coo"][1].append((a + i + 1 + k) * 3 + j)
                    jac["position"]["coo"][2].append((f_p - f_c) / dx)
                for j in range(4):
                    quat_i_[k + 1, j] += dx
                    f_p = yb_r_dot(pos_i_[k + 1] * unit_pos, quat_i_[k + 1])
                    quat_i_[k + 1, j] -= dx
                    jac["quaternion"]["coo"][0].append(iRow + k)
                    jac["quaternion"]["coo"][1].append((a + i + 1 + k) * 4 + j)
                    jac["quaternion"]["coo"][2].append((f_p - f_c) / dx)
            iRow += n

        # same-rate : pitch/yaw is the same as previous section, roll ANGLE is zero
        elif att == "same-rate":
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend([a * 3 - 2] * n)
            jac["u"]["coo"][2].extend([-1.0] * n)
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(a * 3 + 1, (a + n) * 3 + 1, 3)))
            jac["u"]["coo"][2].extend([1.0] * n)
            iRow += n
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend([a * 3 - 1] * n)
            jac["u"]["coo"][2].extend([-1.0] * n)
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(a * 3 + 2, (a + n) * 3 + 2, 3)))
            jac["u"]["coo"][2].extend([1.0] * n)
            iRow += n
            for k in range(n):
                f_c = yb_r_dot(pos_i_[k + 1] * unit_pos, quat_i_[k + 1])
                for j in range(3):
                    pos_i_[k + 1, j] += dx
                    f_p = yb_r_dot(pos_i_[k + 1] * unit_pos, quat_i_[k + 1])
                    pos_i_[k + 1, j] -= dx
                    jac["position"]["coo"][0].append(iRow + k)
                    jac["position"]["coo"][1].append((a + i + 1 + k) * 3 + j)
                    jac["position"]["coo"][2].append((f_p - f_c) / dx)
                for j in range(4):
                    quat_i_[k + 1, j] += dx
                    f_p = yb_r_dot(pos_i_[k + 1] * unit_pos, quat_i_[k + 1])
                    quat_i_[k + 1, j] -= dx
                    jac["quaternion"]["coo"][0].append(iRow + k)
                    jac["quaternion"]["coo"][1].append((a + i + 1 + k) * 4 + j)
                    jac["quaternion"]["coo"][2].append((f_p - f_c) / dx)
            iRow += n

        # zero-lift-turn or free : roll hold
        elif att == "zero-lift-turn" or att == "free":
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(a * 3, (a + n) * 3, 3)))
            jac["u"]["coo"][2].extend([1.0] * n)
            iRow += n

        else:
            print("ERROR: UNKNOWN ATTITUDE OPTION! ({})".format(att))
            sys.exit()

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def inequality_time(xdict, pdict, unitdict, condition):
    """Inequality constraint about time at knots."""

    con = []
    t_normal = xdict["t"]

    for i in range(pdict["num_sections"]):
        if not (
            pdict["params"][i]["time_ref"] in pdict["event_index"].keys()
            and pdict["params"][i + 1]["time_ref"] in pdict["event_index"].keys()
        ):
            con.append(t_normal[i + 1] - t_normal[i])

    return np.array(con)


def inequality_jac_time(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_time."""

    jac = {}

    data = []
    row = []
    col = []

    counter = 0
    for i in range(pdict["num_sections"]):
        if not (
            pdict["params"][i]["time_ref"] in pdict["event_index"].keys()
            and pdict["params"][i + 1]["time_ref"] in pdict["event_index"].keys()
        ):
            data.extend([-1.0, 1.0])
            row.extend([counter, counter])
            col.extend([i, i + 1])
            counter += 1

    jac["t"] = {
        "coo": [
            np.array(row, dtype="i4"),
            np.array(col, dtype="i4"),
            np.array(data, dtype="f8"),
        ],
        "shape": (counter, len(xdict["t"])),
    }
    return jac


def inequality_mass(xdict, pdict, unitdict, condition):
    """Inequality constraint about max propellant mass."""

    con = []

    mass_ = xdict["mass"]
    for index, stage in pdict["RocketStage"].items():

        # read index number
        section_ig = [
            i
            for i, value in enumerate(pdict["params"])
            if value["name"] == stage["ignition_at"]
        ][0]
        section_co = [
            i
            for i, value in enumerate(pdict["params"])
            if value["name"] == stage["cutoff_at"]
        ][0]

        mass_ig = mass_[pdict["ps_params"][section_ig]["index_start"] + section_ig]
        mass_co = mass_[pdict["ps_params"][section_co]["index_start"] + section_co]

        d_mass = stage["mass_propellant"]
        if stage["dropMass"] is not None:
            d_mass += sum([item["mass"] for item in stage["dropMass"].values()])
        con.append(-mass_ig + mass_co + d_mass / unitdict["mass"])

    return con


def inequality_jac_mass(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_mass"""

    jac = {}

    data = []
    row = []
    col = []

    counter = 0
    for index, stage in pdict["RocketStage"].items():
        section_ig = [
            i
            for i, value in enumerate(pdict["params"])
            if value["name"] == stage["ignition_at"]
        ][0]
        section_co = [
            i
            for i, value in enumerate(pdict["params"])
            if value["name"] == stage["cutoff_at"]
        ][0]
        data.extend([-1.0, 1.0])
        row.extend([counter, counter])
        col.extend(
            [
                pdict["ps_params"][section_ig]["index_start"] + section_ig,
                pdict["ps_params"][section_co]["index_start"] + section_co,
            ]
        )
        counter += 1

    jac["mass"] = {
        "coo": [
            np.array(row, dtype="i4"),
            np.array(col, dtype="i4"),
            np.array(data, dtype="f8"),
        ],
        "shape": (counter, len(xdict["mass"])),
    }
    return jac


def inequality_kickturn(xdict, pdict, unitdict, condition):
    """Inequality constraint about minimum rate at kick-turn."""

    con = []
    unit_u = unitdict["u"]
    u_ = xdict["u"].reshape(-1, 3) * unit_u
    num_sections = pdict["num_sections"]

    for i in range(num_sections - 1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        u_i_ = u_[a:b]

        # kick turn
        if "kick" in pdict["params"][i]["attitude"]:
            con.append(-u_i_[:, 1])
            # con.append(u_i_[:,1]+0.36)

    return np.concatenate(con, axis=None)


def inequality_jac_kickturn(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_kickturn."""

    jac = {}
    num_sections = pdict["num_sections"]

    data = []
    row = []
    col = []

    nRow = 0
    for i in range(num_sections - 1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        # kick turn
        if "kick" in pdict["params"][i]["attitude"]:
            row.extend(range(nRow, nRow + n))
            col.extend(range(a * 3 + 1, b * 3 + 1, 3))
            data.extend([-1.0] * n)
            nRow += n

    jac["u"] = {
        "coo": [
            np.array(row, dtype="i4"),
            np.array(col, dtype="i4"),
            np.array(data, dtype="f8"),
        ],
        "shape": (nRow, len(xdict["u"])),
    }

    return jac


def inequality_max_alpha(xdict, pdict, unitdict, condition):
    """Inequality constraint about maximum angle of attack."""

    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    wind = pdict["wind_table"]

    for i in range(num_sections - 1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        pos_i_ = pos_[a + i : b + i + 1]
        vel_i_ = vel_[a + i : b + i + 1]
        quat_i_ = quat_[a + i : b + i + 1]
        to = t[i]
        tf = t[i + 1]
        t_i_ = np.zeros(n + 1)
        t_i_[0] = to
        t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf - to) / 2.0 + (tf + to) / 2.0

        section_name = pdict["params"][i]["name"]

        # angle of attack
        if section_name in condition["AOA_max"]:
            aoa_max = condition["AOA_max"][section_name]["value"] * np.pi / 180.0
            units[3] = aoa_max
            if condition["AOA_max"][section_name]["range"] == "all":
                con.append(
                    1.0
                    - aoa_zerolift_array_dimless(
                        pos_i_, vel_i_, quat_i_, t_i_, wind, units
                    )
                )
            elif condition["AOA_max"][section_name]["range"] == "initial":
                con.append(
                    1.0
                    - angle_of_attack_all_dimless(
                        pos_i_[0], vel_i_[0], quat_i_[0], to, wind, units
                    )
                )

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)


def inequality_max_q(xdict, pdict, unitdict, condition):
    """Inequality constraint about maximum dynamic pressure."""

    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    wind = pdict["wind_table"]

    for i in range(num_sections - 1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        pos_i_ = pos_[a + i : b + i + 1]
        vel_i_ = vel_[a + i : b + i + 1]
        to = t[i]
        tf = t[i + 1]
        t_i_ = np.zeros(n + 1)
        t_i_[0] = to
        t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf - to) / 2.0 + (tf + to) / 2.0

        section_name = pdict["params"][i]["name"]

        # max-Q
        if section_name in condition["dynamic_pressure_max"]:
            q_max = condition["dynamic_pressure_max"][section_name]["value"]
            units[3] = q_max
            if condition["dynamic_pressure_max"][section_name]["range"] == "all":
                con.append(
                    1.0
                    - dynamic_pressure_array_dimless(pos_i_, vel_i_, t_i_, wind, units)
                )
            elif condition["dynamic_pressure_max"][section_name]["range"] == "initial":
                con.append(
                    1.0
                    - dynamic_pressure_dimless(pos_i_[0], vel_i_[0], to, wind, units)
                )

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)


def inequality_max_qalpha(xdict, pdict, unitdict, condition):
    """Inequality constraint about maximum Q-alpha
    (product of angle of attack and dynamic pressure).
    """

    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    wind = pdict["wind_table"]

    for i in range(num_sections - 1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        pos_i_ = pos_[a + i : b + i + 1]
        vel_i_ = vel_[a + i : b + i + 1]
        quat_i_ = quat_[a + i : b + i + 1]
        to = t[i]
        tf = t[i + 1]
        t_i_ = np.zeros(n + 1)
        t_i_[0] = to
        t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf - to) / 2.0 + (tf + to) / 2.0

        section_name = pdict["params"][i]["name"]

        # max-Qalpha
        if section_name in condition["Q_alpha_max"]:
            qalpha_max = condition["Q_alpha_max"][section_name]["value"] * np.pi / 180.0
            units[3] = qalpha_max
            if condition["Q_alpha_max"][section_name]["range"] == "all":
                con.append(
                    1.0
                    - q_alpha_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units)
                )
            elif condition["Q_alpha_max"][section_name]["range"] == "initial":
                con.append(
                    1.0
                    - q_alpha_dimless(pos_i_[0], vel_i_[0], quat_i_[0], to, wind, units)
                )

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)


def inequality_jac_max_alpha(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_max_alpha."""

    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)

    t = xdict["t"]

    wind = pdict["wind_table"]
    num_sections = pdict["num_sections"]

    nRow = inequality_length_max_alpha(xdict, pdict, unitdict, condition)
    if nRow == 0:
        return None

    jac["position"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["velocity"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["quaternion"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 4)}
    jac["t"] = {"coo": [[], [], []], "shape": (nRow, num_sections + 1)}

    iRow = 0

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]

        # angle of attack
        if section_name in condition["AOA_max"]:
            a = pdict["ps_params"][i]["index_start"]
            n = pdict["ps_params"][i]["nodes"]
            b = a + n

            pos_i_ = pos_[a + i : b + i + 1]
            vel_i_ = vel_[a + i : b + i + 1]
            quat_i_ = quat_[a + i : b + i + 1]
            to = t[i]
            tf = t[i + 1]
            t_i_ = np.zeros(n + 1)
            t_i_[0] = to
            t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf - to) / 2.0 + (tf + to) / 2.0
            t_i_p1_ = np.zeros(n + 1)
            t_i_p2_ = np.zeros(n + 1)

            aoa_max = condition["AOA_max"][section_name]["value"] * np.pi / 180.0
            units[3] = aoa_max

            if condition["AOA_max"][section_name]["range"] == "all":
                nk = range(n + 1)
            elif condition["AOA_max"][section_name]["range"] == "initial":
                nk = [0]

            f_c = aoa_zerolift_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units)

            to_p = to + dx
            t_i_p1_[0] = to_p
            t_i_p1_[1:] = (
                pdict["ps_params"][i]["tau"] * (tf - to_p) / 2.0 + (tf + to_p) / 2.0
            )

            tf_p = tf + dx
            t_i_p2_[0] = to
            t_i_p2_[1:] = (
                pdict["ps_params"][i]["tau"] * (tf_p - to) / 2.0 + (tf_p + to) / 2.0
            )

            for j in range(3):
                pos_i_[nk, j] += dx
                f_p = aoa_zerolift_array_dimless(
                    pos_i_[nk], vel_i_[nk], quat_i_[nk], t_i_[nk], wind, units
                )
                pos_i_[nk, j] -= dx
                for k in nk:
                    jac["position"]["coo"][0].append(iRow + k)
                    jac["position"]["coo"][1].append((a + i + k) * 3 + j)
                    jac["position"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            for j in range(3):
                vel_i_[nk, j] += dx
                f_p = aoa_zerolift_array_dimless(
                    pos_i_[nk], vel_i_[nk], quat_i_[nk], t_i_[nk], wind, units
                )
                vel_i_[nk, j] -= dx
                for k in nk:
                    jac["velocity"]["coo"][0].append(iRow + k)
                    jac["velocity"]["coo"][1].append((a + i + k) * 3 + j)
                    jac["velocity"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            for j in range(4):
                quat_i_[nk, j] += dx
                f_p = aoa_zerolift_array_dimless(
                    pos_i_[nk], vel_i_[nk], quat_i_[nk], t_i_[nk], wind, units
                )
                quat_i_[nk, j] -= dx
                for k in nk:
                    jac["quaternion"]["coo"][0].append(iRow + k)
                    jac["quaternion"]["coo"][1].append((a + i + k) * 4 + j)
                    jac["quaternion"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            f_p = aoa_zerolift_array_dimless(
                pos_i_[nk], vel_i_[nk], quat_i_[nk], t_i_p1_[nk], wind, units
            )
            for k in nk:
                jac["t"]["coo"][0].append(iRow + k)
                jac["t"]["coo"][1].append(i)
                jac["t"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            f_p = aoa_zerolift_array_dimless(
                pos_i_[nk], vel_i_[nk], quat_i_[nk], t_i_p2_[nk], wind, units
            )
            for k in nk:
                jac["t"]["coo"][0].append(iRow + k)
                jac["t"]["coo"][1].append(i + 1)
                jac["t"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            iRow += len(nk)

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def inequality_jac_max_q(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_max_q."""

    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)

    t = xdict["t"]

    wind = pdict["wind_table"]
    num_sections = pdict["num_sections"]

    nRow = inequality_length_max_q(xdict, pdict, unitdict, condition)
    if nRow == 0:
        return None

    jac["position"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["velocity"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["quaternion"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 4)}
    jac["t"] = {"coo": [[], [], []], "shape": (nRow, num_sections + 1)}

    iRow = 0

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]

        # angle of attack
        if section_name in condition["dynamic_pressure_max"]:
            a = pdict["ps_params"][i]["index_start"]
            n = pdict["ps_params"][i]["nodes"]
            b = a + n

            pos_i_ = pos_[a + i : b + i + 1]
            vel_i_ = vel_[a + i : b + i + 1]
            to = t[i]
            tf = t[i + 1]
            t_i_ = np.zeros(n + 1)
            t_i_[0] = to
            t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf - to) / 2.0 + (tf + to) / 2.0
            t_i_p1_ = np.zeros(n + 1)
            t_i_p2_ = np.zeros(n + 1)

            q_max = condition["dynamic_pressure_max"][section_name]["value"]
            units[3] = q_max

            if condition["dynamic_pressure_max"][section_name]["range"] == "all":
                nk = range(n + 1)
            elif condition["dynamic_pressure_max"][section_name]["range"] == "initial":
                nk = [0]

            f_c = dynamic_pressure_array_dimless(pos_i_, vel_i_, t_i_, wind, units)

            to_p = to + dx
            t_i_p1_[0] = to_p
            t_i_p1_[1:] = (
                pdict["ps_params"][i]["tau"] * (tf - to_p) / 2.0 + (tf + to_p) / 2.0
            )
            tf_p = tf + dx
            t_i_p2_[0] = to
            t_i_p2_[1:] = (
                pdict["ps_params"][i]["tau"] * (tf_p - to) / 2.0 + (tf_p + to) / 2.0
            )

            for j in range(3):
                pos_i_[nk, j] += dx
                f_p = dynamic_pressure_array_dimless(
                    pos_i_[nk], vel_i_[nk], t_i_[nk], wind, units
                )
                pos_i_[nk, j] -= dx
                for k in nk:
                    jac["position"]["coo"][0].append(iRow + k)
                    jac["position"]["coo"][1].append((a + i + k) * 3 + j)
                    jac["position"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            for j in range(3):
                vel_i_[nk, j] += dx
                f_p = dynamic_pressure_array_dimless(
                    pos_i_[nk], vel_i_[nk], t_i_[nk], wind, units
                )
                vel_i_[nk, j] -= dx
                for k in nk:
                    jac["velocity"]["coo"][0].append(iRow + k)
                    jac["velocity"]["coo"][1].append((a + i + k) * 3 + j)
                    jac["velocity"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            f_p = dynamic_pressure_array_dimless(
                pos_i_[nk], vel_i_[nk], t_i_p1_[nk], wind, units
            )
            for k in nk:
                jac["t"]["coo"][0].append(iRow + k)
                jac["t"]["coo"][1].append(i)
                jac["t"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            f_p = dynamic_pressure_array_dimless(
                pos_i_[nk], vel_i_[nk], t_i_p2_[nk], wind, units
            )
            for k in nk:
                jac["t"]["coo"][0].append(iRow + k)
                jac["t"]["coo"][1].append(i + 1)
                jac["t"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            iRow += len(nk)

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def inequality_jac_max_qalpha(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_max_qalpha."""

    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)

    t = xdict["t"]

    wind = pdict["wind_table"]
    num_sections = pdict["num_sections"]

    nRow = inequality_length_max_qalpha(xdict, pdict, unitdict, condition)
    if nRow == 0:
        return None

    jac["position"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["velocity"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["quaternion"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 4)}
    jac["t"] = {"coo": [[], [], []], "shape": (nRow, num_sections + 1)}

    iRow = 0

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]

        # angle of attack
        if section_name in condition["Q_alpha_max"]:
            a = pdict["ps_params"][i]["index_start"]
            n = pdict["ps_params"][i]["nodes"]
            b = a + n

            pos_i_ = pos_[a + i : b + i + 1]
            vel_i_ = vel_[a + i : b + i + 1]
            quat_i_ = quat_[a + i : b + i + 1]
            to = t[i]
            tf = t[i + 1]
            t_i_ = np.zeros(n + 1)
            t_i_[0] = to
            t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf - to) / 2.0 + (tf + to) / 2.0
            t_i_p1_ = np.zeros(n + 1)
            t_i_p2_ = np.zeros(n + 1)
            to_p = to + dx
            t_i_p1_[0] = to_p
            t_i_p1_[1:] = (
                pdict["ps_params"][i]["tau"] * (tf - to_p) / 2.0 + (tf + to_p) / 2.0
            )
            tf_p = tf + dx
            t_i_p2_[0] = to
            t_i_p2_[1:] = (
                pdict["ps_params"][i]["tau"] * (tf_p - to) / 2.0 + (tf_p + to) / 2.0
            )

            qalpha_max = condition["Q_alpha_max"][section_name]["value"] * np.pi / 180.0
            units[3] = qalpha_max

            if condition["Q_alpha_max"][section_name]["range"] == "all":
                nk = range(n + 1)
            elif condition["Q_alpha_max"][section_name]["range"] == "initial":
                nk = [0]

            f_c = q_alpha_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units)

            for j in range(3):
                pos_i_[nk, j] += dx
                f_p = q_alpha_array_dimless(
                    pos_i_[nk], vel_i_[nk], quat_i_[nk], t_i_[nk], wind, units
                )
                pos_i_[nk, j] -= dx
                for k in nk:
                    jac["position"]["coo"][0].append(iRow + k)
                    jac["position"]["coo"][1].append((a + i + k) * 3 + j)
                    jac["position"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            for j in range(3):
                vel_i_[nk, j] += dx
                f_p = q_alpha_array_dimless(
                    pos_i_[nk], vel_i_[nk], quat_i_[nk], t_i_[nk], wind, units
                )
                vel_i_[nk, j] -= dx
                for k in nk:
                    jac["velocity"]["coo"][0].append(iRow + k)
                    jac["velocity"]["coo"][1].append((a + i + k) * 3 + j)
                    jac["velocity"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            for j in range(4):
                quat_i_[nk, j] += dx
                f_p = q_alpha_array_dimless(
                    pos_i_[nk], vel_i_[nk], quat_i_[nk], t_i_[nk], wind, units
                )
                quat_i_[nk, j] -= dx
                for k in nk:
                    jac["quaternion"]["coo"][0].append(iRow + k)
                    jac["quaternion"]["coo"][1].append((a + i + k) * 4 + j)
                    jac["quaternion"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            f_p = q_alpha_array_dimless(
                pos_i_[nk], vel_i_[nk], quat_i_[nk], t_i_p1_[nk], wind, units
            )
            for k in nk:
                jac["t"]["coo"][0].append(iRow + k)
                jac["t"]["coo"][1].append(i)
                jac["t"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            f_p = q_alpha_array_dimless(
                pos_i_[nk], vel_i_[nk], quat_i_[nk], t_i_p2_[nk], wind, units
            )
            for k in nk:
                jac["t"]["coo"][0].append(iRow + k)
                jac["t"]["coo"][1].append(i + 1)
                jac["t"]["coo"][2].append(-(f_p[k] - f_c[k]) / dx)

            iRow += len(nk)

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


@jit(nopython=True)
def dynamic_pressure_array_dimless(pos, vel, t, wind, units):
    """Returns array of dynamic pressure for each state values."""
    return np.array(
        [
            dynamic_pressure_dimless(pos[i], vel[i], t[i], wind, units)
            for i in range(len(t))
        ]
    )


@jit(nopython=True)
def aoa_zerolift_array_dimless(pos, vel, quat, t, wind, units):
    """Returns array of angle of attack for each state values."""
    return np.array(
        [
            angle_of_attack_all_dimless(pos[i], vel[i], quat[i], t[i], wind, units)
            for i in range(len(t))
        ]
    )


@jit(nopython=True)
def q_alpha_array_dimless(pos, vel, quat, t, wind, units):
    """Returns array of Q-alpha for each state values."""
    return np.array(
        [
            q_alpha_dimless(pos[i], vel[i], quat[i], t[i], wind, units)
            for i in range(len(t))
        ]
    )


@jit(nopython=True)
def roll_direction_array(pos, quat):
    """Returns array of sine of roll angles for each state values."""
    return np.array([yb_r_dot(pos[i], quat[i]) for i in range(len(pos))])


@jit(nopython=True)
def yb_r_dot(pos_eci, quat_eci2body):
    """Returns sine of roll angles."""
    yb_dir_eci = quatrot(conj(quat_eci2body), np.array([0.0, 1.0, 0.0]))
    return yb_dir_eci.dot(normalize(pos_eci))


@jit(nopython=True)
def angle_of_attack_all_dimless(pos_eci_e, vel_eci_e, quat, t_e, wind, units):
    """Returns angle of attack normalized by its maximum value."""
    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return angle_of_attack_all_rad(pos_eci, vel_eci, quat, t, wind) / units[3]


@jit(nopython=True)
def angle_of_attack_ab_dimless(pos_eci_e, vel_eci_e, quat, t_e, wind, units):
    """Returns pitch and yaw angles of attack normalized by
    their maximum values.
    """
    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return angle_of_attack_ab_rad(pos_eci, vel_eci, quat, t, wind) / units[3]


@jit(nopython=True)
def dynamic_pressure_dimless(pos_eci_e, vel_eci_e, t_e, wind, units):
    """Returns dynamic pressure normalized by its maximum value."""
    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return dynamic_pressure_pa(pos_eci, vel_eci, t, wind) / units[3]


@jit(nopython=True)
def q_alpha_dimless(pos_eci_e, vel_eci_e, quat, t_e, wind, units):
    """Returns Q-alpha normalized by its maximum value."""
    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return (
        dynamic_pressure_pa(pos_eci, vel_eci, t, wind)
        * angle_of_attack_all_rad(pos_eci, vel_eci, quat, t, wind)
        / units[3]
    )


def inequality_length_max_alpha(xdict, pdict, unitdict, condition):
    """Length of inequality_max_alpha."""
    res = 0

    num_sections = pdict["num_sections"]

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        # max-Qalpha
        if section_name in condition["AOA_max"]:

            if condition["AOA_max"][section_name]["range"] == "all":
                res += pdict["ps_params"][i]["nodes"] + 1
            elif condition["AOA_max"][section_name]["range"] == "initial":
                res += 1

    return res


def inequality_length_max_q(xdict, pdict, unitdict, condition):
    """Length of inequality_max_q."""
    res = 0

    num_sections = pdict["num_sections"]

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        # max-Qalpha
        if section_name in condition["dynamic_pressure_max"]:

            if condition["dynamic_pressure_max"][section_name]["range"] == "all":
                res += pdict["ps_params"][i]["nodes"] + 1
            elif condition["dynamic_pressure_max"][section_name]["range"] == "initial":
                res += 1

    return res


def inequality_length_max_qalpha(xdict, pdict, unitdict, condition):
    """Length of inequality_max_qalpha."""
    res = 0

    num_sections = pdict["num_sections"]

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        # max-Qalpha
        if section_name in condition["Q_alpha_max"]:

            if condition["Q_alpha_max"][section_name]["range"] == "all":
                res += pdict["ps_params"][i]["nodes"] + 1
            elif condition["Q_alpha_max"][section_name]["range"] == "initial":
                res += 1

    return res


def inequality_antenna(xdict, pdict, unitdict, condition):
    """Inequality constraint about antenna elevation angle."""
    con = []

    unit_pos = unitdict["position"]
    unit_t = unitdict["t"]

    pos_ = xdict["position"].reshape(-1, 3)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    if "antenna" not in condition:
        return None

    for antenna in condition["antenna"].values():

        posECEF_ANT = geodetic2ecef(antenna["lat"], antenna["lon"], antenna["altitude"])

        for i in range(num_sections - 1):

            section_name = pdict["params"][i]["name"]
            if section_name in antenna["elevation_min"]:

                elevation_min = antenna["elevation_min"][section_name]
                a = pdict["ps_params"][i]["index_start"]
                pos_o_ = pos_[a + i]
                to_ = t[i]
                sin_elv = sin_elevation(pos_o_, to_, posECEF_ANT, unit_pos, unit_t)
                con.append(sin_elv - np.sin(elevation_min * np.pi / 180.0))

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)


def inequality_jac_antenna(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_antenna."""

    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_t = unitdict["t"]
    pos_ = xdict["position"].reshape(-1, 3)
    t = xdict["t"]
    num_sections = pdict["num_sections"]

    f_center = inequality_antenna(xdict, pdict, unitdict, condition)
    if hasattr(f_center, "__len__"):
        nRow = len(f_center)
    elif f_center is None:
        return None
    else:
        nRow = 1

    jac["position"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["t"] = {"coo": [[], [], []], "shape": (nRow, num_sections + 1)}

    iRow = 0
    for antenna in condition["antenna"].values():

        posECEF_ANT = geodetic2ecef(antenna["lat"], antenna["lon"], antenna["altitude"])
        for i in range(num_sections - 1):

            section_name = pdict["params"][i]["name"]
            if section_name in antenna["elevation_min"]:

                a = pdict["ps_params"][i]["index_start"]
                pos_o_ = pos_[a + i]
                to_ = t[i]
                f_c = sin_elevation(pos_o_, to_, posECEF_ANT, unit_pos, unit_t)

                for j in range(3):
                    pos_o_[j] += dx
                    f_p = sin_elevation(pos_o_, to_, posECEF_ANT, unit_pos, unit_t)
                    pos_o_[j] -= dx
                    jac["position"]["coo"][0].append(iRow)
                    jac["position"]["coo"][1].append((a + i) * 3 + j)
                    jac["position"]["coo"][2].append((f_p - f_c) / dx)

                f_p = sin_elevation(pos_o_, to_ + dx, posECEF_ANT, unit_pos, unit_t)
                jac["t"]["coo"][0].append(iRow)
                jac["t"]["coo"][1].append(i)
                jac["t"]["coo"][2].append((f_p - f_c) / dx)

                iRow += 1

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def sin_elevation(pos_, t_, posECEF_ANT, unit_pos, unit_t):
    pos = pos_ * unit_pos
    to = t_ * unit_t
    posECEF = eci2ecef(pos, to)
    direction_ANT = normalize(posECEF - posECEF_ANT)
    vertical_ANT = quatrot(quat_nedg2ecef(posECEF_ANT), np.array([0, 0, -1.0]))
    return np.dot(direction_ANT, vertical_ANT)


def equality_IIP(xdict, pdict, unitdict, condition):
    """Equality constraint about IIP position."""
    con = []
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    if "waypoint" not in condition:
        return None

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        if section_name in condition["waypoint"]:

            waypoint = condition["waypoint"][section_name]
            a = pdict["ps_params"][i]["index_start"]
            pos = pos_[a + i] * unit_pos
            vel = vel_[a + i] * unit_vel
            to = t[i] * unit_t
            posECEF = eci2ecef(pos, to)
            velECEF = vel_eci2ecef(vel, pos, to)
            posLLH_IIP = posLLH_IIP_FAA(posECEF, velECEF)
            # latitude
            if "lat_IIP" in waypoint:
                if "exact" in waypoint["lat_IIP"]:
                    con.append((posLLH_IIP[0] - waypoint["lat_IIP"]["exact"]) / 90.0)

            # longitude
            if "lon_IIP" in waypoint:
                if "exact" in waypoint["lon_IIP"]:
                    con.append((posLLH_IIP[1] - waypoint["lon_IIP"]["exact"]) / 180.0)

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)


def equality_jac_IIP(xdict, pdict, unitdict, condition):
    """Jacobian of equality_IIP."""
    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    t_ = xdict["t"]
    num_sections = pdict["num_sections"]

    f_center = equality_IIP(xdict, pdict, unitdict, condition)
    if hasattr(f_center, "__len__"):
        nRow = len(f_center)
    elif f_center is None:
        return None
    else:
        nRow = 1

    if "waypoint" not in condition:
        return None

    jac["position"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["velocity"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["t"] = {"coo": [[], [], []], "shape": (nRow, num_sections + 1)}

    iRow = 0
    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        if section_name in condition["waypoint"]:

            waypoint = condition["waypoint"][section_name]
            a = pdict["ps_params"][i]["index_start"]
            pos_o_ = pos_[a + i]
            vel_o_ = vel_[a + i]
            to_ = t_[i]
            posLLH_IIP_c = posLLH_IIP_FAA(
                eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                vel_eci2ecef(vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t),
            )
            # latitude
            if "lat_IIP" in waypoint:
                if "exact" in waypoint["lat_IIP"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (posLLH_IIP_p[0] - posLLH_IIP_c[0]) / dx / 90.0
                        )

                    for j in range(3):
                        vel_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        vel_o_[j] -= dx
                        jac["velocity"]["coo"][0].append(iRow)
                        jac["velocity"]["coo"][1].append((a + i) * 3 + j)
                        jac["velocity"]["coo"][2].append(
                            (posLLH_IIP_p[0] - posLLH_IIP_c[0]) / dx / 90.0
                        )

                    posLLH_IIP_p = posLLH_IIP_FAA(
                        eci2ecef(pos_o_ * unit_pos, (to_ + dx) * unit_t),
                        vel_eci2ecef(
                            vel_o_ * unit_vel, pos_o_ * unit_pos, (to_ + dx) * unit_t
                        ),
                    )
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        (posLLH_IIP_p[0] - posLLH_IIP_c[0]) / dx / 90.0
                    )
                    iRow += 1

            # longitude
            if "lon_IIP" in waypoint:
                # min
                if "exact" in waypoint["lon_IIP"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (posLLH_IIP_p[1] - posLLH_IIP_c[1]) / dx / 180.0
                        )

                    for j in range(3):
                        vel_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        vel_o_[j] -= dx
                        jac["velocity"]["coo"][0].append(iRow)
                        jac["velocity"]["coo"][1].append((a + i) * 3 + j)
                        jac["velocity"]["coo"][2].append(
                            (posLLH_IIP_p[1] - posLLH_IIP_c[1]) / dx / 180.0
                        )

                    posLLH_IIP_p = posLLH_IIP_FAA(
                        eci2ecef(pos_o_ * unit_pos, (to_ + dx) * unit_t),
                        vel_eci2ecef(
                            vel_o_ * unit_vel, pos_o_ * unit_pos, (to_ + dx) * unit_t
                        ),
                    )
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        (posLLH_IIP_p[1] - posLLH_IIP_c[1]) / dx / 180.0
                    )
                    iRow += 1

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def inequality_IIP(xdict, pdict, unitdict, condition):
    """Inequality constraint about IIP position."""
    con = []
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    if "waypoint" not in condition:
        return None

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        if section_name in condition["waypoint"]:

            waypoint = condition["waypoint"][section_name]
            a = pdict["ps_params"][i]["index_start"]
            pos = pos_[a + i] * unit_pos
            vel = vel_[a + i] * unit_vel
            to = t[i] * unit_t
            posECEF = eci2ecef(pos, to)
            velECEF = vel_eci2ecef(vel, pos, to)
            posLLH_IIP = posLLH_IIP_FAA(posECEF, velECEF)
            # latitude
            if "lat_IIP" in waypoint:
                # min
                if "min" in waypoint["lat_IIP"]:
                    con.append((posLLH_IIP[0] - waypoint["lat_IIP"]["min"]) / 90.0)
                # max
                if "max" in waypoint["lat_IIP"]:
                    con.append((waypoint["lat_IIP"]["max"] - posLLH_IIP[0]) / 90.0)

            # longitude
            if "lon_IIP" in waypoint:
                # min
                if "min" in waypoint["lon_IIP"]:
                    con.append((posLLH_IIP[1] - waypoint["lon_IIP"]["min"]) / 180.0)
                # max
                if "max" in waypoint["lon_IIP"]:
                    con.append((waypoint["lon_IIP"]["max"] - posLLH_IIP[1]) / 180.0)

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)


def inequality_jac_IIP(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_IIP."""
    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    t_ = xdict["t"]
    num_sections = pdict["num_sections"]

    f_center = inequality_IIP(xdict, pdict, unitdict, condition)
    if hasattr(f_center, "__len__"):
        nRow = len(f_center)
    elif f_center is None:
        return None
    else:
        nRow = 1

    if "waypoint" not in condition:
        return None

    jac["position"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["velocity"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["t"] = {"coo": [[], [], []], "shape": (nRow, num_sections + 1)}

    iRow = 0
    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        if section_name in condition["waypoint"]:

            waypoint = condition["waypoint"][section_name]
            a = pdict["ps_params"][i]["index_start"]
            pos_o_ = pos_[a + i]
            vel_o_ = vel_[a + i]
            to_ = t_[i]
            posLLH_IIP_c = posLLH_IIP_FAA(
                eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                vel_eci2ecef(vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t),
            )
            # latitude
            if "lat_IIP" in waypoint:
                if "min" in waypoint["lat_IIP"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (posLLH_IIP_p[0] - posLLH_IIP_c[0]) / dx / 90.0
                        )

                    for j in range(3):
                        vel_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        vel_o_[j] -= dx
                        jac["velocity"]["coo"][0].append(iRow)
                        jac["velocity"]["coo"][1].append((a + i) * 3 + j)
                        jac["velocity"]["coo"][2].append(
                            (posLLH_IIP_p[0] - posLLH_IIP_c[0]) / dx / 90.0
                        )

                    posLLH_IIP_p = posLLH_IIP_FAA(
                        eci2ecef(pos_o_ * unit_pos, (to_ + dx) * unit_t),
                        vel_eci2ecef(
                            vel_o_ * unit_vel, pos_o_ * unit_pos, (to_ + dx) * unit_t
                        ),
                    )
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        (posLLH_IIP_p[0] - posLLH_IIP_c[0]) / dx / 90.0
                    )
                    iRow += 1

                if "max" in waypoint["lat_IIP"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (posLLH_IIP_p[0] - posLLH_IIP_c[0]) / dx / -90.0
                        )

                    for j in range(3):
                        vel_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        vel_o_[j] -= dx
                        jac["velocity"]["coo"][0].append(iRow)
                        jac["velocity"]["coo"][1].append((a + i) * 3 + j)
                        jac["velocity"]["coo"][2].append(
                            (posLLH_IIP_p[0] - posLLH_IIP_c[0]) / dx / -90.0
                        )

                    posLLH_IIP_p = posLLH_IIP_FAA(
                        eci2ecef(pos_o_ * unit_pos, (to_ + dx) * unit_t),
                        vel_eci2ecef(
                            vel_o_ * unit_vel, pos_o_ * unit_pos, (to_ + dx) * unit_t
                        ),
                    )
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        (posLLH_IIP_p[0] - posLLH_IIP_c[0]) / dx / -90.0
                    )
                    iRow += 1

            # longitude
            if "lon_IIP" in waypoint:
                # min
                if "min" in waypoint["lon_IIP"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (posLLH_IIP_p[1] - posLLH_IIP_c[1]) / dx / 180.0
                        )

                    for j in range(3):
                        vel_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        vel_o_[j] -= dx
                        jac["velocity"]["coo"][0].append(iRow)
                        jac["velocity"]["coo"][1].append((a + i) * 3 + j)
                        jac["velocity"]["coo"][2].append(
                            (posLLH_IIP_p[1] - posLLH_IIP_c[1]) / dx / 180.0
                        )

                    posLLH_IIP_p = posLLH_IIP_FAA(
                        eci2ecef(pos_o_ * unit_pos, (to_ + dx) * unit_t),
                        vel_eci2ecef(
                            vel_o_ * unit_vel, pos_o_ * unit_pos, (to_ + dx) * unit_t
                        ),
                    )
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        (posLLH_IIP_p[1] - posLLH_IIP_c[1]) / dx / 180.0
                    )
                    iRow += 1

                # max
                if "max" in waypoint["lon_IIP"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (posLLH_IIP_p[1] - posLLH_IIP_c[1]) / dx / -180.0
                        )

                    for j in range(3):
                        vel_o_[j] += dx
                        posLLH_IIP_p = posLLH_IIP_FAA(
                            eci2ecef(pos_o_ * unit_pos, to_ * unit_t),
                            vel_eci2ecef(
                                vel_o_ * unit_vel, pos_o_ * unit_pos, to_ * unit_t
                            ),
                        )
                        vel_o_[j] -= dx
                        jac["velocity"]["coo"][0].append(iRow)
                        jac["velocity"]["coo"][1].append((a + i) * 3 + j)
                        jac["velocity"]["coo"][2].append(
                            (posLLH_IIP_p[1] - posLLH_IIP_c[1]) / dx / -180.0
                        )

                    posLLH_IIP_p = posLLH_IIP_FAA(
                        eci2ecef(pos_o_ * unit_pos, (to_ + dx) * unit_t),
                        vel_eci2ecef(
                            vel_o_ * unit_vel, pos_o_ * unit_pos, (to_ + dx) * unit_t
                        ),
                    )
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        (posLLH_IIP_p[1] - posLLH_IIP_c[1]) / dx / -180.0
                    )
                    iRow += 1

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def equality_posLLH(xdict, pdict, unitdict, condition):
    """Equality constraint about IIP position."""
    con = []
    unit_pos = unitdict["position"]
    unit_t = unitdict["t"]

    pos_ = xdict["position"].reshape(-1, 3)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    if "waypoint" not in condition:
        return None

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        if section_name in condition["waypoint"]:

            waypoint = condition["waypoint"][section_name]
            a = pdict["ps_params"][i]["index_start"]
            pos = pos_[a + i] * unit_pos
            to = t[i] * unit_t
            posLLH = eci2geodetic(pos, to)
            lon_origin = pdict["LaunchCondition"]["lon"]
            lat_origin = pdict["LaunchCondition"]["lat"]
            downrange = distance_vincenty(lat_origin, lon_origin, posLLH[0], posLLH[1])

            # altitude
            if "altitude" in waypoint:
                if "exact" in waypoint["altitude"]:
                    con.append((posLLH[2] / waypoint["altitude"]["exact"]) - 1.0)

            # downrange
            if "downrange" in waypoint:
                if "exact" in waypoint["downrange"]:
                    con.append((downrange / waypoint["downrange"]["exact"]) - 1.0)

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)


def equality_jac_posLLH(xdict, pdict, unitdict, condition):
    """Jacobian of equality_posLLH."""

    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_t = unitdict["t"]
    pos_ = xdict["position"].reshape(-1, 3)
    t_ = xdict["t"]
    num_sections = pdict["num_sections"]

    f_center = equality_posLLH(xdict, pdict, unitdict, condition)
    if hasattr(f_center, "__len__"):
        nRow = len(f_center)
    elif f_center is None:
        return None
    else:
        nRow = 1

    lon_origin = pdict["LaunchCondition"]["lon"]
    lat_origin = pdict["LaunchCondition"]["lat"]

    jac["position"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["t"] = {"coo": [[], [], []], "shape": (nRow, num_sections + 1)}

    iRow = 0
    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        if section_name in condition["waypoint"]:

            waypoint = condition["waypoint"][section_name]
            a = pdict["ps_params"][i]["index_start"]
            pos_o_ = pos_[a + i]
            to_ = t_[i]
            posLLH_c = eci2geodetic(pos_o_ * unit_pos, to_ * unit_t)
            downrange_c = distance_vincenty(
                lat_origin, lon_origin, posLLH_c[0], posLLH_c[1]
            )

            # altitude
            if "altitude" in waypoint:
                if "exact" in waypoint["altitude"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_p = eci2geodetic(pos_o_ * unit_pos, to_ * unit_t)
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (posLLH_p[2] - posLLH_c[2])
                            / dx
                            / waypoint["altitude"]["exact"]
                        )

                    posLLH_p = eci2geodetic(pos_o_ * unit_pos, (to_ + dx) * unit_t)
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        (posLLH_p[2] - posLLH_c[2]) / dx / waypoint["altitude"]["exact"]
                    )
                    iRow += 1

            # downrange
            if "downrange" in waypoint:
                if "exact" in waypoint["downrange"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_p = eci2geodetic(pos_o_ * unit_pos, to_ * unit_t)
                        downrange_p = distance_vincenty(
                            lat_origin, lon_origin, posLLH_p[0], posLLH_p[1]
                        )
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (downrange_p - downrange_c)
                            / dx
                            / waypoint["downrange"]["exact"]
                        )

                    posLLH_p = eci2geodetic(pos_o_ * unit_pos, (to_ + dx) * unit_t)
                    downrange_p = distance_vincenty(
                        lat_origin, lon_origin, posLLH_p[0], posLLH_p[1]
                    )
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["position"]["coo"][2].append(
                        (downrange_p - downrange_c)
                        / dx
                        / waypoint["downrange"]["exact"]
                    )
                    iRow += 1

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def inequality_posLLH(xdict, pdict, unitdict, condition):
    """Inequality constraint about IIP position."""
    con = []
    unit_pos = unitdict["position"]
    unit_t = unitdict["t"]

    pos_ = xdict["position"].reshape(-1, 3)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    if "waypoint" not in condition:
        return None

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        if section_name in condition["waypoint"]:

            waypoint = condition["waypoint"][section_name]
            a = pdict["ps_params"][i]["index_start"]
            pos = pos_[a + i] * unit_pos
            to = t[i] * unit_t
            posLLH = eci2geodetic(pos, to)
            lon_origin = pdict["LaunchCondition"]["lon"]
            lat_origin = pdict["LaunchCondition"]["lat"]
            downrange = distance_vincenty(lat_origin, lon_origin, posLLH[0], posLLH[1])

            # altitude
            if "altitude" in waypoint:
                # min
                if "min" in waypoint["altitude"]:
                    con.append((posLLH[2] / waypoint["altitude"]["min"]) - 1.0)
                # max
                if "max" in waypoint["altitude"]:
                    con.append(-(posLLH[2] / waypoint["altitude"]["max"]) + 1.0)

            # downrange
            if "downrange" in waypoint:
                # min
                if "min" in waypoint["downrange"]:
                    con.append((downrange / waypoint["downrange"]["min"]) - 1.0)
                # max
                if "max" in waypoint["downrange"]:
                    con.append(-(downrange / waypoint["downrange"]["min"]) + 1.0)

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)


def inequality_jac_posLLH(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_posLLH."""

    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_t = unitdict["t"]
    pos_ = xdict["position"].reshape(-1, 3)
    t_ = xdict["t"]
    num_sections = pdict["num_sections"]

    f_center = inequality_posLLH(xdict, pdict, unitdict, condition)
    if hasattr(f_center, "__len__"):
        nRow = len(f_center)
    elif f_center is None:
        return None
    else:
        nRow = 1

    lon_origin = pdict["LaunchCondition"]["lon"]
    lat_origin = pdict["LaunchCondition"]["lat"]

    jac["position"] = {"coo": [[], [], []], "shape": (nRow, pdict["M"] * 3)}
    jac["t"] = {"coo": [[], [], []], "shape": (nRow, num_sections + 1)}

    iRow = 0
    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        if section_name in condition["waypoint"]:

            waypoint = condition["waypoint"][section_name]
            a = pdict["ps_params"][i]["index_start"]
            pos_o_ = pos_[a + i]
            to_ = t_[i]
            posLLH_c = eci2geodetic(pos_o_ * unit_pos, to_ * unit_t)
            downrange_c = distance_vincenty(
                lat_origin, lon_origin, posLLH_c[0], posLLH_c[1]
            )

            # altitude
            if "altitude" in waypoint:
                if "min" in waypoint["altitude"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_p = eci2geodetic(pos_o_ * unit_pos, to_ * unit_t)
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (posLLH_p[2] - posLLH_c[2])
                            / dx
                            / waypoint["altitude"]["min"]
                        )

                    posLLH_p = eci2geodetic(pos_o_ * unit_pos, (to_ + dx) * unit_t)
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        (posLLH_p[2] - posLLH_c[2]) / dx / waypoint["altitude"]["min"]
                    )
                    iRow += 1

                if "max" in waypoint["altitude"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_p = eci2geodetic(pos_o_ * unit_pos, to_ * unit_t)
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (posLLH_p[2] - posLLH_c[2])
                            / dx
                            / -waypoint["altitude"]["max"]
                        )

                    posLLH_p = eci2geodetic(pos_o_ * unit_pos, (to_ + dx) * unit_t)
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        (posLLH_p[2] - posLLH_c[2]) / dx / -waypoint["altitude"]["max"]
                    )
                    iRow += 1

            # downrange
            if "downrange" in waypoint:
                if "min" in waypoint["downrange"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_p = eci2geodetic(pos_o_ * unit_pos, to_ * unit_t)
                        downrange_p = distance_vincenty(
                            lat_origin, lon_origin, posLLH_p[0], posLLH_p[1]
                        )
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (downrange_p - downrange_c)
                            / dx
                            / waypoint["downrange"]["min"]
                        )

                    posLLH_p = eci2geodetic(pos_o_ * unit_pos, (to_ + dx) * unit_t)
                    downrange_p = distance_vincenty(
                        lat_origin, lon_origin, posLLH_p[0], posLLH_p[1]
                    )
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["position"]["coo"][2].append(
                        (downrange_p - downrange_c) / dx / waypoint["downrange"]["min"]
                    )
                    iRow += 1

                if "max" in waypoint["downrange"]:
                    for j in range(3):
                        pos_o_[j] += dx
                        posLLH_p = eci2geodetic(pos_o_ * unit_pos, to_ * unit_t)
                        downrange_p = distance_vincenty(
                            lat_origin, lon_origin, posLLH_p[0], posLLH_p[1]
                        )
                        pos_o_[j] -= dx
                        jac["position"]["coo"][0].append(iRow)
                        jac["position"]["coo"][1].append((a + i) * 3 + j)
                        jac["position"]["coo"][2].append(
                            (downrange_p - downrange_c)
                            / dx
                            / -waypoint["downrange"]["max"]
                        )

                    posLLH_p = eci2geodetic(pos_o_ * unit_pos, (to_ + dx) * unit_t)
                    downrange_p = distance_vincenty(
                        lat_origin, lon_origin, posLLH_p[0], posLLH_p[1]
                    )
                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["position"]["coo"][2].append(
                        (downrange_p - downrange_c) / dx / -waypoint["downrange"]["max"]
                    )
                    iRow += 1

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def equality_jac_user(xdict, pdict, unitdict, condition):
    """Jacobian of user-defined equality constraint."""
    if equality_user(xdict, pdict, unitdict, condition) is not None:
        return jac_fd(equality_user, xdict, pdict, unitdict, condition)


def inequality_jac_user(xdict, pdict, unitdict, condition):
    """Jacobian of user-defined inequality constraint."""
    if inequality_user(xdict, pdict, unitdict, condition) is not None:
        return jac_fd(inequality_user, xdict, pdict, unitdict, condition)


def cost_6DoF(xdict, condition):
    """Objective function of the optimization problem."""
    if condition["OptimizationMode"] == "Payload":
        return -xdict["mass"][0]  # 初期質量(無次元)を最大化
    else:
        return xdict["t"][-1]  # 到達時間を最小化(=余剰推進剤を最大化)


def cost_jac(xdict, condition):
    """Gradient of the objective function."""

    jac = {}
    if condition["OptimizationMode"] == "Payload":
        jac["mass"] = np.zeros(xdict["mass"].size)
        jac["mass"][0] = -1.0
    else:
        jac["t"] = np.zeros(xdict["t"].size)
        jac["t"][-1] = 1.0
    return jac
