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
from math import cos, radians
from coordinate import (
    angular_momentum,
    orbit_energy,
    inclination_rad,
    angular_momentum_from_altitude,
    orbit_energy_from_altitude,
)


# constraints_a.py
# constraints about initial, knotting and terminal conditions


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
            index_ig = pdict["ps_params"].index_start_x(section_ig)
            index_sep = pdict["ps_params"].index_start_x(section_sep)
            con.append(
                mass_[index_ig] - mass_[index_sep] - mass_stage / unitdict["mass"]
            )

    for i in range(1, num_sections):
        xa = pdict["ps_params"].index_start_x(i)

        param[0] = pdict["params"][i]["thrust"]
        param[1] = pdict["params"][i]["massflow"]
        param[2] = pdict["params"][i]["reference_area"]
        param[4] = pdict["params"][i]["nozzle_area"]

        # knotting constraints
        mass_init_ = mass_[xa]
        mass_prev_ = mass_[xa - 1]
        if not (i in section_sep_list):
            con.append(
                mass_init_
                - mass_prev_
                + pdict["params"][i]["mass_jettison"] / unitdict["mass"]
            )

        pos_init_ = pos_[xa]
        pos_prev_ = pos_[xa - 1]
        con.append(pos_init_ - pos_prev_)

        vel_init_ = vel_[xa]
        vel_prev_ = vel_[xa - 1]
        con.append(vel_init_ - vel_prev_)

        quat_init_ = quat_[xa]
        quat_prev_ = quat_[xa - 1]
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
            index_ig = pdict["ps_params"].index_start_x(section_ig)
            index_sep = pdict["ps_params"].index_start_x(section_sep)
            jac["mass"]["coo"][0].extend([iRow, iRow])
            jac["mass"]["coo"][1].extend([index_ig, index_sep])
            jac["mass"]["coo"][2].extend([1.0, -1.0])
            iRow += 1

    for i in range(1, num_sections):
        xa = pdict["ps_params"].index_start_x(i)

        if not (i in section_sep_list):
            jac["mass"]["coo"][0].extend([iRow, iRow])
            jac["mass"]["coo"][1].extend([xa - 1, xa])
            jac["mass"]["coo"][2].extend([-1.0, 1.0])
            iRow += 1

        jac["position"]["coo"][0].extend(list(range(iRow, iRow + 3)))
        jac["position"]["coo"][1].extend(list(range((xa - 1) * 3, (xa) * 3)))
        jac["position"]["coo"][2].extend([-1.0] * 3)
        jac["position"]["coo"][0].extend(list(range(iRow, iRow + 3)))
        jac["position"]["coo"][1].extend(list(range((xa) * 3, (xa + 1) * 3)))
        jac["position"]["coo"][2].extend([1.0] * 3)
        iRow += 3

        jac["velocity"]["coo"][0].extend(list(range(iRow, iRow + 3)))
        jac["velocity"]["coo"][1].extend(list(range((xa - 1) * 3, (xa) * 3)))
        jac["velocity"]["coo"][2].extend([-1.0] * 3)
        jac["velocity"]["coo"][0].extend(list(range(iRow, iRow + 3)))
        jac["velocity"]["coo"][1].extend(list(range((xa) * 3, (xa + 1) * 3)))
        jac["velocity"]["coo"][2].extend([1.0] * 3)
        iRow += 3

        jac["quaternion"]["coo"][0].extend(list(range(iRow, iRow + 4)))
        jac["quaternion"]["coo"][1].extend(list(range((xa - 1) * 4, (xa) * 4)))
        jac["quaternion"]["coo"][2].extend([-1.0] * 4)
        jac["quaternion"]["coo"][0].extend(list(range(iRow, iRow + 4)))
        jac["quaternion"]["coo"][1].extend(list(range((xa) * 4, (xa + 1) * 4)))
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

    # terminal conditions

    pos_f = xdict["position"][-3:] * unit_pos
    vel_f = xdict["velocity"][-3:] * unit_vel

    GMe = 3.986004418e14
    if (
        condition["altitude_perigee"] is not None
        and condition["altitude_apogee"] is not None
    ):
        c_target = angular_momentum_from_altitude(
            condition["altitude_perigee"],
            condition["altitude_apogee"],
        )
        e_target = orbit_energy_from_altitude(
            condition["altitude_perigee"],
            condition["altitude_apogee"],
        )
    else:
        c_target = condition["radius"] * condition["vel_tangential_geocentric"]
        vf_target = condition["vel_tangential_geocentric"] / cos(
            radians(condition["flightpath_vel_inertial_geocentric"])
        )
        e_target = vf_target**2 / 2.0 - GMe / condition["radius"]

    c = angular_momentum(pos_f, vel_f)
    e = orbit_energy(pos_f, vel_f)
    con.append((e / e_target) - 1.0)  # orbit energy
    con.append((c / c_target) - 1.0)  # angular momentum

    if condition["inclination"] is not None:
        inc = inclination_rad(pos_f, vel_f)
        inc_target = radians(condition["inclination"])
        con.append(inc - inc_target)

    return np.concatenate(con, axis=None)


def equality_jac_6DoF_LGR_terminal(xdict, pdict, unitdict, condition):
    """Jacobian of equality_terminal."""

    jac = {}
    dx = pdict["dx"]

    f_center = equality_6DoF_LGR_terminal(xdict, pdict, unitdict, condition)

    nRow = 2
    if condition["inclination"] is not None:
        nRow += 1

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
