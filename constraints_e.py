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

# constraints_e.py
# constraints about waypoint conditions


import numpy as np
from utils import *
from USStandardAtmosphere import *
from coordinate import geodetic2ecef, eci2ecef, normalize, quatrot, quat_nedg2ecef, vel_eci2ecef, eci2geodetic
from tools.IIP import posLLH_IIP_FAA
from tools.downrange import distance_vincenty


def sin_elevation(pos_, t_, posECEF_ANT, unit_pos, unit_t):
    pos = pos_ * unit_pos
    to = t_ * unit_t
    posECEF = eci2ecef(pos, to)
    direction_ANT = normalize(posECEF - posECEF_ANT)
    vertical_ANT = quatrot(quat_nedg2ecef(posECEF_ANT), np.array([0, 0, -1.0]))
    return np.dot(direction_ANT, vertical_ANT)


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
                a = pdict["ps_params"].index_start_u(i)
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

                a = pdict["ps_params"].index_start_u(i)
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
            a = pdict["ps_params"].index_start_u(i)
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
            a = pdict["ps_params"].index_start_u(i)
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
            a = pdict["ps_params"].index_start_u(i)
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
            a = pdict["ps_params"].index_start_u(i)
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
            a = pdict["ps_params"].index_start_u(i)
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
            a = pdict["ps_params"].index_start_u(i)
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
            a = pdict["ps_params"].index_start_u(i)
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
            a = pdict["ps_params"].index_start_u(i)
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
