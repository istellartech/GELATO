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


def sin_elevation_gradient(pos_, t_, posECEF_ANT, unit_pos, unit_t, dx):
    grad = {"position": np.zeros(3), "t": 0.0}
    f_center = sin_elevation(pos_, t_, posECEF_ANT, unit_pos, unit_t)
    for j in range(3):
        pos_[j] += dx
        f_p = sin_elevation(pos_, t_, posECEF_ANT, unit_pos, unit_t)
        pos_[j] -= dx
        grad["position"][j] = (f_p - f_center) / dx

    t_ += dx
    f_p = sin_elevation(pos_, t_, posECEF_ANT, unit_pos, unit_t)
    grad["t"] = (f_p - f_center) / dx

    return grad


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
                xa = pdict["ps_params"].index_start_x(i)
                pos_o_ = pos_[xa]
                to_ = t[i]
                sin_elv = sin_elevation(pos_o_, to_, posECEF_ANT, unit_pos, unit_t)
                con.append(sin_elv - np.sin(elevation_min * np.pi / 180.0))

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)


@profile
def inequality_jac_antenna(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_antenna."""

    jac = {}
    dx = pdict["dx"]

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

            if pdict["params"][i]["name"] in antenna["elevation_min"]:

                xa = pdict["ps_params"].index_start_x(i)
                pos_o_ = pos_[xa]
                to_ = t[i]
                dfdx = sin_elevation_gradient(
                    pos_o_, to_, posECEF_ANT, unit_pos, unit_t, dx
                )

                jac["position"]["coo"][0].extend([iRow] * 3)
                jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                jac["position"]["coo"][2].extend(dfdx["position"].tolist())

                jac["t"]["coo"][0].append(iRow)
                jac["t"]["coo"][1].append(i)
                jac["t"]["coo"][2].append(dfdx["t"])

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
            xa = pdict["ps_params"].index_start_x(i)
            pos = pos_[xa] * unit_pos
            vel = vel_[xa] * unit_vel
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


def posLLH_IIP_gradient(pos_ECI, vel_ECI, t, unit_pos, unit_vel, unit_t, dx):
    grad = {
        "position": np.zeros((3, 3)),
        "velocity": np.zeros((3, 3)),
        "t": np.zeros(3),
    }

    def iip_from_eci(pos_, vel_, t_):
        return posLLH_IIP_FAA(
            eci2ecef(pos_ * unit_pos, t_ * unit_t),
            vel_eci2ecef(vel_ * unit_vel, pos_ * unit_pos, t_ * unit_t),
        )

    f_center = iip_from_eci(pos_ECI, vel_ECI, t)

    for j in range(3):
        pos_ECI[j] += dx
        f_pr = iip_from_eci(pos_ECI, vel_ECI, t)
        pos_ECI[j] -= dx
        grad["position"][:, j] = (f_pr - f_center) / dx

        vel_ECI[j] += dx
        f_pv = iip_from_eci(pos_ECI, vel_ECI, t)
        vel_ECI[j] -= dx
        grad["velocity"][:, j] = (f_pv - f_center) / dx

    t += dx
    f_pt = iip_from_eci(pos_ECI, vel_ECI, t)
    grad["t"] = (f_pt - f_center) / dx

    return grad


@profile
def equality_jac_IIP(xdict, pdict, unitdict, condition):
    """Jacobian of equality_IIP."""
    jac = {}
    dx = pdict["dx"]

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
            xa = pdict["ps_params"].index_start_x(i)
            pos_o_ = pos_[xa]
            vel_o_ = vel_[xa]
            to_ = t_[i]


            # latitude
            if "lat_IIP" in waypoint:
                if "exact" in waypoint["lat_IIP"]:
                    dfdx = posLLH_IIP_gradient(
                        pos_o_, vel_o_, to_, unit_pos, unit_vel, unit_t, dx
                    )
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(dfdx["position"][0, :] / 90.0)

                    jac["velocity"]["coo"][0].extend([iRow] * 3)
                    jac["velocity"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["velocity"]["coo"][2].extend(dfdx["velocity"][0, :] / 90.0)

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(dfdx["t"][0] / 90.0)
                    iRow += 1

            # longitude
            if "lon_IIP" in waypoint:
                if "exact" in waypoint["lon_IIP"]:
                    dfdx = posLLH_IIP_gradient(
                        pos_o_, vel_o_, to_, unit_pos, unit_vel, unit_t, dx
                    )
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(dfdx["position"][1, :] / 180.0)

                    jac["velocity"]["coo"][0].extend([iRow] * 3)
                    jac["velocity"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["velocity"]["coo"][2].extend(dfdx["velocity"][1, :] / 180.0)

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(dfdx["t"][1] / 180.0)
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
            xa = pdict["ps_params"].index_start_x(i)
            pos = pos_[xa] * unit_pos
            vel = vel_[xa] * unit_vel
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


@profile
def inequality_jac_IIP(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_IIP."""
    jac = {}
    dx = pdict["dx"]

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
            xa = pdict["ps_params"].index_start_x(i)
            pos_o_ = pos_[xa]
            vel_o_ = vel_[xa]
            to_ = t_[i]

            # latitude
            if "lat_IIP" in waypoint:
                if "min" in waypoint["lat_IIP"]:
                    dfdx = posLLH_IIP_gradient(
                        pos_o_, vel_o_, to_, unit_pos, unit_vel, unit_t, dx
                    )
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(dfdx["position"][0, :] / 90.0)

                    jac["velocity"]["coo"][0].extend([iRow] * 3)
                    jac["velocity"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["velocity"]["coo"][2].extend(dfdx["velocity"][0, :] / 90.0)

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(dfdx["t"][0] / 90.0)
                    iRow += 1

                if "max" in waypoint["lat_IIP"]:
                    dfdx = posLLH_IIP_gradient(
                        pos_o_, vel_o_, to_, unit_pos, unit_vel, unit_t, dx
                    )
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(-dfdx["position"][0, :] / 90.0)

                    jac["velocity"]["coo"][0].extend([iRow] * 3)
                    jac["velocity"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["velocity"]["coo"][2].extend(-dfdx["velocity"][0, :] / 90.0)

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(-dfdx["t"][0] / 90.0)
                    iRow += 1

            # longitude
            if "lon_IIP" in waypoint:
                # min
                if "min" in waypoint["lon_IIP"]:
                    dfdx = posLLH_IIP_gradient(
                        pos_o_, vel_o_, to_, unit_pos, unit_vel, unit_t, dx
                    )
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(dfdx["position"][1, :] / 180.0)

                    jac["velocity"]["coo"][0].extend([iRow] * 3)
                    jac["velocity"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["velocity"]["coo"][2].extend(dfdx["velocity"][1, :] / 180.0)

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(dfdx["t"][1] / 180.0)
                    iRow += 1

                # max
                if "max" in waypoint["lon_IIP"]:
                    dfdx = posLLH_IIP_gradient(
                        pos_o_, vel_o_, to_, unit_pos, unit_vel, unit_t, dx
                    )
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(-dfdx["position"][1, :] / 180.0)

                    jac["velocity"]["coo"][0].extend([iRow] * 3)
                    jac["velocity"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["velocity"]["coo"][2].extend(-dfdx["velocity"][1, :] / 180.0)

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(-dfdx["t"][1] / 180.0)
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
            xa = pdict["ps_params"].index_start_x(i)
            pos = pos_[xa] * unit_pos
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


def posLLH_gradient(pos_ECI, t, unit_pos, unit_t, dx):
    grad = {"position": np.zeros((3, 3)), "t": np.zeros(3)}

    def pos_llh(pos_, t_):
        return eci2geodetic(pos_ * unit_pos, t_ * unit_t)

    f_center = pos_llh(pos_ECI, t)

    for j in range(3):
        pos_ECI[j] += dx
        f_pr = pos_llh(pos_ECI, t)
        pos_ECI[j] -= dx
        grad["position"][:, j] = (f_pr - f_center) / dx

    t += dx
    f_pt = pos_llh(pos_ECI, t)
    grad["t"] = (f_pt - f_center) / dx

    return grad


def downrange_gradient(pos_ECI, t, unit_pos, unit_t, lat0, lon0, dx):
    grad = {"position": np.zeros(3), "t": 0.0}

    def downrange(pos_, t_):
        pos_llh = eci2geodetic(pos_ * unit_pos, t_ * unit_t)
        return distance_vincenty(
            lat0,
            lon0,
            pos_llh[0],
            pos_llh[1],
        )

    f_center = downrange(pos_ECI, t)

    for j in range(3):
        pos_ECI[j] += dx
        f_pr = downrange(pos_ECI, t)
        pos_ECI[j] -= dx
        grad["position"][j] = (f_pr - f_center) / dx

    t += dx
    f_pt = downrange(pos_ECI, t)
    grad["t"] = (f_pt - f_center) / dx

    return grad


def equality_jac_posLLH(xdict, pdict, unitdict, condition):
    """Jacobian of equality_posLLH."""

    jac = {}
    dx = pdict["dx"]

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
            xa = pdict["ps_params"].index_start_x(i)
            pos_o_ = pos_[xa]
            to_ = t_[i]

            # altitude
            if "altitude" in waypoint:
                if "exact" in waypoint["altitude"]:
                    dfdx_LLH = posLLH_gradient(pos_o_, to_, unit_pos, unit_t, dx)
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(
                        dfdx_LLH["position"][2, :] / waypoint["altitude"]["exact"]
                    )

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        dfdx_LLH["t"][2] / waypoint["altitude"]["exact"]
                    )
                    iRow += 1

            # downrange
            if "downrange" in waypoint:
                if "exact" in waypoint["downrange"]:
                    dfdx_downrange = downrange_gradient(
                        pos_o_, to_, unit_pos, unit_t, lat_origin, lon_origin, dx
                    )
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(
                        dfdx_downrange["position"] / waypoint["downrange"]["exact"]
                    )

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["position"]["coo"][2].append(
                        dfdx_downrange["t"] / waypoint["downrange"]["exact"]
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
            xa = pdict["ps_params"].index_start_x(i)
            pos = pos_[xa] * unit_pos
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

@profile
def inequality_jac_posLLH(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_posLLH."""

    jac = {}
    dx = pdict["dx"]

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
            xa = pdict["ps_params"].index_start_x(i)
            pos_o_ = pos_[xa]
            to_ = t_[i]

            # altitude
            if "altitude" in waypoint:
                if "min" in waypoint["altitude"]:
                    dfdx_LLH = posLLH_gradient(pos_o_, to_, unit_pos, unit_t, dx)
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(
                        dfdx_LLH["position"][2, :] / waypoint["altitude"]["min"]
                    )

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        dfdx_LLH["t"][2] / waypoint["altitude"]["min"]
                    )
                    iRow += 1

                if "max" in waypoint["altitude"]:
                    dfdx_LLH = posLLH_gradient(pos_o_, to_, unit_pos, unit_t, dx)
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(
                        -dfdx_LLH["position"][2, :] / waypoint["altitude"]["max"]
                    )

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["t"]["coo"][2].append(
                        -dfdx_LLH["t"][2] / waypoint["altitude"]["max"]
                    )
                    iRow += 1

            # downrange
            if "downrange" in waypoint:
                if "min" in waypoint["downrange"]:
                    dfdx_downrange = downrange_gradient(
                        pos_o_, to_, unit_pos, unit_t, lat_origin, lon_origin, dx
                    )
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(
                        dfdx_downrange["position"] / waypoint["downrange"]["min"]
                    )

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["position"]["coo"][2].append(
                        dfdx_downrange["t"] / waypoint["downrange"]["min"]
                    )
                    iRow += 1

                if "max" in waypoint["downrange"]:
                    dfdx_downrange = downrange_gradient(
                        pos_o_, to_, unit_pos, unit_t, lat_origin, lon_origin, dx
                    )
                    jac["position"]["coo"][0].extend([iRow] * 3)
                    jac["position"]["coo"][1].extend(range(xa * 3, (xa + 1) * 3))
                    jac["position"]["coo"][2].extend(
                        -dfdx_downrange["position"] / waypoint["downrange"]["min"]
                    )

                    jac["t"]["coo"][0].append(iRow)
                    jac["t"]["coo"][1].append(i)
                    jac["position"]["coo"][2].append(
                        -dfdx_downrange["t"] / waypoint["downrange"]["max"]
                    )
                    iRow += 1

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac
