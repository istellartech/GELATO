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

# constraints_b.py
# constraints about stage mass and turn rate


import sys
import numpy as np


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

        mass_ig = mass_[pdict["ps_params"].index_start_x(section_ig)]
        mass_co = mass_[pdict["ps_params"].index_start_x(section_co)]

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
                pdict["ps_params"].index_start_x(section_ig),
                pdict["ps_params"].index_start_x(section_co),
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
        # kick turn
        if "kick" in pdict["params"][i]["attitude"]:
            ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
            u_i_ = u_[ua:ub]
            con.append(-u_i_[:, 1])

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
        # kick turn
        if "kick" in pdict["params"][i]["attitude"]:
            ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
            row.extend(range(nRow, nRow + n))
            col.extend(range(ua * 3 + 1, ub * 3 + 1, 3))
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


def equality_6DoF_rate(xdict, pdict, unitdict, condition):
    """Equality constraint about angular rate."""

    con = []

    u_ = xdict["u"].reshape(-1, 3)

    num_sections = pdict["num_sections"]

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
        u_i_ = u_[ua:ub]

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

        # pitch-yaw : pitch/yaw constant, roll rate is zero
        elif att == "pitch-yaw":
            con.append(u_i_[1:, 1] - u_i_[0, 1])
            con.append(u_i_[1:, 2] - u_i_[0, 2])
            con.append(u_i_[:, 0])

        # same-rate : pitch/yaw is the same as previous section, roll rate is zero
        elif att == "same-rate":
            uf_prev = u_[ua - 1]
            con.append(u_i_[:, 1] - uf_prev[1])
            con.append(u_i_[:, 2] - uf_prev[2])
            con.append(u_i_[:, 0])

        # zero-lift-turn or free : roll hold
        elif att == "zero-lift-turn" or att == "free":
            con.append(u_i_[:, 0])

        else:
            print("ERROR: UNKNOWN ATTITUDE OPTION! ({})".format(att))
            sys.exit()

    return np.concatenate(con, axis=None)


def equality_length_6DoF_rate(xdict, pdict, unitdict, condition):
    """length of equality_6DoF_rate"""

    res = 0
    num_sections = pdict["num_sections"]

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
        # rate constraint

        att = pdict["params"][i]["attitude"]

        # attitude hold : angular velocity is zero
        if att in ["hold", "vertical"]:
            res += 3 * n

        # kick-turn : pitch rate constant, roll/yaw rate is zero
        elif att == "kick-turn" or att == "pitch":
            res += 3 * n - 1

        # pitch-yaw : pitch/yaw constant, roll rate is zero
        elif att == "pitch-yaw":
            res += 3 * n - 2

        # same-rate : pitch/yaw is the same as previous section, roll rate is zero
        elif att == "same-rate":
            res += 3 * n

        # zero-lift-turn or free : roll hold
        elif att == "zero-lift-turn" or att == "free":
            res += n

        else:
            print("ERROR: UNKNOWN ATTITUDE OPTION! ({})".format(att))
            sys.exit()

    return res


def equality_jac_6DoF_rate(xdict, pdict, unitdict, condition):
    """Jacobian of equality_rate."""

    jac = {}

    num_sections = pdict["num_sections"]

    nRow = equality_length_6DoF_rate(xdict, pdict, unitdict, condition)
    jac["u"] = {"coo": [[], [], []], "shape": (nRow, pdict["N"] * 3)}

    iRow = 0

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)

        # rate constraint

        att = pdict["params"][i]["attitude"]
        # attitude hold : angular velocity is zero
        if att in ["hold", "vertical"]:
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n * 3)))
            jac["u"]["coo"][1].extend(list(range(ua * 3, (ua + n) * 3)))
            jac["u"]["coo"][2].extend([1.0] * (n * 3))
            iRow += n * 3

        # kick-turn : pitch rate constant, roll/yaw rate is zero
        elif att == "kick-turn" or att == "pitch":
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(ua * 3, (ua + n) * 3, 3)))
            jac["u"]["coo"][2].extend([1.0] * n)
            iRow += n
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(ua * 3 + 2, (ua + n) * 3 + 2, 3)))
            jac["u"]["coo"][2].extend([1.0] * n)
            iRow += n
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend([ua * 3 + 1] * (n - 1))
            jac["u"]["coo"][2].extend([-1.0] * (n - 1))
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend(
                list(range((ua + 1) * 3 + 1, (ua + n) * 3 + 1, 3))
            )
            jac["u"]["coo"][2].extend([1.0] * (n - 1))
            iRow += n - 1

        # pitch-yaw : pitch/yaw constant, roll rate is zero
        elif att == "pitch-yaw":
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend([ua * 3 + 1] * (n - 1))
            jac["u"]["coo"][2].extend([-1.0] * (n - 1))
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend(
                list(range((ua + 1) * 3 + 1, (ua + n) * 3 + 1, 3))
            )
            jac["u"]["coo"][2].extend([1.0] * (n - 1))
            iRow += n - 1
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend([ua * 3 + 2] * (n - 1))
            jac["u"]["coo"][2].extend([-1.0] * (n - 1))
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n - 1)))
            jac["u"]["coo"][1].extend(
                list(range((ua + 1) * 3 + 2, (ua + n) * 3 + 2, 3))
            )
            jac["u"]["coo"][2].extend([1.0] * (n - 1))
            iRow += n - 1
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(ua * 3, (ua + n) * 3, 3)))
            jac["u"]["coo"][2].extend([1.0] * n)
            iRow += n

        # same-rate : pitch/yaw is the same as previous section, roll rate is zero
        elif att == "same-rate":
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend([ua * 3 - 2] * n)
            jac["u"]["coo"][2].extend([-1.0] * n)
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(ua * 3 + 1, (ua + n) * 3 + 1, 3)))
            jac["u"]["coo"][2].extend([1.0] * n)
            iRow += n
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend([ua * 3 - 1] * n)
            jac["u"]["coo"][2].extend([-1.0] * n)
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(ua * 3 + 2, (ua + n) * 3 + 2, 3)))
            jac["u"]["coo"][2].extend([1.0] * n)
            iRow += n
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(ua * 3, (ua + n) * 3, 3)))
            jac["u"]["coo"][2].extend([1.0] * n)
            iRow += n

        # zero-lift-turn or free : roll hold
        elif att == "zero-lift-turn" or att == "free":
            jac["u"]["coo"][0].extend(list(range(iRow, iRow + n)))
            jac["u"]["coo"][1].extend(list(range(ua * 3, (ua + n) * 3, 3)))
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
