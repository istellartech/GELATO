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

# constraints_c.py
# constraints about aerodynamic conditions

import numpy as np
from numba import jit
from utils import dynamic_pressure_pa, angle_of_attack_all_rad, q_alpha_pa_rad, dynamic_pressure_array_pa, angle_of_attack_all_array_rad, q_alpha_array_pa_rad
from coordinate import *



@jit(nopython=True)
def dynamic_pressure_dimless(pos_eci_e, vel_eci_e, t_e, wind, units):
    """Returns dynamic pressure normalized by its maximum value."""
    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return dynamic_pressure_pa(pos_eci, vel_eci, t, wind) / units[3]


@jit(nopython=True)
def dynamic_pressure_array_dimless(pos_eci_e, vel_eci_e, t_e, wind, units):
    """Returns array of dynamic pressure for each state values."""
    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return dynamic_pressure_array_pa(pos_eci, vel_eci, t, wind) / units[3]


@jit(nopython=True)
def angle_of_attack_all_dimless(pos_eci_e, vel_eci_e, quat, t_e, wind, units):
    """Returns angle of attack normalized by its maximum value."""
    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return angle_of_attack_all_rad(pos_eci, vel_eci, quat, t, wind) / units[3]


def angle_of_attack_all_array_dimless(pos_eci_e, vel_eci_e, quat, t_e, wind, units):
    """Returns array of angle of attack for each state values."""
    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return angle_of_attack_all_array_rad(pos_eci, vel_eci, quat, t, wind) / units[3]


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


def q_alpha_array_dimless(pos_eci_e, vel_eci_e, quat, t_e, wind, units):
    """Returns array of Q-alpha for each state values."""
    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return (
        q_alpha_array_pa_rad(pos_eci, vel_eci, quat, t, wind)
        / units[3]
    )


def inequality_max_alpha(xdict, pdict, unitdict, condition):
    """Inequality constraint about maximum angle of attack."""

    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = [unit_pos, unit_vel, unit_t, 1.0]

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    wind = pdict["wind_table"]

    for i in range(num_sections - 1):
        section_name = pdict["params"][i]["name"]

        # angle of attack
        if section_name in condition["AOA_max"]:

            ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
            pos_i_ = pos_[xa:xb]
            vel_i_ = vel_[xa:xb]
            quat_i_ = quat_[xa:xb]

            to = t[i]
            tf = t[i + 1]

            aoa_max = condition["AOA_max"][section_name]["value"] * np.pi / 180.0
            units[3] = aoa_max
            if condition["AOA_max"][section_name]["range"] == "all":
                t_i_ = pdict["ps_params"].time_nodes(i, to, tf)
                con.append(
                    1.0
                    - angle_of_attack_all_array_dimless(
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
    units = [unit_pos, unit_vel, unit_t, 1.0]

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    wind = pdict["wind_table"]

    for i in range(num_sections - 1):
        section_name = pdict["params"][i]["name"]
        # max-Q
        if section_name in condition["dynamic_pressure_max"]:

            ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
            pos_i_ = pos_[xa:xb]
            vel_i_ = vel_[xa:xb]
            to = t[i]
            tf = t[i + 1]
            q_max = condition["dynamic_pressure_max"][section_name]["value"]
            units[3] = q_max
            if condition["dynamic_pressure_max"][section_name]["range"] == "all":
                t_i_ = pdict["ps_params"].time_nodes(i, to, tf)
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
    units = [unit_pos, unit_vel, unit_t, 1.0]

    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    wind = pdict["wind_table"]

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        # max-Qalpha
        if section_name in condition["Q_alpha_max"]:

            ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
            pos_i_ = pos_[xa:xb]
            vel_i_ = vel_[xa:xb]
            quat_i_ = quat_[xa:xb]
            to = t[i]
            tf = t[i + 1]

            qalpha_max = condition["Q_alpha_max"][section_name]["value"] * np.pi / 180.0
            units[3] = qalpha_max
            if condition["Q_alpha_max"][section_name]["range"] == "all":
                t_i_ = pdict["ps_params"].time_nodes(i, to, tf)
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

def inequality_length_max_alpha(xdict, pdict, unitdict, condition):
    """Length of inequality_max_alpha."""
    res = 0

    num_sections = pdict["num_sections"]

    for i in range(num_sections - 1):

        section_name = pdict["params"][i]["name"]
        # max-Qalpha
        if section_name in condition["AOA_max"]:

            if condition["AOA_max"][section_name]["range"] == "all":
                res += (pdict["ps_params"].nodes(i) + 1)
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
                res += (pdict["ps_params"].nodes(i) + 1)
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
                res += (pdict["ps_params"].nodes(i) + 1)
            elif condition["Q_alpha_max"][section_name]["range"] == "initial":
                res += 1

    return res


def angle_of_attack_all_gradient_dimless(pos_eci_e, vel_eci_e, quat, to, tf, wind, units, time_nodes, dx, ki):
    """Returns gradient of Q-alpha."""
    n = len(ki)
    grad = {
        "position": np.zeros((n, 3)),
        "velocity": np.zeros((n, 3)),
        "quaternion": np.zeros((n, 4)),
        "to": np.zeros(n),
        "tf": np.zeros(n),
    }

    pos_ki = pos_eci_e[ki]
    vel_ki = vel_eci_e[ki]
    quat_ki = quat[ki]

    t_nodes = time_nodes(to, tf)
    t_ki = t_nodes[ki]

    f_c = angle_of_attack_all_array_dimless(
        pos_ki, vel_ki, quat_ki, t_ki, wind, units
    )

    for j in range(3):
        pos_ki[:, j] += dx
        f_p = angle_of_attack_all_array_dimless(
            pos_ki, vel_ki, quat_ki, t_ki, wind, units
        )
        pos_ki[:, j] -= dx
        grad["position"][:, j] = (f_p - f_c).ravel() / dx

    for j in range(3):
        vel_ki[:, j] += dx
        f_p = angle_of_attack_all_array_dimless(
            pos_ki, vel_ki, quat_ki, t_ki, wind, units
        )
        vel_ki[:, j] -= dx
        grad["velocity"][:, j] = (f_p - f_c).ravel() / dx

    for j in range(4):
        quat_ki[:, j] += dx
        f_p = angle_of_attack_all_array_dimless(
            pos_ki, vel_ki, quat_ki, t_ki, wind, units
        )
        quat_ki[:, j] -= dx
        grad["quaternion"][:, j] = (f_p - f_c).ravel() / dx

    to_p = to + dx
    t_nodes_p1 = time_nodes(to_p, tf)
    f_p = angle_of_attack_all_array_dimless(
        pos_ki, vel_ki, quat_ki, t_nodes_p1[ki], wind, units
    )
    grad["to"] = (f_p - f_c).ravel() / dx

    tf_p = tf + dx
    t_nodes_p2 = time_nodes(to, tf_p)
    f_p = angle_of_attack_all_array_dimless(
        pos_ki, vel_ki, quat_ki, t_nodes_p2[ki], wind, units
    )
    grad["tf"] = (f_p - f_c).ravel() / dx

    return grad


def inequality_jac_max_alpha(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_max_alpha."""

    jac = {}
    dx = pdict["dx"]

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = [unit_pos, unit_vel, unit_t, 1.0]

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
            ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
            pos_i_ = pos_[xa:xb]
            vel_i_ = vel_[xa:xb]
            quat_i_ = quat_[xa:xb]

            to = t[i]
            tf = t[i + 1]

            aoa_max = condition["AOA_max"][section_name]["value"] * np.pi / 180.0
            units[3] = aoa_max

            if condition["AOA_max"][section_name]["range"] == "all":
                ki = range(n + 1)
                nk = n + 1
            elif condition["AOA_max"][section_name]["range"] == "initial":
                ki = [0]
                nk = 1

            def time_nodes(t1, t2):
                return pdict["ps_params"].time_nodes(i, t1, t2)
            dfdx = angle_of_attack_all_gradient_dimless(
                pos_i_, vel_i_, quat_i_, to, tf, wind, units, time_nodes, dx, ki
            )

            for j in range(3):
                jac["position"]["coo"][0].extend(range(iRow, iRow + nk))
                jac["position"]["coo"][1].extend(range(xa * 3 + j, (xa + nk) * 3 + j, 3))
            jac["position"]["coo"][2].extend(-dfdx["position"].ravel(order="F"))

            for j in range(3):
                jac["velocity"]["coo"][0].extend(range(iRow, iRow + nk))
                jac["velocity"]["coo"][1].extend(range(xa * 3 + j, (xa + nk) * 3 + j, 3))
            jac["velocity"]["coo"][2].extend(-dfdx["velocity"].ravel(order="F"))

            for j in range(4):
                jac["quaternion"]["coo"][0].extend(range(iRow, iRow + nk))
                jac["quaternion"]["coo"][1].extend(range(xa * 4 + j, (xa + nk) * 4 + j, 4))
            jac["quaternion"]["coo"][2].extend(-dfdx["quaternion"].ravel(order="F"))

            jac["t"]["coo"][0].extend(range(iRow, iRow + nk))
            jac["t"]["coo"][1].extend([i] * nk)
            jac["t"]["coo"][2].extend(-dfdx["to"])

            jac["t"]["coo"][0].extend(range(iRow, iRow + nk))
            jac["t"]["coo"][1].extend([i + 1] * nk)
            jac["t"]["coo"][2].extend(-dfdx["tf"])

            iRow += nk

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def dynamic_pressure_gradient_dimless(pos_eci_e, vel_eci_e, to, tf, wind, units, time_nodes, dx, ki):
    """Returns gradient of dynamic pressure."""
    n = len(ki)
    grad = {
        "position": np.zeros((n, 3)),
        "velocity": np.zeros((n, 3)),
        "to": np.zeros(n),
        "tf": np.zeros(n),
    }

    pos_ki = pos_eci_e[ki]
    vel_ki = vel_eci_e[ki]

    t_nodes = time_nodes(to, tf)
    t_ki = t_nodes[ki]

    f_c = dynamic_pressure_array_dimless(
        pos_ki, vel_ki, t_ki, wind, units
    )

    for j in range(3):
        pos_ki[:, j] += dx
        f_p = dynamic_pressure_array_dimless(
            pos_ki, vel_ki, t_ki, wind, units
        )
        pos_ki[:, j] -= dx
        grad["position"][:, j] = (f_p - f_c).ravel() / dx

    for j in range(3):
        vel_ki[:, j] += dx
        f_p = dynamic_pressure_array_dimless(
            pos_ki, vel_ki, t_ki, wind, units
        )
        vel_ki[:, j] -= dx
        grad["velocity"][:, j] = (f_p - f_c).ravel() / dx

    to_p = to + dx
    t_nodes_p1 = time_nodes(to_p, tf)
    f_p = dynamic_pressure_array_dimless(
        pos_ki, vel_ki, t_nodes_p1[ki], wind, units
    )
    grad["to"] = (f_p - f_c).ravel() / dx

    tf_p = tf + dx
    t_nodes_p2 = time_nodes(to, tf_p)
    f_p = dynamic_pressure_array_dimless(
        pos_ki, vel_ki, t_nodes_p2[ki], wind, units
    )
    grad["tf"] = (f_p - f_c).ravel() / dx

    return grad


def inequality_jac_max_q(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_max_q."""

    jac = {}
    dx = pdict["dx"]

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = [unit_pos, unit_vel, unit_t, 1.0]

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
            ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
            pos_i_ = pos_[xa:xb]
            vel_i_ = vel_[xa:xb]

            to = t[i]
            tf = t[i + 1]

            q_max = condition["dynamic_pressure_max"][section_name]["value"]
            units[3] = q_max

            if condition["dynamic_pressure_max"][section_name]["range"] == "all":
                ki = range(n + 1)
                nk = n + 1
            elif condition["dynamic_pressure_max"][section_name]["range"] == "initial":
                ki = [0]
                nk = 1

            def time_nodes(t1, t2):
                return pdict["ps_params"].time_nodes(i, t1, t2)
            dfdx = dynamic_pressure_gradient_dimless(
                pos_i_, vel_i_, to, tf, wind, units, time_nodes, dx, ki
            )

            for j in range(3):
                jac["position"]["coo"][0].extend(range(iRow, iRow + nk))
                jac["position"]["coo"][1].extend(range(xa * 3 + j, (xa + nk) * 3 + j, 3))
            jac["position"]["coo"][2].extend(-dfdx["position"].ravel(order="F"))

            for j in range(3):
                jac["velocity"]["coo"][0].extend(range(iRow, iRow + nk))
                jac["velocity"]["coo"][1].extend(range(xa * 3 + j, (xa + nk) * 3 + j, 3))
            jac["velocity"]["coo"][2].extend(-dfdx["velocity"].ravel(order="F"))

            jac["t"]["coo"][0].extend(range(iRow, iRow + nk))
            jac["t"]["coo"][1].extend([i] * nk)
            jac["t"]["coo"][2].extend(-dfdx["to"])

            jac["t"]["coo"][0].extend(range(iRow, iRow + nk))
            jac["t"]["coo"][1].extend([i + 1] * nk)
            jac["t"]["coo"][2].extend(-dfdx["tf"])

            iRow += nk

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac

@profile
def q_alpha_gradient_dimless(pos_eci_e, vel_eci_e, quat, to, tf, wind, units, time_nodes, dx, ki):
    """Returns gradient of Q-alpha."""
    n = len(ki)
    grad = {
        "position": np.zeros((n, 3)),
        "velocity": np.zeros((n, 3)),
        "quaternion": np.zeros((n, 4)),
        "to": np.zeros(n),
        "tf": np.zeros(n),
    }

    pos_ki = pos_eci_e[ki]
    vel_ki = vel_eci_e[ki]
    quat_ki = quat[ki]

    t_nodes = time_nodes(to, tf)
    t_ki = t_nodes[ki]

    f_c = q_alpha_array_dimless(
        pos_ki, vel_ki, quat_ki, t_ki, wind, units
    )

    for j in range(3):
        pos_ki[:, j] += dx
        f_p = q_alpha_array_dimless(
            pos_ki, vel_ki, quat_ki, t_ki, wind, units
        )
        pos_ki[:, j] -= dx
        grad["position"][:, j] = (f_p - f_c).ravel() / dx

    for j in range(3):
        vel_ki[:, j] += dx
        f_p = q_alpha_array_dimless(
            pos_ki, vel_ki, quat_ki, t_ki, wind, units
        )
        vel_ki[:, j] -= dx
        grad["velocity"][:, j] = (f_p - f_c).ravel() / dx

    for j in range(4):
        quat_ki[:, j] += dx
        f_p = q_alpha_array_dimless(
            pos_ki, vel_ki, quat_ki, t_ki, wind, units
        )
        quat_ki[:, j] -= dx
        grad["quaternion"][:, j] = (f_p - f_c).ravel() / dx

    to_p = to + dx
    t_nodes_p1 = time_nodes(to_p, tf)
    f_p = q_alpha_array_dimless(
        pos_ki, vel_ki, quat_ki, t_nodes_p1[ki], wind, units
    )
    grad["to"] = (f_p - f_c).ravel() / dx

    tf_p = tf + dx
    t_nodes_p2 = time_nodes(to, tf_p)
    f_p = q_alpha_array_dimless(
        pos_ki, vel_ki, quat_ki, t_nodes_p2[ki], wind, units
    )
    grad["tf"] = (f_p - f_c).ravel() / dx

    return grad


@profile
def inequality_jac_max_qalpha(xdict, pdict, unitdict, condition):
    """Jacobian of inequality_max_qalpha."""

    jac = {}
    dx = pdict["dx"]

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = [unit_pos, unit_vel, unit_t, 1.0]

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
            ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
            pos_i_ = pos_[xa:xb]
            vel_i_ = vel_[xa:xb]
            quat_i_ = quat_[xa:xb]

            to = t[i]
            tf = t[i + 1]

            qalpha_max = condition["Q_alpha_max"][section_name]["value"] * np.pi / 180.0
            units[3] = qalpha_max

            if condition["Q_alpha_max"][section_name]["range"] == "all":
                ki = range(n + 1)
                nk = n + 1
            elif condition["Q_alpha_max"][section_name]["range"] == "initial":
                ki = [0]
                nk = 1

            def time_nodes(t1, t2):
                return pdict["ps_params"].time_nodes(i, t1, t2)
            dfdx = q_alpha_gradient_dimless(
                pos_i_, vel_i_, quat_i_, to, tf, wind, units, time_nodes, dx, ki
            )
            for j in range(3):
                jac["position"]["coo"][0].extend(range(iRow, iRow + nk))
                jac["position"]["coo"][1].extend(range(xa * 3 + j, (xa + nk) * 3 + j, 3))
            jac["position"]["coo"][2].extend(-dfdx["position"].ravel(order="F"))

            for j in range(3):
                jac["velocity"]["coo"][0].extend(range(iRow, iRow + nk))
                jac["velocity"]["coo"][1].extend(range(xa * 3 + j, (xa + nk) * 3 + j, 3))
            jac["velocity"]["coo"][2].extend(-dfdx["velocity"].ravel(order="F"))

            for j in range(4):
                jac["quaternion"]["coo"][0].extend(range(iRow, iRow + nk))
                jac["quaternion"]["coo"][1].extend(range(xa * 4 + j, (xa + nk) * 4 + j, 4))
            jac["quaternion"]["coo"][2].extend(-dfdx["quaternion"].ravel(order="F"))


            jac["t"]["coo"][0].extend(range(iRow, iRow + nk))
            jac["t"]["coo"][1].extend([i] * nk)
            jac["t"]["coo"][2].extend(-dfdx["to"])

            jac["t"]["coo"][0].extend(range(iRow, iRow + nk))
            jac["t"]["coo"][1].extend([i + 1] * nk)
            jac["t"]["coo"][2].extend(-dfdx["tf"])

            iRow += nk

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac
