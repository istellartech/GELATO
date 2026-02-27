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

# constraints_d.py
# constraints about dynamics

from itertools import chain, repeat
import numpy as np
from .dynamics_c import (
    dynamics_velocity_array,
    dynamics_quaternion_array,
    dynamics_velocity_rh_gradient,
    dynamics_quaternion_rh_gradient,
    jac_dynamics_velocity_section,
    jac_dynamics_quaternion_section,
    jac_dynamics_position_section,
    jac_dynamics_mass_section,
)


@profile
def equality_dynamics_mass(xdict, pdict, unitdict, condition):
    """Equality constraint about dynamics of mass."""

    con = []

    unit_mass = unitdict["mass"]
    unit_t = unitdict["t"]
    mass_ = xdict["mass"]
    t = xdict["t"]

    num_sections = pdict["num_sections"]

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
        mass_i_ = mass_[xa:xb]
        to = t[i]
        tf = t[i + 1]
        # t_nodes = pdict["ps_params"].tau(i) * (tf-to) / 2.0 + (tf+to) / 2.0

        if pdict["params"][i]["engineOn"]:
            lh = pdict["ps_params"].D(i).dot(mass_i_)
            rh = np.full(
                n,
                -pdict["params"][i]["massflow"] / unit_mass * (tf - to) * unit_t / 2.0,
            )  # dynamics_mass
            con.append(lh - rh)
        else:
            con.append(mass_i_[1:] - mass_i_[0])

    return np.concatenate(con, axis=None)


@profile
def equality_jac_dynamics_mass(xdict, pdict, unitdict, condition):
    """Jacobian of equality_dynamics_mass."""

    jac = {}

    unit_mass = unitdict["mass"]
    unit_t = unitdict["t"]
    num_sections = pdict["num_sections"]

    # Collect COO arrays from all sections, then concatenate once
    mass_rows, mass_cols, mass_data = [], [], []
    t_rows, t_cols, t_data = [], [], []

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)

        engine_on = pdict["params"][i]["engineOn"]
        massflow_coeff = pdict["params"][i]["massflow"] / unit_mass * unit_t / 2.0
        Di = pdict["ps_params"].D(i)

        sec = jac_dynamics_mass_section(
            n, ua, xa, i,
            engine_on, massflow_coeff,
            Di
        )

        mass_rows.append(sec["mass"]["row"])
        mass_cols.append(sec["mass"]["col"])
        mass_data.append(sec["mass"]["data"])
        t_rows.append(sec["t"]["row"])
        t_cols.append(sec["t"]["col"])
        t_data.append(sec["t"]["data"])

    jac["mass"] = {
        "coo": [
            np.concatenate(mass_rows).astype("i4"),
            np.concatenate(mass_cols).astype("i4"),
            np.concatenate(mass_data),
        ],
        "shape": (pdict["N"], pdict["M"]),
    }
    jac["t"] = {
        "coo": [
            np.concatenate(t_rows).astype("i4"),
            np.concatenate(t_cols).astype("i4"),
            np.concatenate(t_data),
        ],
        "shape": (pdict["N"], num_sections + 1),
    }

    return jac


@profile
def equality_dynamics_position(xdict, pdict, unitdict, condition):
    """Equality constraint about dynamics of position."""

    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    t = xdict["t"]

    num_sections = pdict["num_sections"]

    param = np.zeros(5)

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
        pos_i_ = pos_[xa:xb]
        vel_i_ = vel_[xa:xb]

        to = t[i]
        tf = t[i + 1]
        # t_nodes = pdict["ps_params"].tau(i) * (tf-to) / 2.0 + (tf+to) / 2.0

        param[0] = pdict["params"][i]["thrust"]
        param[1] = pdict["params"][i]["massflow"]
        param[2] = pdict["params"][i]["reference_area"]
        param[4] = pdict["params"][i]["nozzle_area"]

        lh = pdict["ps_params"].D(i).dot(pos_i_)
        rh = (
            vel_i_[1:] * unit_vel * (tf - to) * unit_t / 2.0 / unit_pos
        )  # dynamics_position
        con.append((lh - rh).ravel())

    return np.concatenate(con, axis=None)


@profile
def equality_jac_dynamics_position(xdict, pdict, unitdict, condition):
    """Jacobian of equality_dynamics_position."""

    jac = {}

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    vel_ = xdict["velocity"].reshape(-1, 3)
    t = xdict["t"]

    num_sections = pdict["num_sections"]

    # Collect COO arrays from all sections, then concatenate once
    pos_rows, pos_cols, pos_data = [], [], []
    vel_rows, vel_cols, vel_data = [], [], []
    t_rows, t_cols, t_data = [], [], []

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
        vel_i_ = vel_[xa:xb]
        to = t[i]
        tf = t[i + 1]

        Di = pdict["ps_params"].D(i)

        sec = jac_dynamics_position_section(
            n, ua, xa, i,
            vel_i_, to, tf,
            unit_vel, unit_pos, unit_t,
            Di
        )

        pos_rows.append(sec["position"]["row"])
        pos_cols.append(sec["position"]["col"])
        pos_data.append(sec["position"]["data"])
        vel_rows.append(sec["velocity"]["row"])
        vel_cols.append(sec["velocity"]["col"])
        vel_data.append(sec["velocity"]["data"])
        t_rows.append(sec["t"]["row"])
        t_cols.append(sec["t"]["col"])
        t_data.append(sec["t"]["data"])

    jac["position"] = {
        "coo": [
            np.concatenate(pos_rows).astype("i4"),
            np.concatenate(pos_cols).astype("i4"),
            np.concatenate(pos_data),
        ],
        "shape": (pdict["N"] * 3, pdict["M"] * 3),
    }
    jac["velocity"] = {
        "coo": [
            np.concatenate(vel_rows).astype("i4"),
            np.concatenate(vel_cols).astype("i4"),
            np.concatenate(vel_data),
        ],
        "shape": (pdict["N"] * 3, pdict["M"] * 3),
    }
    jac["t"] = {
        "coo": [
            np.concatenate(t_rows).astype("i4"),
            np.concatenate(t_cols).astype("i4"),
            np.concatenate(t_data),
        ],
        "shape": (pdict["N"] * 3, num_sections + 1),
    }

    return jac


@profile
def equality_dynamics_velocity(xdict, pdict, unitdict, condition):
    """Equality constraint about dynamics of velocity."""

    con = []

    unit_mass = unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    mass_ = xdict["mass"]
    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)
    t = xdict["t"]

    units = np.array([unit_mass, unit_pos, unit_vel])

    num_sections = pdict["num_sections"]

    param = np.zeros(5)

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
        mass_i_ = mass_[xa:xb]
        pos_i_ = pos_[xa:xb]
        vel_i_ = vel_[xa:xb]
        quat_i_ = quat_[xa:xb]

        to = t[i]
        tf = t[i + 1]
        t_nodes = pdict["ps_params"].time_nodes(i, to, tf)

        param[0] = pdict["params"][i]["thrust"]
        param[1] = pdict["params"][i]["massflow"]
        param[2] = pdict["params"][i]["reference_area"]
        param[4] = pdict["params"][i]["nozzle_area"]

        wind = pdict["wind_table"]
        ca = pdict["ca_table"]

        lh = pdict["ps_params"].D(i).dot(vel_i_)
        rh = (
            dynamics_velocity_array(
                mass_i_[1:],
                pos_i_[1:],
                vel_i_[1:],
                quat_i_[1:],
                t_nodes[1:],
                param,
                wind,
                ca,
                units,
            )
            * (tf - to)
            * unit_t
            / 2.0
        )
        con.append((lh - rh).ravel())

    return np.concatenate(con, axis=None)


@profile
def equality_jac_dynamics_velocity(xdict, pdict, unitdict, condition):
    """Jacobian of equality_dynamics_velocity."""

    jac = {}
    dx = pdict["dx"]

    unit_mass = unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    mass_ = xdict["mass"]
    pos_ = xdict["position"].reshape(-1, 3)
    vel_ = xdict["velocity"].reshape(-1, 3)
    quat_ = xdict["quaternion"].reshape(-1, 4)
    t = xdict["t"]

    units = np.array([unit_mass, unit_pos, unit_vel])

    num_sections = pdict["num_sections"]

    param = np.zeros(5)

    # Collect COO arrays from all sections, then concatenate once
    mass_rows, mass_cols, mass_data = [], [], []
    pos_rows, pos_cols, pos_data = [], [], []
    vel_rows, vel_cols, vel_data = [], [], []
    quat_rows, quat_cols, quat_data = [], [], []
    t_rows, t_cols, t_data = [], [], []

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
        mass_i_ = mass_[xa:xb]
        pos_i_ = pos_[xa:xb]
        vel_i_ = vel_[xa:xb]
        quat_i_ = quat_[xa:xb]
        to = t[i]
        tf = t[i + 1]
        t_nodes = pdict["ps_params"].time_nodes(i, to, tf)

        param[0] = pdict["params"][i]["thrust"]
        param[1] = pdict["params"][i]["massflow"]
        param[2] = pdict["params"][i]["reference_area"]
        param[4] = pdict["params"][i]["nozzle_area"]

        wind = pdict["wind_table"]
        ca = pdict["ca_table"]

        Di = pdict["ps_params"].D(i)

        sec = jac_dynamics_velocity_section(
            n, ua, xa, i,
            mass_i_, pos_i_, vel_i_, quat_i_,
            t_nodes, param, wind, ca, units,
            to, tf, unit_t, dx, Di
        )

        mass_rows.append(sec["mass"]["row"])
        mass_cols.append(sec["mass"]["col"])
        mass_data.append(sec["mass"]["data"])
        pos_rows.append(sec["position"]["row"])
        pos_cols.append(sec["position"]["col"])
        pos_data.append(sec["position"]["data"])
        vel_rows.append(sec["velocity"]["row"])
        vel_cols.append(sec["velocity"]["col"])
        vel_data.append(sec["velocity"]["data"])
        quat_rows.append(sec["quaternion"]["row"])
        quat_cols.append(sec["quaternion"]["col"])
        quat_data.append(sec["quaternion"]["data"])
        t_rows.append(sec["t"]["row"])
        t_cols.append(sec["t"]["col"])
        t_data.append(sec["t"]["data"])

    jac["mass"] = {
        "coo": [
            np.concatenate(mass_rows).astype("i4"),
            np.concatenate(mass_cols).astype("i4"),
            np.concatenate(mass_data),
        ],
        "shape": (pdict["N"] * 3, pdict["M"]),
    }
    jac["position"] = {
        "coo": [
            np.concatenate(pos_rows).astype("i4"),
            np.concatenate(pos_cols).astype("i4"),
            np.concatenate(pos_data),
        ],
        "shape": (pdict["N"] * 3, pdict["M"] * 3),
    }
    jac["velocity"] = {
        "coo": [
            np.concatenate(vel_rows).astype("i4"),
            np.concatenate(vel_cols).astype("i4"),
            np.concatenate(vel_data),
        ],
        "shape": (pdict["N"] * 3, pdict["M"] * 3),
    }
    jac["quaternion"] = {
        "coo": [
            np.concatenate(quat_rows).astype("i4"),
            np.concatenate(quat_cols).astype("i4"),
            np.concatenate(quat_data),
        ],
        "shape": (pdict["N"] * 3, pdict["M"] * 4),
    }
    jac["t"] = {
        "coo": [
            np.concatenate(t_rows).astype("i4"),
            np.concatenate(t_cols).astype("i4"),
            np.concatenate(t_data),
        ],
        "shape": (pdict["N"] * 3, num_sections + 1),
    }

    return jac


@profile
def equality_dynamics_quaternion(xdict, pdict, unitdict, condition):
    """Equality constraint about dynamics of quaternion."""

    con = []

    unit_u = unitdict["u"]
    unit_t = unitdict["t"]
    quat_ = xdict["quaternion"].reshape(-1, 4)
    u_ = xdict["u"].reshape(-1, 3)
    t = xdict["t"]

    num_sections = pdict["num_sections"]

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
        quat_i_ = quat_[xa:xb]
        u_i_ = u_[ua:ub]

        to = t[i]
        tf = t[i + 1]
        # t_nodes = pdict["ps_params"].tau(i) * (tf-to) / 2.0 + (tf+to) / 2.0

        if pdict["params"][i]["attitude"] in ["hold", "vertical"]:
            con.append((quat_i_[1:] - quat_i_[0]).ravel())
        else:
            lh = pdict["ps_params"].D(i).dot(quat_i_)
            rh = (
                dynamics_quaternion_array(quat_i_[1:], u_i_, unit_u)
                * (tf - to)
                * unit_t
                / 2.0
            )
            con.append((lh - rh).ravel())

    return np.concatenate(con, axis=None)


@profile
def equality_jac_dynamics_quaternion(xdict, pdict, unitdict, condition):
    """Jacobian of equality_dynamics_quaternion."""

    jac = {}
    dx = pdict["dx"]

    unit_u = unitdict["u"]
    unit_t = unitdict["t"]
    quat_ = xdict["quaternion"].reshape(-1, 4)
    u_ = xdict["u"].reshape(-1, 3)
    t = xdict["t"]

    num_sections = pdict["num_sections"]

    # Collect COO arrays from all sections, then concatenate once
    quat_rows, quat_cols, quat_data = [], [], []
    u_rows, u_cols, u_data = [], [], []
    t_rows, t_cols, t_data = [], [], []

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
        quat_i_ = quat_[xa:xb]
        u_i_ = u_[ua:ub]
        to = t[i]
        tf = t[i + 1]

        is_hold = pdict["params"][i]["attitude"] in ["hold", "vertical"]
        Di = pdict["ps_params"].D(i)

        sec = jac_dynamics_quaternion_section(
            n, ua, xa, i,
            quat_i_, u_i_,
            unit_u, to, tf, unit_t, dx,
            Di, is_hold
        )

        quat_rows.append(sec["quaternion"]["row"])
        quat_cols.append(sec["quaternion"]["col"])
        quat_data.append(sec["quaternion"]["data"])
        u_rows.append(sec["u"]["row"])
        u_cols.append(sec["u"]["col"])
        u_data.append(sec["u"]["data"])
        t_rows.append(sec["t"]["row"])
        t_cols.append(sec["t"]["col"])
        t_data.append(sec["t"]["data"])

    jac["quaternion"] = {
        "coo": [
            np.concatenate(quat_rows).astype("i4"),
            np.concatenate(quat_cols).astype("i4"),
            np.concatenate(quat_data),
        ],
        "shape": (pdict["N"] * 4, pdict["M"] * 4),
    }
    jac["u"] = {
        "coo": [
            np.concatenate(u_rows).astype("i4"),
            np.concatenate(u_cols).astype("i4"),
            np.concatenate(u_data),
        ],
        "shape": (pdict["N"] * 4, pdict["N"] * 3),
    }
    jac["t"] = {
        "coo": [
            np.concatenate(t_rows).astype("i4"),
            np.concatenate(t_cols).astype("i4"),
            np.concatenate(t_data),
        ],
        "shape": (pdict["N"] * 4, num_sections + 1),
    }

    return jac
