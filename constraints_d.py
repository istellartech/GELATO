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
from dynamics import dynamics_velocity, dynamics_velocity_NoAir, dynamics_quaternion


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


def equality_jac_dynamics_mass(xdict, pdict, unitdict, condition):
    """Jacobian of equality_dynamics_mass."""

    jac = {}

    unit_mass = unitdict["mass"]
    unit_t = unitdict["t"]
    num_sections = pdict["num_sections"]

    jac["mass"] = {"coo": [[], [], []], "shape": (pdict["N"], pdict["M"])}
    jac["t"] = {"coo": [[], [], []], "shape": (pdict["N"], num_sections + 1)}

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)

        if pdict["params"][i]["engineOn"]:

            jac["mass"]["coo"][0].extend(
                chain.from_iterable(
                    repeat(j, n + 1) for j in range(ua, ub)
                )
            )
            jac["mass"]["coo"][1].extend(list(range(xa, xb)) * (n))
            jac["mass"]["coo"][2].extend(
                pdict["ps_params"].D(i).ravel(order="C")
            )

            jac["t"]["coo"][0].extend(range(ua, ub))
            jac["t"]["coo"][1].extend([i] * n)
            jac["t"]["coo"][2].extend(
                [-pdict["params"][i]["massflow"] / unit_mass * unit_t / 2.0] * n
            )  # rh(to)
            jac["t"]["coo"][0].extend(range(ua, ub))
            jac["t"]["coo"][1].extend([i + 1] * n)
            jac["t"]["coo"][2].extend(
                [pdict["params"][i]["massflow"] / unit_mass * unit_t / 2.0] * n
            )  # rh(tf)

        else:
            jac["mass"]["coo"][0].extend(range(ua, ub))
            jac["mass"]["coo"][1].extend([xa] * n)
            jac["mass"]["coo"][2].extend([-1.0] * n)
            jac["mass"]["coo"][0].extend(range(ua, ub))
            jac["mass"]["coo"][1].extend(range(xa + 1, xb))
            jac["mass"]["coo"][2].extend([1.0] * n)

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


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


def equality_jac_dynamics_position(xdict, pdict, unitdict, condition):
    """Jacobian of equality_dynamics_position."""

    jac = {}

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    vel_ = xdict["velocity"].reshape(-1, 3)
    t = xdict["t"]

    num_sections = pdict["num_sections"]

    jac["position"] = {"coo": [[], [], []], "shape": (pdict["N"] * 3, pdict["M"] * 3)}
    jac["velocity"] = {"coo": [[], [], []], "shape": (pdict["N"] * 3, pdict["M"] * 3)}
    jac["t"] = {"coo": [[], [], []], "shape": (pdict["N"] * 3, num_sections + 1)}

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
        vel_i_ = vel_[xa:xb]
        to = t[i]
        tf = t[i + 1]

        submat_pos = np.zeros(
            (n * 3, (n + 1) * 3)
        )  # jac["position"][ua*3:ub*3, xa*3:xb*3]
        submat_pos[::3, ::3] = pdict["ps_params"].D(i)
        submat_pos[1::3, 1::3] = pdict["ps_params"].D(i)
        submat_pos[2::3, 2::3] = pdict["ps_params"].D(i)

        rh_vel = -unit_vel * (tf - to) * unit_t / 2.0 / unit_pos  # rh vel
        jac["velocity"]["coo"][0].extend(range(ua * 3, ub * 3))
        jac["velocity"]["coo"][1].extend(range((xa + 1) * 3, xb * 3))
        jac["velocity"]["coo"][2].extend([rh_vel] * (n * 3))

        #t_o
        rh_to = vel_i_[1:].ravel() * unit_vel * unit_t / 2.0 / unit_pos  # rh to
        jac["t"]["coo"][0].extend(range(ua * 3, ub * 3))
        jac["t"]["coo"][1].extend([i] * n * 3)
        jac["t"]["coo"][2].extend(rh_to)

        #t_f
        rh_tf = -rh_to  # rh tf
        jac["t"]["coo"][0].extend(range(ua * 3, ub * 3))
        jac["t"]["coo"][1].extend([i + 1] * n * 3)
        jac["t"]["coo"][2].extend(rh_tf)

        jac["position"]["coo"][0].extend(
            chain.from_iterable(
                repeat(j, (n + 1) * 3) for j in range(ua * 3, ub * 3)
            )
        )
        jac["position"]["coo"][1].extend(
            list(range(xa * 3, xb * 3)) * (n * 3)
        )
        jac["position"]["coo"][2].extend(submat_pos.ravel())

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


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
        if param[2] == 0.0:
            rh = (
                dynamics_velocity_NoAir(
                    mass_i_[1:],
                    pos_i_[1:],
                    quat_i_[1:],
                    param,
                    units,
                )
                * (tf - to)
                * unit_t
                / 2.0
            )
        else:
            rh = (
                dynamics_velocity(
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

    jac["mass"] = {"coo": [[], [], []], "shape": (pdict["N"] * 3, pdict["M"])}
    jac["position"] = {"coo": [[], [], []], "shape": (pdict["N"] * 3, pdict["M"] * 3)}
    jac["velocity"] = {"coo": [[], [], []], "shape": (pdict["N"] * 3, pdict["M"] * 3)}
    jac["quaternion"] = {"coo": [[], [], []], "shape": (pdict["N"] * 3, pdict["M"] * 4)}
    jac["t"] = {"coo": [[], [], []], "shape": (pdict["N"] * 3, num_sections + 1)}

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

        submat_vel = np.zeros(
            (n * 3, (n + 1) * 3)
        )  # jac["velocity"][a*3:b*3, (a+i)*3:(b+i+1)*3]
        submat_vel[::3, ::3] = pdict["ps_params"].D(i)
        submat_vel[1::3, 1::3] = pdict["ps_params"].D(i)
        submat_vel[2::3, 2::3] = pdict["ps_params"].D(i)

        def dynamics(mass, pos, vel, quat, t):
            if param[2] == 0.0:
                return dynamics_velocity_NoAir(mass, pos, quat, param, units)
            else:
                return dynamics_velocity(
                    mass, pos, vel, quat, t, param, wind, ca, units
                )

        f_center = dynamics(
            mass_i_[1:],
            pos_i_[1:],
            vel_i_[1:],
            quat_i_[1:],
            t_nodes[1:],
        )

        #mass
        mass_i_[1:] += dx
        f_p = dynamics(
            mass_i_[1:],
            pos_i_[1:],
            vel_i_[1:],
            quat_i_[1:],
            t_nodes[1:],
        )
        mass_i_[1:] -= dx

        rh_mass = (
            -(f_p - f_center) / dx * (tf - to) * unit_t / 2.0
        )  # rh acc mass
        jac["mass"]["coo"][0].extend(
            chain.from_iterable(range(j * 3, (j + 1) * 3) for j in range(ua, ub))
        )
        jac["mass"]["coo"][1].extend(
            chain.from_iterable(repeat(j, 3) for j in range(xa+1, xb))
        )
        jac["mass"]["coo"][2].extend(rh_mass.ravel())


        #position
        for k in range(3):
            pos_i_[1:, k] += dx
            f_p = dynamics(
                mass_i_[1:],
                pos_i_[1:],
                vel_i_[1:],
                quat_i_[1:],
                t_nodes[1:],
            )
            pos_i_[1:, k] -= dx
            rh_pos = (
                -(f_p - f_center) / dx * (tf - to) * unit_t / 2.0
            )  # rh acc pos

            jac["position"]["coo"][0].extend(
                chain.from_iterable(range(j * 3, (j + 1) * 3) for j in range(ua, ub))
            )
            jac["position"]["coo"][1].extend(
                chain.from_iterable(repeat(j * 3 + k, 3) for j in range(xa+1, xb))
            )
            jac["position"]["coo"][2].extend(rh_pos.ravel())

        #velocity
        if param[2] > 0.0:
            for k in range(3):
                vel_i_[1:, k] += dx
                f_p = dynamics(
                    mass_i_[1:],
                    pos_i_[1:],
                    vel_i_[1:],
                    quat_i_[1:],
                    t_nodes[1:],
                )
                vel_i_[1:, k] -= dx
                rh_vel = (
                    -(f_p - f_center) / dx * (tf - to) * unit_t / 2.0
                )
                for j in range(n):
                    submat_vel[j * 3 : j * 3 + 3, (j + 1) * 3 + k] += rh_vel[j]

        jac["velocity"]["coo"][0].extend(
            chain.from_iterable(
                repeat(j, (n + 1) * 3) for j in range(ua * 3, ub * 3)
            )
        )
        jac["velocity"]["coo"][1].extend(
            list(range(xa * 3, xb * 3)) * (n * 3)
        )
        jac["velocity"]["coo"][2].extend(submat_vel.ravel())

        #quaternion
        for k in range(4):
            quat_i_[1:, k] += dx
            f_p = dynamics(
                mass_i_[1:],
                pos_i_[1:],
                vel_i_[1:],
                quat_i_[1:],
                t_nodes[1:],
            )
            quat_i_[1:, k] -= dx
            rh_quat = (
                -(f_p - f_center) / dx * (tf - to) * unit_t / 2.0
            )  # rh acc quat

            jac["quaternion"]["coo"][0].extend(
                chain.from_iterable(range(j * 3, (j + 1) * 3) for j in range(ua, ub))
            )
            jac["quaternion"]["coo"][1].extend(
                chain.from_iterable(repeat(j * 4 + k, 3) for j in range(xa+1, xb))
            )
            jac["quaternion"]["coo"][2].extend(rh_quat.ravel())

        #t_o, t_f
        to_p = to + dx

        if param[2] > 0.0:
            t_i_p1_ = pdict["ps_params"].time_nodes(i, to_p, tf)
            f_p = dynamics(
                mass_i_[1:],
                pos_i_[1:],
                vel_i_[1:],
                quat_i_[1:],
                t_i_p1_[1:],
            )
            rh_to = -(f_p * (tf - to_p) - f_center * (tf - to)).ravel() / dx * unit_t / 2.0 
            tf_p = tf + dx
            t_i_p2_ = pdict["ps_params"].time_nodes(i, to, tf_p)
            f_p = dynamics(
                mass_i_[1:],
                pos_i_[1:],
                vel_i_[1:],
                quat_i_[1:],
                t_i_p2_[1:],
            )
            rh_tf = -(f_p * (tf_p - to) - f_center * (tf - to)).ravel() / dx * unit_t / 2.0
        else:
            rh_to = f_center.ravel() * unit_t / 2.0
            rh_tf = -rh_to

        jac["t"]["coo"][0].extend(range(ua * 3, ub * 3))
        jac["t"]["coo"][1].extend([i] * n * 3)
        jac["t"]["coo"][2].extend(rh_to)

        #t_f
        jac["t"]["coo"][0].extend(range(ua * 3, ub * 3))
        jac["t"]["coo"][1].extend([i + 1] * n * 3)
        jac["t"]["coo"][2].extend(rh_tf)

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


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
                dynamics_quaternion(quat_i_[1:], u_i_, unit_u)
                * (tf - to)
                * unit_t
                / 2.0
            )
            con.append((lh - rh).ravel())

    return np.concatenate(con, axis=None)


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

    jac["quaternion"] = {"coo": [[], [], []], "shape": (pdict["N"] * 4, pdict["M"] * 4)}
    jac["u"] = {"coo": [[], [], []], "shape": (pdict["N"] * 4, pdict["N"] * 3)}
    jac["t"] = {"coo": [[], [], []], "shape": (pdict["N"] * 4, num_sections + 1)}

    for i in range(num_sections):
        ua, ub, xa, xb, n = pdict["ps_params"].get_index(i)
        quat_i_ = quat_[xa:xb]
        u_i_ = u_[ua:ub]
        to = t[i]
        tf = t[i + 1]

        if pdict["params"][i]["attitude"] in ["hold", "vertical"]:

            jac["quaternion"]["coo"][0].extend(range(ua * 4, ub * 4))
            jac["quaternion"]["coo"][1].extend(
                list(range(xa * 4, (xa + 1) * 4)) * n
            )
            jac["quaternion"]["coo"][2].extend([-1.0] * (4 * n))

            jac["quaternion"]["coo"][0].extend(range(ua * 4, ub * 4))
            jac["quaternion"]["coo"][1].extend(
                list(range((xa + 1) * 4, xb * 4))
            )
            jac["quaternion"]["coo"][2].extend([1.0] * (4 * n))

        else:
            submat_quat = np.zeros(
                (n * 4, (n + 1) * 4)
            )  # jac["quaternion"][a*4:b*4, (a+i)*4:(b+i+1)*4]
            submat_quat[::4, ::4] = pdict["ps_params"].D(i)
            submat_quat[1::4, 1::4] = pdict["ps_params"].D(i)
            submat_quat[2::4, 2::4] = pdict["ps_params"].D(i)
            submat_quat[3::4, 3::4] = pdict["ps_params"].D(i)

            f_center = dynamics_quaternion(quat_i_[1:], u_i_, unit_u)

            # quaternion
            for k in range(4):
                quat_i_[1:, k] += dx
                f_p = dynamics_quaternion(quat_i_[1:], u_i_, unit_u)
                quat_i_[1:, k] -= dx
                rh_quat = (
                    -(f_p - f_center) / dx * (tf - to) * unit_t / 2.0
                )
                for j in range(n):
                    submat_quat[j * 4 : j * 4 + 4, (j + 1) * 4 + k] += rh_quat[j]

            jac["quaternion"]["coo"][0].extend(
                chain.from_iterable(
                    repeat(j, (n + 1) * 4) for j in range(ua * 4, ub * 4)
                )
            )
            jac["quaternion"]["coo"][1].extend(
                list(range(xa * 4, xb * 4)) * (n * 4)
            )
            jac["quaternion"]["coo"][2].extend(submat_quat.ravel())

            # u (angular velocity)
            for k in range(3):
                u_i_[:, k] += dx
                f_p = dynamics_quaternion(quat_i_[1:], u_i_, unit_u)
                u_i_[:, k] -= dx
                rh_u = (
                    -(f_p - f_center) / dx * (tf - to) * unit_t / 2.0
                )
                jac["u"]["coo"][0].extend(
                    chain.from_iterable(range(j * 4, (j + 1) * 4) for j in range(ua, ub))
                )
                jac["u"]["coo"][1].extend(
                    chain.from_iterable(repeat(j * 3 + k, 4) for j in range(ua, ub))
                )
                jac["u"]["coo"][2].extend(rh_u.ravel())

            #t_o
            rh_to = f_center.ravel() * unit_t / 2.0  # rh to
            jac["t"]["coo"][0].extend(range(ua * 4, ub * 4))
            jac["t"]["coo"][1].extend([i] * n * 4)
            jac["t"]["coo"][2].extend(rh_to)

            #t_f
            rh_tf = -rh_to  # rh tf
            jac["t"]["coo"][0].extend(range(ua * 4, ub * 4))
            jac["t"]["coo"][1].extend([i + 1] * n * 4)
            jac["t"]["coo"][2].extend(rh_tf)


    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac
