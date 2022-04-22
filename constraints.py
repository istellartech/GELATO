import sys
from copy import copy, deepcopy
import numpy as np
from scipy import sparse
from numba import jit
from utils import *
from USStandardAtmosphere import *
from coordinate import *
from user_constraints import *


@jit(nopython=True)
def dynamics_velocity(mass_e, pos_eci_e, vel_eci_e, quat_eci2body, t, param, wind, ca, units):

    mass = mass_e * units[0]
    pos_eci = pos_eci_e * units[1]
    vel_eci = vel_eci_e * units[2]
    acc_eci = np.zeros(vel_eci_e.shape)

    thrust_vac_n = param[0]
    airArea_m2 = param[2]
    nozzleArea_m2 = param[4]

    for i in range(len(mass)):
        pos_llh = ecef2geodetic(pos_eci[i,0],pos_eci[i,1],pos_eci[i,2])
        altitude_m = geopotential_altitude(pos_llh[2])
        rho = airdensity_at(altitude_m)
        p = airpressure_at(altitude_m)

        vel_ecef = vel_eci2ecef(vel_eci[i], pos_eci[i], t[i])
        vel_wind_ned = wind_ned(altitude_m, wind)

        vel_wind_eci = quatrot(quat_nedg2eci(pos_eci[i], t[i]), vel_wind_ned)
        vel_air_eci = ecef2eci(vel_ecef, t[i]) - vel_wind_eci
        mach_number = norm(vel_air_eci) / speed_of_sound(altitude_m)

        airAxialForce_coeff = np.interp(mach_number, ca[:,0], ca[:,1])

        aero_n_eci = 0.5 * rho * norm(vel_air_eci) * -vel_air_eci * airArea_m2 * airAxialForce_coeff

        thrust_n = thrust_vac_n - nozzleArea_m2 * p
        thrustdir_eci = quatrot(conj(quat_eci2body[i]), np.array([1.0, 0.0, 0.0]))
        thrust_n_eci = thrustdir_eci * thrust_n
        gravity_eci = gravity(pos_eci[i])

        acc_eci[i] = gravity_eci + (thrust_n_eci + aero_n_eci) / mass[i]

    return acc_eci / units[2]

@jit(nopython=True)
def dynamics_quaternion(quat_eci2body, u_e, unit_u):

    u = u_e * unit_u

    d_quat = np.zeros(quat_eci2body.shape)
    for i in range(len(u)):
        omega_rps_body = np.deg2rad(np.array([0.0, u[i,0], u[i,1], u[i,2]]))
        d_quat[i] = 0.5 * quatmult(quat_eci2body[i], omega_rps_body)    

    return d_quat



def equality_init(xdict, pdict, unitdict, condition):

    con = []
    mass_ = xdict["mass"]
    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)

    #initial condition
    if condition["OptimizationMode"] != "Payload":
        con.append(mass_[0] - condition["init"]["mass"] / unitdict["mass"])
    con.append(pos_[0] - condition["init"]["position"] / unitdict["position"])
    con.append(vel_[0] - condition["init"]["velocity"] / unitdict["velocity"])
    con.append(quat_[0] - condition["init"]["quaternion"])

    return np.concatenate(con, axis=None)


def equality_jac_init(xdict, pdict, unitdict, condition):
    jac = {}

    if condition["OptimizationMode"] == "Payload":
        rowcol_pos = (range(0,3), range(0,3))
        rowcol_vel = (range(3,6), range(0,3))
        rowcol_quat = (range(6,10), range(0,4))

        jac["position"]   = sparse.coo_matrix(([1.0]*3, rowcol_pos),  shape=(10, pdict["M"]*3))
        jac["velocity"]   = sparse.coo_matrix(([1.0]*3, rowcol_vel),  shape=(10, pdict["M"]*3))
        jac["quaternion"] = sparse.coo_matrix(([1.0]*4, rowcol_quat), shape=(10, pdict["M"]*4))

    else:
        rowcol_mass = ([0], [0])
        rowcol_pos = (range(1,4), range(0,3))
        rowcol_vel = (range(4,7), range(0,3))
        rowcol_quat = (range(7,11), range(0,4))

        jac["mass"]       = sparse.coo_matrix(([1.0]  , rowcol_mass), shape=(11, pdict["M"]))
        jac["position"]   = sparse.coo_matrix(([1.0]*3, rowcol_pos),  shape=(11, pdict["M"]*3))
        jac["velocity"]   = sparse.coo_matrix(([1.0]*3, rowcol_vel),  shape=(11, pdict["M"]*3))
        jac["quaternion"] = sparse.coo_matrix(([1.0]*4, rowcol_quat), shape=(11, pdict["M"]*4))


    return jac

def equality_time(xdict, pdict, unitdict, condition):
    con = []
    unit_t = unitdict["t"]

    t_ = xdict["t"]

    num_sections = pdict["num_sections"]

    #knotting time
    con.append([t_[i] - pdict["params"][i]["timeAt_sec"] / unit_t for i in range(num_sections+1) if pdict["params"][i]["timeFixed"]])

    return np.concatenate(con, axis=None)

def equality_jac_time(xdict, pdict, unitdict, condition):
    jac = {}

    data = []
    row = []
    col = []

    counter = 0
    for i in range(pdict["num_sections"]+1):
        if pdict["params"][i]["timeFixed"]:
            data.append(1.0)
            row.append(counter)
            col.append(i)
            counter += 1

    jac["t"] = sparse.coo_matrix((data, (row, col)), shape=[len(data), len(xdict["t"])])

    return jac

def equality_dynamics_mass(xdict, pdict, unitdict, condition):
    con = []

    unit_mass = unitdict["mass"]
    unit_t = unitdict["t"]
    mass_ = xdict["mass"]
    t = xdict["t"]

    num_sections = pdict["num_sections"]

    param = np.zeros(5)

    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        mass_i_ = mass_[a+i:b+i+1]
        to = t[i]
        tf = t[i+1]
        # t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0

        if pdict["params"][i]["engineOn"]:    
            lh = pdict["ps_params"][i]["D"].dot(mass_i_)
            rh = np.full(n, -pdict["params"][i]["massflow_kgps"] / unit_mass * (tf-to) * unit_t / 2.0 ) #dynamics_mass
            con.append(lh - rh)
        else:
            con.append(mass_i_[1:] - mass_i_[0])

    return np.concatenate(con, axis=None)

def equality_jac_dynamics_mass(xdict, pdict, unitdict, condition):
    jac = {}

    unit_mass = unitdict["mass"]
    unit_t = unitdict["t"]
    num_sections = pdict["num_sections"]

    jac["mass"] = sparse.lil_matrix((pdict["N"], pdict["M"]))
    jac["t"] = sparse.lil_matrix((pdict["N"], num_sections+1))

    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        if pdict["params"][i]["engineOn"]:
            jac["mass"][a:b, a+i:b+i+1] = pdict["ps_params"][i]["D"] # lh
            jac["t"][a:b, i]   = -pdict["params"][i]["massflow_kgps"] / unit_mass * unit_t / 2.0 # rh(to)
            jac["t"][a:b, i+1] =  pdict["params"][i]["massflow_kgps"] / unit_mass * unit_t / 2.0 # rh(tf)

        else:
            jac["mass"][a:b, a+i] = -1.0
            jac["mass"][a:b, a+i+1:b+i+1] = np.eye(n)

    return jac

def equality_dynamics_position(xdict, pdict, unitdict, condition):
    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    t = xdict["t"]

    num_sections = pdict["num_sections"]

    param = np.zeros(5)

    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        to = t[i]
        tf = t[i+1]
        # t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0

        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]

        lh = pdict["ps_params"][i]["D"].dot(pos_i_)
        rh = vel_i_[1:] * unit_vel * (tf-to) * unit_t / 2.0 / unit_pos #dynamics_position
        con.append((lh - rh).ravel())

    return np.concatenate(con, axis=None)

def equality_jac_dynamics_position(xdict, pdict, unitdict, condition):
    jac = {}

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    t = xdict["t"]

    num_sections = pdict["num_sections"]

    jac["position"] = np.zeros((pdict["N"]*3, pdict["M"]*3))
    jac["velocity"] = {"coo": [[], [], []], "shape":(pdict["N"]*3, pdict["M"]*3)}
    jac["t"] = {"coo": [[], [], []], "shape":(pdict["N"]*3, num_sections+1)}


    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        to = t[i]
        tf = t[i+1]

        jac["position"][a*3  : b*3  : 3, (a+i)*3  : (b+i+1)*3  : 3] = pdict["ps_params"][i]["D"] # lh x
        jac["position"][a*3+1: b*3+1: 3, (a+i)*3+1: (b+i+1)*3+1: 3] = pdict["ps_params"][i]["D"] # lh y
        jac["position"][a*3+2: b*3+2: 3, (a+i)*3+2: (b+i+1)*3+2: 3] = pdict["ps_params"][i]["D"] # lh z

        rh_vel = (-unit_vel * (tf-to) * unit_t / 2.0 / unit_pos) # rh vel
        jac["velocity"]["coo"][0].extend(list(range(a*3,b*3)))
        jac["velocity"]["coo"][1].extend(list(range((a+i+1)*3,(b+i+1)*3)))
        jac["velocity"]["coo"][2].extend([rh_vel] * (n*3))

        rh_to = vel_i_[1:].ravel() * unit_vel * unit_t / 2.0 / unit_pos # rh to
        rh_tf = -rh_to  # rh tf
        jac["t"]["coo"][0].extend(sum([[k]*2 for k in range(a*3,b*3)], []))
        jac["t"]["coo"][1].extend([i, i+1] * n*3)
        jac["t"]["coo"][2].extend(sum([[rh_to[k], rh_tf[k]] for k in range(3*n)], []))

    for key in ["velocity", "t"]:
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")


    return jac

def equality_dynamics_velocity(xdict, pdict, unitdict, condition):
    con = []

    unit_mass = unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    mass_ = xdict["mass"]
    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)
    t = xdict["t"]

    units = np.array([unit_mass, unit_pos, unit_vel])

    num_sections = pdict["num_sections"]

    param = np.zeros(5)

    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        mass_i_ = mass_[a+i:b+i+1]
        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        quat_i_ = quat_[a+i:b+i+1]
        to = t[i]
        tf = t[i+1]
        t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) * unit_t / 2.0 + (tf+to) * unit_t / 2.0

        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]

        wind = pdict["wind_table"]
        ca = pdict["ca_table"]

        lh = pdict["ps_params"][i]["D"].dot(vel_i_)
        rh = dynamics_velocity(mass_i_[1:], pos_i_[1:], vel_i_[1:], quat_i_[1:], t_nodes, param, wind, ca, units) * (tf-to) * unit_t / 2.0
        con.append((lh - rh).ravel())

    return np.concatenate(con, axis=None)

def equality_jac_dynamics_velocity(xdict, pdict, unitdict, condition):
    jac = {}
    dx = 1.0e-8

    unit_mass = unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    mass_ = xdict["mass"]
    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)
    t = xdict["t"]

    units = np.array([unit_mass, unit_pos, unit_vel])

    num_sections = pdict["num_sections"]

    param = np.zeros(5)

    jac["mass"] = {"coo": [[], [], []], "shape":(pdict["N"]*3, pdict["M"])}
    jac["position"] = {"coo": [[], [], []], "shape":(pdict["N"]*3, pdict["M"]*3)}
    jac["velocity"] = np.zeros((pdict["N"]*3, pdict["M"]*3))
    jac["quaternion"] = {"coo": [[], [], []], "shape":(pdict["N"]*3, pdict["M"]*4)}
    jac["t"] = {"coo": [[], [], []], "shape":(pdict["N"]*3, num_sections+1)}

    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        mass_i_ = mass_[a+i:b+i+1]
        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        quat_i_ = quat_[a+i:b+i+1]
        to = t[i]
        tf = t[i+1]
        t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) * unit_t / 2.0 + (tf+to) * unit_t / 2.0

        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]

        wind = pdict["wind_table"]
        ca = pdict["ca_table"]

        jac["velocity"][a*3  : b*3  : 3, (a+i)*3  : (b+i+1)*3  : 3] = pdict["ps_params"][i]["D"] # lh x
        jac["velocity"][a*3+1: b*3+1: 3, (a+i)*3+1: (b+i+1)*3+1: 3] = pdict["ps_params"][i]["D"] # lh y
        jac["velocity"][a*3+2: b*3+2: 3, (a+i)*3+2: (b+i+1)*3+2: 3] = pdict["ps_params"][i]["D"] # lh z

        f_center = dynamics_velocity(mass_i_[1:], pos_i_[1:], vel_i_[1:], quat_i_[1:], t_nodes, param, wind, ca, units)

        for j in range(n):
            mass_i_p_ = deepcopy(mass_i_)
            mass_i_p_[j+1] += dx
            f_p = dynamics_velocity(mass_i_p_[1:], pos_i_[1:], vel_i_[1:], quat_i_[1:], t_nodes, param, wind, ca, units)
            rh_mass = -(f_p[j] - f_center[j]) / dx * (tf-to) * unit_t / 2.0 # rh acc mass
            jac["mass"]["coo"][0].extend(list(range((a+j)*3, (a+j+1)*3)))
            jac["mass"]["coo"][1].extend([(a+i+j+1)] * 3)
            jac["mass"]["coo"][2].extend(rh_mass.tolist())

            for k in range(3):
                pos_i_p_ = deepcopy(pos_i_)
                pos_i_p_[j+1, k] += dx
                f_p = dynamics_velocity(mass_i_[1:], pos_i_p_[1:], vel_i_[1:], quat_i_[1:], t_nodes, param, wind, ca, units)
                rh_pos = -(f_p[j] - f_center[j]) / dx * (tf-to) * unit_t / 2.0 # rh acc pos
                jac["position"]["coo"][0].extend(list(range((a+j)*3, (a+j+1)*3)))
                jac["position"]["coo"][1].extend([(a+i+j+1)*3+k] * 3)
                jac["position"]["coo"][2].extend(rh_pos.tolist())
            
            for k in range(3):
                vel_i_p_ = deepcopy(vel_i_)
                vel_i_p_[j+1, k] += dx
                f_p = dynamics_velocity(mass_i_[1:], pos_i_[1:], vel_i_p_[1:], quat_i_[1:], t_nodes, param, wind, ca, units)
                jac["velocity"][(a+j)*3,   (a+i+j+1)*3+k] += -(f_p[j,0] - f_center[j,0]) / dx * (tf-to) * unit_t / 2.0 # rh acc_x vel
                jac["velocity"][(a+j)*3+1, (a+i+j+1)*3+k] += -(f_p[j,1] - f_center[j,1]) / dx * (tf-to) * unit_t / 2.0 # rh acc_y vel
                jac["velocity"][(a+j)*3+2, (a+i+j+1)*3+k] += -(f_p[j,2] - f_center[j,2]) / dx * (tf-to) * unit_t / 2.0 # rh acc_z vel

            for k in range(4):
                quat_i_p_ = deepcopy(quat_i_)
                quat_i_p_[j+1, k] += dx
                f_p = dynamics_velocity(mass_i_[1:], pos_i_[1:], vel_i_[1:], quat_i_p_[1:], t_nodes, param, wind, ca, units)
                rh_quat = -(f_p[j] - f_center[j]) / dx * (tf-to) * unit_t / 2.0 # rh acc quat
                jac["quaternion"]["coo"][0].extend(list(range((a+j)*3, (a+j+1)*3)))
                jac["quaternion"]["coo"][1].extend([(a+i+j+1)*4+k] * 3)
                jac["quaternion"]["coo"][2].extend(rh_quat.tolist())


        rh_to =  f_center.ravel() * unit_t / 2.0  # rh to
        rh_tf = -rh_to  # rh tf
        jac["t"]["coo"][0].extend(sum([[k]*2 for k in range(a*3,b*3)], []))
        jac["t"]["coo"][1].extend([i, i+1] * n*3)
        jac["t"]["coo"][2].extend(sum([[rh_to[k], rh_tf[k]] for k in range(3*n)], []))

    for key in ["mass", "position", "quaternion", "t"]:
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac



def equality_dynamics_quaternion(xdict, pdict, unitdict, condition):
    con = []

    unit_u = unitdict["u"]
    unit_t = unitdict["t"]
    quat_ = xdict["quaternion"].reshape(-1,4)
    u_ = xdict["u"].reshape(-1,3)
    t = xdict["t"]

    num_sections = pdict["num_sections"]

    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        quat_i_ = quat_[a+i:b+i+1]
        u_i_ = u_[a:b]
        to = t[i]
        tf = t[i+1]
        # t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0

        if pdict["params"][i]["attitude"] in ["hold", "vertical"]:
            con.append((quat_i_[1:] - quat_i_[0]).ravel())
        else:
            lh = pdict["ps_params"][i]["D"].dot(quat_i_)
            rh = dynamics_quaternion(quat_i_[1:], u_i_, unit_u) * (tf-to) * unit_t / 2.0
            con.append((lh - rh).ravel())

    return np.concatenate(con, axis=None)

def equality_jac_dynamics_quaternion(xdict, pdict, unitdict, condition):
    jac = {}
    dx = 1.0e-8

    unit_u = unitdict["u"]
    unit_t = unitdict["t"]
    quat_ = xdict["quaternion"].reshape(-1,4)
    u_ = xdict["u"].reshape(-1,3)
    t = xdict["t"]

    num_sections = pdict["num_sections"]

    jac["quaternion"] = np.zeros((pdict["N"]*4, pdict["M"]*4))
    jac["u"] = np.zeros((pdict["N"]*4, pdict["N"]*3))
    jac["t"] = {"coo": [[], [], []], "shape":(pdict["N"]*4, num_sections+1)}


    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        quat_i_ = quat_[a+i:b+i+1]
        u_i_ = u_[a:b]
        to = t[i]
        tf = t[i+1]

        if pdict["params"][i]["attitude"] in ["hold", "vertical"]:

            for j in range(n):
                jac["quaternion"][(a+j)*4:(a+j+1)*4, (a+i)*4:(a+i+1)*4] = -np.eye(4)
                jac["quaternion"][(a+j)*4:(a+j+1)*4, (a+i+j+1)*4:(a+i+j+2)*4] = np.eye(4)


        else:
            jac["quaternion"][a*4  : b*4  : 4, (a+i)*4  : (b+i+1)*4  : 4] = pdict["ps_params"][i]["D"] # lh q0
            jac["quaternion"][a*4+1: b*4+1: 4, (a+i)*4+1: (b+i+1)*4+1: 4] = pdict["ps_params"][i]["D"] # lh q1
            jac["quaternion"][a*4+2: b*4+2: 4, (a+i)*4+2: (b+i+1)*4+2: 4] = pdict["ps_params"][i]["D"] # lh q2
            jac["quaternion"][a*4+3: b*4+3: 4, (a+i)*4+3: (b+i+1)*4+3: 4] = pdict["ps_params"][i]["D"] # lh q3

            f_center = dynamics_quaternion(quat_i_[1:], u_i_, unit_u)

            for j in range(n):

                for k in range(4):
                    quat_i_p_ = deepcopy(quat_i_)
                    quat_i_p_[j+1, k] += dx
                    f_p = dynamics_quaternion(quat_i_p_[1:], u_i_, unit_u)
                    jac["quaternion"][(a+j)*4,   (a+i+j+1)*4+k] += -(f_p[j,0] - f_center[j,0]) / dx * (tf-to) * unit_t / 2.0 # rh q0 quat
                    jac["quaternion"][(a+j)*4+1, (a+i+j+1)*4+k] += -(f_p[j,1] - f_center[j,1]) / dx * (tf-to) * unit_t / 2.0 # rh q1 quat
                    jac["quaternion"][(a+j)*4+2, (a+i+j+1)*4+k] += -(f_p[j,2] - f_center[j,2]) / dx * (tf-to) * unit_t / 2.0 # rh q2 quat
                    jac["quaternion"][(a+j)*4+3, (a+i+j+1)*4+k] += -(f_p[j,3] - f_center[j,3]) / dx * (tf-to) * unit_t / 2.0 # rh q3 quat

                for k in range(3):
                    u_i_p_ = deepcopy(u_i_)
                    u_i_p_[j, k] += dx
                    f_p = dynamics_quaternion(quat_i_[1:], u_i_p_, unit_u)
                    jac["u"][(a+j)*4,   (a+j)*3+k] = -(f_p[j,0] - f_center[j,0]) / dx * (tf-to) * unit_t / 2.0 # rh q0 quat
                    jac["u"][(a+j)*4+1, (a+j)*3+k] = -(f_p[j,1] - f_center[j,1]) / dx * (tf-to) * unit_t / 2.0 # rh q1 quat
                    jac["u"][(a+j)*4+2, (a+j)*3+k] = -(f_p[j,2] - f_center[j,2]) / dx * (tf-to) * unit_t / 2.0 # rh q2 quat
                    jac["u"][(a+j)*4+3, (a+j)*3+k] = -(f_p[j,3] - f_center[j,3]) / dx * (tf-to) * unit_t / 2.0 # rh q3 quat

            rh_to =  f_center.ravel() * unit_t / 2.0  # rh to
            rh_tf = -rh_to  # rh tf
            jac["t"]["coo"][0].extend(sum([[k]*2 for k in range(a*4,b*4)], []))
            jac["t"]["coo"][1].extend([i, i+1] * n*4)
            jac["t"]["coo"][2].extend(sum([[rh_to[k], rh_tf[k]] for k in range(4*n)], []))

    jac["t"]["coo"][0] = np.array(jac["t"]["coo"][0], dtype="i4")
    jac["t"]["coo"][1] = np.array(jac["t"]["coo"][1], dtype="i4")
    jac["t"]["coo"][2] = np.array(jac["t"]["coo"][2], dtype="f8")
                        
    return jac


def equality_knot_LGR(xdict, pdict, unitdict, condition):
    con = []

    mass_ = xdict["mass"]
    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)

    num_sections = pdict["num_sections"]


    param = np.zeros(5)

    for i in range(num_sections-1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        mass_i_ = mass_[a+i:b+i+1]
        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        quat_i_ = quat_[a+i:b+i+1]

        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]

        # knotting constraints: 現在のsectionの末尾と次のsectionの先頭の連続性
        mass_next_ = mass_[b+i+1]
        mass_final_ = mass_i_[-1]
        con.append(mass_next_ - mass_final_ + pdict["params"][i+1]["mass_jettison_kg"] / unitdict["mass"])

        pos_next_ = pos_[b+i+1]
        pos_final_ = pos_i_[-1]
        con.append(pos_next_ - pos_final_)

        vel_next_ = vel_[b+i+1]
        vel_final_ = vel_i_[-1]
        con.append(vel_next_ - vel_final_)

        quat_next_ = quat_[b+i+1]
        quat_final_ = quat_i_[-1]
        con.append(quat_next_ - quat_final_)

    return np.concatenate(con, axis=None)

def equality_jac_knot_LGR(xdict, pdict, unitdict, condition):
    jac = {}

    num_sections = pdict["num_sections"]

    jac["mass"]       = sparse.lil_matrix(((num_sections-1)*11, pdict["M"]))
    jac["position"]   = sparse.lil_matrix(((num_sections-1)*11, pdict["M"]*3))
    jac["velocity"]   = sparse.lil_matrix(((num_sections-1)*11, pdict["M"]*3))
    jac["quaternion"] = sparse.lil_matrix(((num_sections-1)*11, pdict["M"]*4))

    for i in range(num_sections-1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        jac["mass"][i*11, b+i]   = -1.0
        jac["mass"][i*11, b+i+1] = 1.0

        jac["position"][i*11+1:i*11+4, (b+i)*3:(b+i+1)*3] = -np.eye(3)
        jac["position"][i*11+1:i*11+4, (b+i+1)*3:(b+i+2)*3] = np.eye(3)

        jac["velocity"][i*11+4:i*11+7, (b+i)*3:(b+i+1)*3] = -np.eye(3)
        jac["velocity"][i*11+4:i*11+7, (b+i+1)*3:(b+i+2)*3] = np.eye(3)

        jac["quaternion"][i*11+7:i*11+11, (b+i)*4:(b+i+1)*4] = -np.eye(4)
        jac["quaternion"][i*11+7:i*11+11, (b+i+1)*4:(b+i+2)*4] = np.eye(4)

    return jac

def equality_6DoF_LGR_terminal(xdict, pdict, unitdict, condition):
    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]

    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)

    # terminal conditions

    pos_f = pos_[-1] * unit_pos
    vel_f = vel_[-1] * unit_vel

    elem = orbital_elements(pos_f, vel_f)

    if condition["hp_m"] is not None:
        hp = elem[0] * (1.0 - elem[1]) + 6378137
        con.append((hp - condition["hp_m"]) / unit_pos)

    if condition["ha_m"] is not None:
        ha = elem[0] * (1.0 + elem[1]) + 6378137
        con.append((ha - condition["ha_m"]) / unit_pos)

    if condition["rf_m"] is not None:
        rf = norm(pos_f)
        con.append((rf - condition["rf_m"]) / unit_pos)

    if condition["vtf_mps"] is not None:
        vrf = vel_f.dot(normalize(pos_f))
        vtf = sqrt(norm(vel_f)**2 - vrf**2)
        con.append((vtf - condition["vtf_mps"]) / unit_vel)

    if condition["vf_elev_deg"] is not None:
        cos_vf_angle = normalize(vel_f).dot(normalize(pos_f))
        con.append(cos(radians(90.0-condition["vf_elev_deg"])) - cos_vf_angle)

    if condition["vrf_mps"] is not None:
        vrf = vel_f.dot(normalize(pos_f))
        con.append((vrf - condition["vrf_mps"]) / unit_vel)

    if condition["inclination_deg"] is not None:
        con.append((elem[2] - condition["inclination_deg"]) / 90.0)            


    return np.concatenate(con, axis=None)

def equality_jac_6DoF_LGR_terminal(xdict, pdict, unitdict, condition):
    jac = {}
    dx = 1.0e-8

    f_center = equality_6DoF_LGR_terminal(xdict, pdict, unitdict, condition)

    if hasattr(f_center, "__len__"):
        nRow = len(f_center)
    else:
        nRow = 1
    
    jac_base = {
        "position" : np.zeros((nRow, pdict["M"]*3)),
        "velocity" : np.zeros((nRow, pdict["M"]*3))
    }


    for key in ["position", "velocity"]:

        jac_base[key][:,-3:] = 1.0
        jac_base_coo = sparse.coo_matrix(jac_base[key])
        jac[key] = {
            "coo" : [jac_base_coo.row, jac_base_coo.col, jac_base_coo.data],
            "shape" : jac_base[key].shape
        }

        for j in range(-3,0):
            xdict_p = deepcopy(xdict)
            xdict_p[key][j] += dx
            f_p = equality_6DoF_LGR_terminal(xdict_p, pdict, unitdict, condition)
            jac_base[key][:, j] = (f_p - f_center) / dx
        
        for i in range(len(jac[key]["coo"][0])):
            jac[key]["coo"][2][i] = jac_base[key][jac[key]["coo"][0][i], jac[key]["coo"][1][i]]

    return jac

def equality_6DoF_rate(xdict, pdict, unitdict, condition):
    con = []

    unit_pos = unitdict["position"]

    pos_ = xdict["position"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)

    u_ = xdict["u"].reshape(-1,3)

    num_sections = pdict["num_sections"]


    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        pos_i_ = pos_[a+i:b+i+1]
        quat_i_ = quat_[a+i:b+i+1]
        u_i_ = u_[a:b]


        #rate constraint

        att = pdict["params"][i]["attitude"]

        # attitude hold : angular velocity is zero
        if att in ["hold", "vertical"]:
            con.append(u_i_)


        # kick-turn : pitch rate constant, roll/yaw rate is zero
        elif att == "kick-turn" or att == "pitch":
            con.append(u_i_[:,0])
            con.append(u_i_[:,2])
            con.append(u_i_[1:,1] - u_i_[0,1])

        # pitch-yaw : pitch/yaw constant, roll ANGLE is zero
        elif att == "pitch-yaw":
            con.append(u_i_[1:,1] - u_i_[0,1])
            con.append(u_i_[1:,2] - u_i_[0,2])
            con.append(roll_direction_array(pos_i_[1:] * unit_pos, quat_i_[1:]))

        # same-rate : pitch/yaw is the same as previous section, roll ANGLE is zero
        elif att == "same-rate":
            uf_prev = u_[a-1]
            con.append(u_i_[:,1] - uf_prev[1])
            con.append(u_i_[:,2] - uf_prev[2])
            con.append(roll_direction_array(pos_i_[1:] * unit_pos, quat_i_[1:]))

        # zero-lift-turn or free : roll hold
        elif att == "zero-lift-turn" or att == "free":
            con.append(u_i_[:,0])

        else:
            print("ERROR: UNKNOWN ATTITUDE OPTION! ({})".format(att))
            sys.exit()

    return np.concatenate(con, axis=None)

def equality_jac_6DoF_rate(xdict, pdict, unitdict, condition):
    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]

    pos_ = xdict["position"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)

    u_ = xdict["u"].reshape(-1,3)

    num_sections = pdict["num_sections"]

    f_center = equality_6DoF_rate(xdict, pdict, unitdict, condition)
    nRow = len(f_center)
    jac["position"] = sparse.lil_matrix((nRow, pdict["M"]*3))
    jac["quaternion"] = sparse.lil_matrix((nRow, pdict["M"]*4))
    jac["u"] = sparse.lil_matrix((nRow, pdict["N"]*3))

    iRow = 0

    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        pos_i_ = pos_[a+i:b+i+1]
        quat_i_ = quat_[a+i:b+i+1]
        u_i_ = u_[a:b]

        #rate constraint

        att = pdict["params"][i]["attitude"]
        # attitude hold : angular velocity is zero
        if att in ["hold", "vertical"]:
            jac["u"][iRow:iRow+n*3, a*3:(a+n)*3] = np.eye(n*3)
            iRow += n*3


        # kick-turn : pitch rate constant, roll/yaw rate is zero
        elif att == "kick-turn" or att == "pitch":
            jac["u"][iRow:iRow+n, a*3:(a+n)*3:3] = np.eye(n)
            iRow += n
            jac["u"][iRow:iRow+n, a*3+2:(a+n)*3+2:3] = np.eye(n)
            iRow += n
            jac["u"][iRow:iRow+n-1, a*3+1] = -1.0
            jac["u"][iRow:iRow+n-1, (a+1)*3+1:(a+n)*3+1:3] = np.eye(n-1)
            iRow += n-1

        # pitch-yaw : pitch/yaw constant, roll ANGLE is zero
        elif att == "pitch-yaw":
            jac["u"][iRow:iRow+n-1, a*3+1] = -1.0
            jac["u"][iRow:iRow+n-1, (a+1)*3+1:(a+n)*3+1:3] = np.eye(n-1)
            iRow += n-1
            jac["u"][iRow:iRow+n-1, a*3+2] = -1.0
            jac["u"][iRow:iRow+n-1, (a+1)*3+2:(a+n)*3+2:3] = np.eye(n-1)
            iRow += n-1
            for k in range(n):
                f_c = yb_r_dot(pos_i_[k+1] * unit_pos, quat_i_[k+1])
                for j in range(3):
                    pos_i_p_ = deepcopy(pos_i_[k+1])
                    pos_i_p_[j] += dx
                    f_p = yb_r_dot(pos_i_p_ * unit_pos, quat_i_[k+1])
                    jac["position"][iRow+k, (a+i+1+k)*3+j] = (f_p - f_c) / dx
                for j in range(4):
                    quat_i_p_ = deepcopy(quat_i_[k+1])
                    quat_i_p_[j] += dx
                    f_p = yb_r_dot(pos_i_[k+1] * unit_pos, quat_i_p_)
                    jac["quaternion"][iRow+k, (a+i+1+k)*4+j] = (f_p - f_c) / dx
            iRow += n

        # same-rate : pitch/yaw is the same as previous section, roll ANGLE is zero
        elif att == "same-rate":
            jac["u"][iRow:iRow+n, a*3-2] = -1.0
            jac["u"][iRow:iRow+n, a*3+1:(a+n)*3+1:3] = np.eye(n)
            iRow += n
            jac["u"][iRow:iRow+n, a*3-1] = -1.0
            jac["u"][iRow:iRow+n, a*3+2:(a+n)*3+2:3] = np.eye(n)
            iRow += n
            for k in range(n):
                f_c = yb_r_dot(pos_i_[k+1] * unit_pos, quat_i_[k+1])
                for j in range(3):
                    pos_i_p_ = deepcopy(pos_i_[k+1])
                    pos_i_p_[j] += dx
                    f_p = yb_r_dot(pos_i_p_ * unit_pos, quat_i_[k+1])
                    jac["position"][iRow+k, (a+i+1+k)*3+j] = (f_p - f_c) / dx
                for j in range(4):
                    quat_i_p_ = deepcopy(quat_i_[k+1])
                    quat_i_p_[j] += dx
                    f_p = yb_r_dot(pos_i_[k+1] * unit_pos, quat_i_p_)
                    jac["quaternion"][iRow+k, (a+i+1+k)*4+j] = (f_p - f_c) / dx
            iRow += n

        # zero-lift-turn or free : roll hold
        elif att == "zero-lift-turn" or att == "free":
            jac["u"][iRow:iRow+n, a*3:(a+n)*3:3] = np.eye(n)
            iRow += n

        else:
            print("ERROR: UNKNOWN ATTITUDE OPTION! ({})".format(att))
            sys.exit()

    return jac



def inequality_time(xdict, pdict, unitdict, condition):
    con = []
    t_normal = xdict["t"]

    for i in range(pdict["num_sections"]-1):
        if not (pdict["params"][i]["timeFixed"] and pdict["params"][i+1]["timeFixed"]):
            con.append(t_normal[i+1] - t_normal[i])

    return np.array(con)

def inequality_jac_time(xdict, pdict, unitdict, condition):
    jac = {}

    data = []
    row = []
    col = []

    counter = 0
    for i in range(pdict["num_sections"]-1):
        if not (pdict["params"][i]["timeFixed"] and pdict["params"][i+1]["timeFixed"]):
            data.extend([-1.0, 1.0])
            row.extend([counter, counter])
            col.extend([i, i+1])
            counter += 1

    jac["t"] = sparse.coo_matrix((data, (row, col)), shape=[counter, len(xdict["t"])])

    return jac


def inequality_kickturn(xdict, pdict, unitdict, condition):
    con = []
    unit_u = unitdict["u"]
    u_ = xdict["u"].reshape(-1,3) * unit_u
    num_sections = pdict["num_sections"]

    for i in range(num_sections-1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        u_i_ = u_[a:b]

        # kick turn
        if "kick" in pdict["params"][i]["attitude"]:
            con.append(-u_i_[:,1])
            #con.append(u_i_[:,1]+0.36)

    return np.concatenate(con, axis=None)    

def inequality_jac_kickturn(xdict, pdict, unitdict, condition):
    jac = {}
    unit_u = unitdict["u"]
    u_ = xdict["u"].reshape(-1,3) * unit_u
    num_sections = pdict["num_sections"]

    data = []
    row = []
    col = []

    nRow = 0
    for i in range(num_sections-1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        u_i_ = u_[a:b]

        # kick turn
        if "kick" in pdict["params"][i]["attitude"]:
            row.extend(range(nRow, nRow+n))
            col.extend(range(a*3+1, b*3+1, 3))
            data.extend([-1.0] * n)
            nRow += n

    jac["u"] = {
        "coo": [np.array(row,dtype="i4"), np.array(col,dtype="i4"), np.array(data,dtype="f8")],
        "shape": (nRow, len(xdict["u"]))
    }

    return jac

def inequality_max_alpha(xdict, pdict, unitdict, condition):

    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)
    
    t = xdict["t"]

    num_sections = pdict["num_sections"]
    
    wind = pdict["wind_table"]
    
    for i in range(num_sections-1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        quat_i_ = quat_[a+i:b+i+1]
        to = t[i]
        tf = t[i+1]
        t_i_ = np.zeros(n+1)
        t_i_[0] = to
        t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0

        section_name = pdict["params"][i]["name"]

        # angle of attack
        if section_name in condition["aoa_max_deg"]:
            aoa_max = condition["aoa_max_deg"][section_name]["value"] * np.pi / 180.0
            units[3] = aoa_max
            if condition["aoa_max_deg"][section_name]["range"] == "all":
                con.append(1.0 - aoa_zerolift_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units))
            elif condition["aoa_max_deg"][section_name]["range"] == "initial":
                con.append(1.0 - angle_of_attack_all_dimless(pos_i_[0], vel_i_[0], quat_i_[0], to, wind, units))

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)    

def inequality_max_q(xdict, pdict, unitdict, condition):

    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)
    
    t = xdict["t"]

    num_sections = pdict["num_sections"]
    
    wind = pdict["wind_table"]
    
    for i in range(num_sections-1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        quat_i_ = quat_[a+i:b+i+1]
        to = t[i]
        tf = t[i+1]
        t_i_ = np.zeros(n+1)
        t_i_[0] = to
        t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0

        section_name = pdict["params"][i]["name"]

        # max-Q
        if section_name in condition["q_max_pa"]:
            q_max = condition["q_max_pa"][section_name]["value"]
            units[3] = q_max
            if condition["q_max_pa"][section_name]["range"] == "all":
                con.append(1.0 - dynamic_pressure_array_dimless(pos_i_, vel_i_, t_i_, wind, units))
            elif condition["q_max_pa"][section_name]["range"] == "initial":
                con.append(1.0 - dynamic_pressure_dimless(pos_i_[0], vel_i_[0], to, wind, units))

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)    


def inequality_max_qalpha(xdict, pdict, unitdict, condition):

    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)
    
    t = xdict["t"]

    num_sections = pdict["num_sections"]
    
    wind = pdict["wind_table"]
    
    for i in range(num_sections-1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        quat_i_ = quat_[a+i:b+i+1]
        to = t[i]
        tf = t[i+1]
        t_i_ = np.zeros(n+1)
        t_i_[0] = to
        t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0

        section_name = pdict["params"][i]["name"]

        # max-Qalpha
        if section_name in condition["q-alpha_max_pa-deg"]:
            qalpha_max = condition["q-alpha_max_pa-deg"][section_name]["value"] * np.pi / 180.0
            units[3] = qalpha_max
            if condition["q-alpha_max_pa-deg"][section_name]["range"] == "all":
                con.append(1.0 - q_alpha_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units))
            elif condition["q-alpha_max_pa-deg"][section_name]["range"] == "initial":
                con.append(1.0 - q_alpha_dimless(pos_i_[0], vel_i_[0], quat_i_[0], to, wind, units))

    if len(con) == 0:
        return None
    else:
        return np.concatenate(con, axis=None)    


def inequality_jac_max_alpha(xdict, pdict, unitdict, condition):

    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    f_center = inequality_max_alpha(xdict, pdict, unitdict, condition)
    if hasattr(f_center, "__len__"):
        nRow = len(f_center)
    elif f_center is None:
        return None
    else:
        nRow = 1

    jac["position"] = {"coo": [[], [], []], "shape":(nRow, pdict["M"]*3)}
    jac["velocity"] = {"coo": [[], [], []], "shape":(nRow, pdict["M"]*3)}
    jac["quaternion"] = {"coo": [[], [], []], "shape":(nRow, pdict["M"]*4)}
    jac["t"] = {"coo": [[], [], []], "shape":(nRow, num_sections+1)}

    iRow = 0
    
    for i in range(num_sections-1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        quat_i_ = quat_[a+i:b+i+1]
        to = t[i]
        tf = t[i+1]
        t_i_ = np.zeros(n+1)
        t_i_[0] = to
        t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
        t_i_p_ = np.zeros(n+1)

        section_name = pdict["params"][i]["name"]
        wind = pdict["wind_table"]

        # angle of attack
        if section_name in condition["aoa_max_deg"]:
            aoa_max = condition["aoa_max_deg"][section_name]["value"] * np.pi / 180.0
            units[3] = aoa_max

            if condition["aoa_max_deg"][section_name]["range"] == "all":
                nk = range(n+1)
            elif condition["aoa_max_deg"][section_name]["range"] == "initial":
                nk = [0]

            f_c = aoa_zerolift_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units)

            for k in nk:
                for j in range(3):
                    pos_i_p_ = copy(pos_i_[k])
                    pos_i_p_[j] += dx
                    f_p = angle_of_attack_all_dimless(pos_i_p_, vel_i_[k], quat_i_[k], t_i_[k], wind, units)
                    jac["position"]["coo"][0].append(iRow+k)
                    jac["position"]["coo"][1].append((a+i+k)*3+j)
                    jac["position"]["coo"][2].append(-(f_p - f_c[k]) / dx)
                    
                for j in range(3):
                    vel_i_p_ = copy(vel_i_[k])
                    vel_i_p_[j] += dx
                    f_p = angle_of_attack_all_dimless(pos_i_[k], vel_i_p_, quat_i_[k], t_i_[k], wind, units)
                    jac["velocity"]["coo"][0].append(iRow+k)
                    jac["velocity"]["coo"][1].append((a+i+k)*3+j)
                    jac["velocity"]["coo"][2].append(-(f_p - f_c[k]) / dx)

                for j in range(4):
                    quat_i_p_ = copy(quat_i_[k])
                    quat_i_p_[j] += dx
                    f_p = angle_of_attack_all_dimless(pos_i_[k], vel_i_[k], quat_i_p_, t_i_[k], wind, units)
                    jac["quaternion"]["coo"][0].append(iRow+k)
                    jac["quaternion"]["coo"][1].append((a+i+k)*4+j)
                    jac["quaternion"]["coo"][2].append(-(f_p - f_c[k]) / dx)

                to_p = to + dx
                t_i_p_[0] = to_p
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf-to_p) / 2.0 + (tf+to_p) / 2.0
                f_p = aoa_zerolift_array_dimless(pos_i_, vel_i_, quat_i_, t_i_p_, wind, units)
                jac["t"]["coo"][0].extend(list(range(iRow, iRow+n+1)))
                jac["t"]["coo"][1].extend([i] * (n+1))
                jac["t"]["coo"][2].extend((-(f_p - f_c) / dx).tolist())

                tf_p = tf + dx
                t_i_p_[0] = to
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf_p-to) / 2.0 + (tf_p+to) / 2.0
                f_p = aoa_zerolift_array_dimless(pos_i_, vel_i_, quat_i_, t_i_p_, wind, units)
                jac["t"]["coo"][0].extend(list(range(iRow, iRow+n+1)))
                jac["t"]["coo"][1].extend([i+1] * (n+1))
                jac["t"]["coo"][2].extend((-(f_p - f_c) / dx).tolist())

            iRow += len(nk)

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac

def inequality_jac_max_q(xdict, pdict, unitdict, condition):
    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    f_center = inequality_max_q(xdict, pdict, unitdict, condition)
    if hasattr(f_center, "__len__"):
        nRow = len(f_center)
    elif f_center is None:
        return None
    else:
        nRow = 1

    jac["position"] = {"coo": [[], [], []], "shape":(nRow, pdict["M"]*3)}
    jac["velocity"] = {"coo": [[], [], []], "shape":(nRow, pdict["M"]*3)}
    jac["quaternion"] = {"coo": [[], [], []], "shape":(nRow, pdict["M"]*4)}
    jac["t"] = {"coo": [[], [], []], "shape":(nRow, num_sections+1)}

    iRow = 0
    
    for i in range(num_sections-1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        to = t[i]
        tf = t[i+1]
        t_i_ = np.zeros(n+1)
        t_i_[0] = to
        t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
        t_i_p_ = np.zeros(n+1)

        section_name = pdict["params"][i]["name"]
        wind = pdict["wind_table"]

        # angle of attack
        if section_name in condition["q_max_pa"]:
            q_max = condition["q_max_pa"][section_name]["value"]
            units[3] = q_max

            if condition["q_max_pa"][section_name]["range"] == "all":
                nk = range(n+1)
            elif condition["q_max_pa"][section_name]["range"] == "initial":
                nk = [0]

            f_c = dynamic_pressure_array_dimless(pos_i_, vel_i_, t_i_, wind, units)

            for k in nk:
                for j in range(3):
                    pos_i_p_ = copy(pos_i_[k])
                    pos_i_p_[j] += dx
                    f_p = dynamic_pressure_dimless(pos_i_p_, vel_i_[k], t_i_[k], wind, units)
                    jac["position"]["coo"][0].append(iRow+k)
                    jac["position"]["coo"][1].append((a+i+k)*3+j)
                    jac["position"]["coo"][2].append(-(f_p - f_c[k]) / dx)
                    
                for j in range(3):
                    vel_i_p_ = copy(vel_i_[k])
                    vel_i_p_[j] += dx
                    f_p = dynamic_pressure_dimless(pos_i_[k], vel_i_p_, t_i_[k], wind, units)
                    jac["velocity"]["coo"][0].append(iRow+k)
                    jac["velocity"]["coo"][1].append((a+i+k)*3+j)
                    jac["velocity"]["coo"][2].append(-(f_p - f_c[k]) / dx)

                to_p = to + dx
                t_i_p_[0] = to_p
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf-to_p) / 2.0 + (tf+to_p) / 2.0
                f_p = dynamic_pressure_array_dimless(pos_i_, vel_i_, t_i_p_, wind, units)
                jac["t"]["coo"][0].extend(list(range(iRow, iRow+n+1)))
                jac["t"]["coo"][1].extend([i] * (n+1))
                jac["t"]["coo"][2].extend((-(f_p - f_c) / dx).tolist())

                tf_p = tf + dx
                t_i_p_[0] = to
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf_p-to) / 2.0 + (tf_p+to) / 2.0
                f_p = dynamic_pressure_array_dimless(pos_i_, vel_i_, t_i_p_, wind, units)
                jac["t"]["coo"][0].extend(list(range(iRow, iRow+n+1)))
                jac["t"]["coo"][1].extend([i+1] * (n+1))
                jac["t"]["coo"][2].extend((-(f_p - f_c) / dx).tolist())

            iRow += len(nk)

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


def inequality_jac_max_qalpha(xdict, pdict, unitdict, condition):

    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t, 1.0])

    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    f_center = inequality_max_qalpha(xdict, pdict, unitdict, condition)
    if hasattr(f_center, "__len__"):
        nRow = len(f_center)
    elif f_center is None:
        return None
    else:
        nRow = 1

    jac["position"] = {"coo": [[], [], []], "shape":(nRow, pdict["M"]*3)}
    jac["velocity"] = {"coo": [[], [], []], "shape":(nRow, pdict["M"]*3)}
    jac["quaternion"] = {"coo": [[], [], []], "shape":(nRow, pdict["M"]*4)}
    jac["t"] = {"coo": [[], [], []], "shape":(nRow, num_sections+1)}

    iRow = 0
    
    for i in range(num_sections-1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n

        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        quat_i_ = quat_[a+i:b+i+1]
        to = t[i]
        tf = t[i+1]
        t_i_ = np.zeros(n+1)
        t_i_[0] = to
        t_i_[1:] = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
        t_i_p_ = np.zeros(n+1)

        section_name = pdict["params"][i]["name"]
        wind = pdict["wind_table"]

        # angle of attack
        if section_name in condition["q-alpha_max_pa-deg"]:
            qalpha_max = condition["q-alpha_max_pa-deg"][section_name]["value"] * np.pi / 180.0
            units[3] = qalpha_max

            if condition["q-alpha_max_pa-deg"][section_name]["range"] == "all":
                nk = range(n+1)
            elif condition["q-alpha_max_pa-deg"][section_name]["range"] == "initial":
                nk = [0]

            f_c = q_alpha_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units)

            for k in nk:
                for j in range(3):
                    pos_i_p_ = copy(pos_i_[k])
                    pos_i_p_[j] += dx
                    f_p = q_alpha_dimless(pos_i_p_, vel_i_[k], quat_i_[k], t_i_[k], wind, units)
                    jac["position"]["coo"][0].append(iRow+k)
                    jac["position"]["coo"][1].append((a+i+k)*3+j)
                    jac["position"]["coo"][2].append(-(f_p - f_c[k]) / dx)
                    
                for j in range(3):
                    vel_i_p_ = copy(vel_i_[k])
                    vel_i_p_[j] += dx
                    f_p = q_alpha_dimless(pos_i_[k], vel_i_p_, quat_i_[k], t_i_[k], wind, units)
                    jac["velocity"]["coo"][0].append(iRow+k)
                    jac["velocity"]["coo"][1].append((a+i+k)*3+j)
                    jac["velocity"]["coo"][2].append(-(f_p - f_c[k]) / dx)

                for j in range(4):
                    quat_i_p_ = copy(quat_i_[k])
                    quat_i_p_[j] += dx
                    f_p = q_alpha_dimless(pos_i_[k], vel_i_[k], quat_i_p_, t_i_[k], wind, units)
                    jac["quaternion"]["coo"][0].append(iRow+k)
                    jac["quaternion"]["coo"][1].append((a+i+k)*4+j)
                    jac["quaternion"]["coo"][2].append(-(f_p - f_c[k]) / dx)

                to_p = to + dx
                t_i_p_[0] = to_p
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf-to_p) / 2.0 + (tf+to_p) / 2.0
                f_p = q_alpha_array_dimless(pos_i_, vel_i_, quat_i_, t_i_p_, wind, units)
                jac["t"]["coo"][0].extend(list(range(iRow, iRow+n+1)))
                jac["t"]["coo"][1].extend([i] * (n+1))
                jac["t"]["coo"][2].extend((-(f_p - f_c) / dx).tolist())

                tf_p = tf + dx
                t_i_p_[0] = to
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf_p-to) / 2.0 + (tf_p+to) / 2.0
                f_p = q_alpha_array_dimless(pos_i_, vel_i_, quat_i_, t_i_p_, wind, units)
                jac["t"]["coo"][0].extend(list(range(iRow, iRow+n+1)))
                jac["t"]["coo"][1].extend([i+1] * (n+1))
                jac["t"]["coo"][2].extend((-(f_p - f_c) / dx).tolist())

            iRow += len(nk)

    for key in jac.keys():
        jac[key]["coo"][0] = np.array(jac[key]["coo"][0], dtype="i4")
        jac[key]["coo"][1] = np.array(jac[key]["coo"][1], dtype="i4")
        jac[key]["coo"][2] = np.array(jac[key]["coo"][2], dtype="f8")

    return jac


@jit(nopython=True)
def dynamic_pressure_array_dimless(pos, vel, t, wind, units):
    return np.array([dynamic_pressure_dimless(pos[i], vel[i], t[i], wind, units) for i in range(len(t))])

@jit(nopython=True)
def aoa_zerolift_array_dimless(pos, vel, quat, t, wind, units):
    return np.array([angle_of_attack_all_dimless(pos[i], vel[i], quat[i], t[i], wind, units) for i in range(len(t))])

@jit(nopython=True)
def q_alpha_array_dimless(pos, vel, quat, t, wind, units):
    return np.array([q_alpha_dimless(pos[i], vel[i], quat[i], t[i], wind, units) for i in range(len(t))])

@jit(nopython=True)
def roll_direction_array(pos, quat):
    return np.array([yb_r_dot(pos[i], quat[i]) for i in range(len(pos))])

@jit(nopython=True)
def yb_r_dot(pos_eci, quat_eci2body):
    yb_dir_eci = quatrot(conj(quat_eci2body), np.array([0.0, 1.0, 0.0]))
    return yb_dir_eci.dot(normalize(pos_eci))

@jit(nopython=True)
def q_alpha_dimless(pos_eci, vel_eci, quat, t, wind, units):
    return angle_of_attack_all_dimless(pos_eci, vel_eci, quat, t, wind, units) * dynamic_pressure_dimless(pos_eci, vel_eci, t, wind, units)

@jit(nopython=True)
def angle_of_attack_all_dimless(pos_eci_e, vel_eci_e, quat, t_e, wind, units):

    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return angle_of_attack_all_rad(pos_eci, vel_eci, quat, t, wind) / units[3]


@jit(nopython=True)
def angle_of_attack_ab_dimless(pos_eci_e, vel_eci_e, quat, t_e, wind, units):

    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return angle_of_attack_ab_rad(pos_eci, vel_eci, quat, t, wind) / units[3]


@jit(nopython=True)
def dynamic_pressure_dimless(pos_eci_e, vel_eci_e, t_e, wind, units):

    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return dynamic_pressure_pa(pos_eci, vel_eci, t, wind) / units[3]

    
def equality_jac_user(xdict, pdict, unitdict, condition):
    if equality_user(xdict, pdict, unitdict, condition) is not None:
        return jac_fd(equality_user, xdict, pdict, unitdict, condition)

def inequality_jac_user(xdict, pdict, unitdict, condition):
    if inequality_user(xdict, pdict, unitdict, condition) is not None:
        return jac_fd(inequality_user, xdict, pdict, unitdict, condition)


def cost_6DoF(xdict, condition):
    
    if condition["OptimizationMode"] == "Payload":
        return -xdict["mass"][0] #初期質量(無次元)を最大化
    else:
        return xdict["t"][-1] #到達時間を最小化(=余剰推進剤を最大化)

def cost_jac(xdict, condition):

    jac = {}
    if condition["OptimizationMode"] == "Payload":
        jac["mass"] = np.zeros(xdict["mass"].size)
        jac["mass"][0] = -1.0
    else:
        jac["t"] = np.zeros(xdict["t"].size)
        jac["t"][-1] = 1.0
    return jac

