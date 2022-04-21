import sys
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import interp1d
from numba import jit
from utils import *
from USStandardAtmosphere import *
from coordinate import *
from tools.plot_output import display_6DoF
from tools.IIP import posLLH_IIP_FAA
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



@jit(nopython=True)
def dynamics(x, u, t, param, zlt, wind, ca):

    mass = x[0]
    pos_eci = x[1:4]
    vel_eci = x[4:7]
    quat_eci2body = x[7:11]
    d_roll = u[0]
    d_pitch = u[1]
    d_yaw = u[2]

    pos_llh = ecef2geodetic(pos_eci[0],pos_eci[1],pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])
    rho = airdensity_at(altitude_m)
    p = airpressure_at(altitude_m)


    #対気速度

    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)

    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci
    mach_number = norm(vel_air_eci) / speed_of_sound(altitude_m)

    thrust_vac_n = param[0]
    massflow_kgps = param[1]
    airArea_m2 = param[2]
    airAxialForce_coeff = np.interp(mach_number, ca[:,0], ca[:,1])
    nozzleArea_m2 = param[4]

    ret = np.zeros(11)

    aero_n_eci = 0.5 * rho * norm(vel_air_eci) * -vel_air_eci * airArea_m2 * airAxialForce_coeff

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


    ret[0] = -massflow_kgps   
    ret[1:4] = vel_eci
    ret[4:7] = acc_eci
    ret[7:11] = d_quat

    return ret


def rocket_simulation(x_init, u_table, pdict, t_init, t_out, dt=0.1):

    # t_init, x_initからdynamicsをdt間隔で積分し、t_outの時刻の値(の配列)を返す
    # u_table : list of [t, u1, u2, u3]

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

    while(t < t_final):

        tn = t + dt

        if event_index < len(pdict["params"]) - 1:
            if tn > pdict["params"][event_index+1]["timeAt_sec"]:
                event_index += 1
                param[0] = pdict["params"][event_index]["thrust_n"]
                param[1] = pdict["params"][event_index]["massflow_kgps"]
                param[2] = pdict["params"][event_index]["airArea_m2"]
                param[4] = pdict["params"][event_index]["nozzleArea_m2"]
                x[0] -= pdict["params"][event_index]["mass_jettison_kg"]

        u = np.array([np.interp(t, u_table[:,0], u_table[:,i+1]) for i in range(3)])
        x = runge_kutta_4d(lambda xa,ta:dynamics(xa, u, ta, param, zlt, wind, ca), x, t, dt)
        t = t + dt

        if pdict["params"][event_index]["attitude"] == "zero-lift-turn":
            x[7:11] = zerolift_turn_correct(x, t, wind)
        x[7:11] = normalize(x[7:11])

        t_map.append(t)
        x_map.append(x)
        u_map.append(u)

    x_map = np.array(x_map)
    u_map = np.array(u_map)
    x_out = np.vstack([np.interp(t_out, t_map, x_map[:,i]) for i in range(len(x))]).T
    u_out = np.vstack([np.interp(t_out, t_map, u_map[:,i]) for i in range(len(u))]).T


    return x_out, u_out

@jit('f8[:](f8[:],f8,f8[:,:])',nopython=True)
def zerolift_turn_correct(x,t,wind=np.zeros((2,3))):

    #mass = x[0]
    pos_eci = x[1:4]
    vel_eci = x[4:7]
    #quat_eci2body = x[7:11]

    pos_llh = ecef2geodetic(pos_eci[0],pos_eci[1],pos_eci[2])
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

def euler(function, x, t, dt):
    return x + function(x, t) * dt

def runge_kutta_4d(function, x, t, dt):
    k1 = function(x, t)
    k2 = function(x + dt / 2.0 * k1, t + dt / 2.0)
    k3 = function(x + dt / 2.0 * k2, t + dt / 2.0)
    k4 = function(x + dt * k3, t + dt)
    return x + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 * dt


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

def jac_fd(con, xdict, pdict, unitdict, condition):

    jac = {}
    dx = 1.0e-8
    g_base = con(xdict, pdict, unitdict, condition)
    if hasattr(g_base,"__len__"):
        nRows = len(g_base)
    else:
        nRows = 1
    for key,val in xdict.items():
        jac[key] = np.zeros((nRows, val.size))
        for i in range(val.size):
            xdict_p = deepcopy(xdict)
            xdict_p[key][i] += dx
            g_p = con(xdict_p, pdict, unitdict, condition)
            jac[key][:,i] = (g_p - g_base) / dx

    return jac
    
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

def jac_fd(con, xdict, pdict, unitdict, condition):
    
    jac = {}
    dx = 1.0e-8
    g_base = con(xdict, pdict, unitdict, condition)
    if hasattr(g_base,"__len__"):
        nRows = len(g_base)
    else:
        nRows = 1
    for key,val in xdict.items():
        jac[key] = np.zeros((nRows, val.size))
        for i in range(val.size):
            xdict_p = deepcopy(xdict)
            xdict_p[key][i] += dx
            g_p = con(xdict_p, pdict, unitdict, condition)
            jac[key][:,i] = (g_p - g_base) / dx

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
    jac["velocity"] = np.zeros((pdict["N"]*3, pdict["M"]*3))
    jac["t"] = np.zeros((pdict["N"]*3, num_sections+1))

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

        jac["velocity"][a*3:b*3, (a+i+1)*3:(b+i+1)*3] = np.eye(n*3) * (-unit_vel * (tf-to) * unit_t / 2.0 / unit_pos) # rh vel
        jac["t"][a*3:b*3, i]   =  vel_i_[1:].ravel() * unit_vel * unit_t / 2.0 / unit_pos # rh to
        jac["t"][a*3:b*3, i+1] = -vel_i_[1:].ravel() * unit_vel * unit_t / 2.0 / unit_pos # rh tf

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

    jac["mass"] = np.zeros((pdict["N"]*3, pdict["M"]))
    jac["position"] = np.zeros((pdict["N"]*3, pdict["M"]*3))
    jac["velocity"] = np.zeros((pdict["N"]*3, pdict["M"]*3))
    jac["quaternion"] = np.zeros((pdict["N"]*3, pdict["M"]*4))
    jac["t"] = np.zeros((pdict["N"]*3, num_sections+1))

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
            jac["mass"][(a+j)*3,   a+i+j+1] = -(f_p[j,0] - f_center[j,0]) / dx * (tf-to) * unit_t / 2.0 # rh acc_x mass
            jac["mass"][(a+j)*3+1, a+i+j+1] = -(f_p[j,1] - f_center[j,1]) / dx * (tf-to) * unit_t / 2.0 # rh acc_y mass
            jac["mass"][(a+j)*3+2, a+i+j+1] = -(f_p[j,2] - f_center[j,2]) / dx * (tf-to) * unit_t / 2.0 # rh acc_z mass

            for k in range(3):
                pos_i_p_ = deepcopy(pos_i_)
                pos_i_p_[j+1, k] += dx
                f_p = dynamics_velocity(mass_i_[1:], pos_i_p_[1:], vel_i_[1:], quat_i_[1:], t_nodes, param, wind, ca, units)
                jac["position"][(a+j)*3,   (a+i+j+1)*3+k] = -(f_p[j,0] - f_center[j,0]) / dx * (tf-to) * unit_t / 2.0 # rh acc_x pos
                jac["position"][(a+j)*3+1, (a+i+j+1)*3+k] = -(f_p[j,1] - f_center[j,1]) / dx * (tf-to) * unit_t / 2.0 # rh acc_y pos
                jac["position"][(a+j)*3+2, (a+i+j+1)*3+k] = -(f_p[j,2] - f_center[j,2]) / dx * (tf-to) * unit_t / 2.0 # rh acc_z pos

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
                jac["quaternion"][(a+j)*3,   (a+i+j+1)*4+k] = -(f_p[j,0] - f_center[j,0]) / dx * (tf-to) * unit_t / 2.0 # rh acc_x quat
                jac["quaternion"][(a+j)*3+1, (a+i+j+1)*4+k] = -(f_p[j,1] - f_center[j,1]) / dx * (tf-to) * unit_t / 2.0 # rh acc_y quat
                jac["quaternion"][(a+j)*3+2, (a+i+j+1)*4+k] = -(f_p[j,2] - f_center[j,2]) / dx * (tf-to) * unit_t / 2.0 # rh acc_z quat


        jac["t"][a*3:b*3, i]   =  f_center.ravel() * unit_t / 2.0  # rh to
        jac["t"][a*3:b*3, i+1] = -f_center.ravel() * unit_t / 2.0  # rh tf

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
    jac["t"] = np.zeros((pdict["N"]*4, num_sections+1))


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

            jac["t"][a*4:b*4, i]   =  f_center.ravel() * unit_t / 2.0  # rh to
            jac["t"][a*4:b*4, i+1] = -f_center.ravel() * unit_t / 2.0  # rh tf
                        
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

    jac["u"] = sparse.coo_matrix((data, (row, col)), shape=[nRow, len(xdict["u"])])

    return jac



def inequality_6DoF(xdict, pdict, unitdict, condition):

    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t])

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
            if condition["aoa_max_deg"][section_name]["range"] == "all":
                con.append(1.0 - aoa_zerolift_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units) / aoa_max)
            elif condition["aoa_max_deg"][section_name]["range"] == "initial":
                con.append(1.0 - angle_of_attack_all_rad_dimless(pos_i_[0], vel_i_[0], quat_i_[0], to, wind, units) / aoa_max)

        # max-Q
        if section_name in condition["q_max_pa"]:
            q_max = condition["q_max_pa"][section_name]["value"]
            if condition["q_max_pa"][section_name]["range"] == "all":
                con.append(1.0 - dynamic_pressure_array_dimless(pos_i_, vel_i_, t_i_, wind, units) / q_max)
            elif condition["q_max_pa"][section_name]["range"] == "initial":
                con.append(1.0 - dynamic_pressure_pa_dimless(pos_i_[0], vel_i_[0], to, wind, units) / q_max)

        # max-Qalpha
        if section_name in condition["q-alpha_max_pa-deg"]:
            qalpha_max = condition["q-alpha_max_pa-deg"][section_name]["value"] * np.pi / 180.0
            if condition["q-alpha_max_pa-deg"][section_name]["range"] == "all":
                con.append(1.0 - q_alpha_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units) / qalpha_max)
            elif condition["q-alpha_max_pa-deg"][section_name]["range"] == "initial":
                con.append(1.0 - q_alpha_pa_rad_dimless(pos_i_[0], vel_i_[0], quat_i_[0], to, wind, units) / qalpha_max)

    return np.concatenate(con, axis=None)    

def inequality_jac_6DoF(xdict, pdict, unitdict, condition):

    jac = {}
    dx = 1.0e-8

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_t = unitdict["t"]
    units = np.array([unit_pos, unit_vel, unit_t])

    pos_ = xdict["position"].reshape(-1,3)
    vel_ = xdict["velocity"].reshape(-1,3)
    quat_ = xdict["quaternion"].reshape(-1,4)

    t = xdict["t"]

    num_sections = pdict["num_sections"]

    f_center = inequality_6DoF(xdict, pdict, unitdict, condition)
    nRow = len(f_center)

    jac["position"] = np.zeros((nRow, pdict["M"]*3))
    jac["velocity"] = np.zeros((nRow, pdict["M"]*3))
    jac["quaternion"] = np.zeros((nRow, pdict["M"]*4))
    jac["t"] = np.zeros((nRow, num_sections+1))

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
            if condition["aoa_max_deg"][section_name]["range"] == "all":
                f_c = aoa_zerolift_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units)
                for k in range(n+1):
                    for j in range(3):
                        pos_i_p_ = deepcopy(pos_i_[k])
                        pos_i_p_[j] += dx
                        f_p = angle_of_attack_all_rad_dimless(pos_i_p_, vel_i_[k], quat_i_[k], t_i_[k], wind, units)
                        jac["position"][iRow+k, (a+i+k)*3+j] = -(f_p - f_c[k]) / dx / aoa_max
                    for j in range(3):
                        vel_i_p_ = deepcopy(vel_i_[k])
                        vel_i_p_[j] += dx
                        f_p = angle_of_attack_all_rad_dimless(pos_i_[k], vel_i_p_, quat_i_[k], t_i_[k], wind, units)
                        jac["velocity"][iRow+k, (a+i+k)*3+j] = -(f_p - f_c[k]) / dx / aoa_max
                    for j in range(4):
                        quat_i_p_ = deepcopy(quat_i_[k])
                        quat_i_p_[j] += dx
                        f_p = angle_of_attack_all_rad_dimless(pos_i_[k], vel_i_[k], quat_i_p_, t_i_[k], wind, units)
                        jac["quaternion"][iRow+k, (a+i+k)*4+j] = -(f_p - f_c[k]) / dx / aoa_max
                to_p = to + dx
                t_i_p_[0] = to_p
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf-to_p) / 2.0 + (tf+to_p) / 2.0
                f_p = aoa_zerolift_array_dimless(pos_i_, vel_i_, quat_i_, t_i_p_, wind, units)
                jac["t"][iRow:iRow+n+1, i] = -(f_p - f_c) / dx / aoa_max

                tf_p = tf + dx
                t_i_p_[0] = to
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf_p-to) / 2.0 + (tf_p+to) / 2.0
                f_p = aoa_zerolift_array_dimless(pos_i_, vel_i_, quat_i_, t_i_p_, wind, units)
                jac["t"][iRow:iRow+n+1, i+1] = -(f_p - f_c) / dx / aoa_max

                iRow += n+1

            elif condition["aoa_max_deg"][section_name]["range"] == "initial":

                f_c = angle_of_attack_all_rad_dimless(pos_i_[0], vel_i_[0], quat_i_[0], to, wind, units)
                for j in range(3):
                    pos_i_p_ = deepcopy(pos_i_[0])
                    pos_i_p_[j] += dx
                    f_p = angle_of_attack_all_rad_dimless(pos_i_p_, vel_i_[0], quat_i_[0], to, wind, units)
                    jac["position"][iRow, (a+i)*3+j] = -(f_p - f_c) / dx / aoa_max
                for j in range(3):
                    vel_i_p_ = deepcopy(vel_i_[0])
                    vel_i_p_[j] += dx
                    f_p = angle_of_attack_all_rad_dimless(pos_i_[0], vel_i_p_, quat_i_[0], to, wind, units)
                    jac["velocity"][iRow, (a+i)*3+j] = -(f_p - f_c) / dx / aoa_max
                for j in range(4):
                    quat_i_p_ = deepcopy(quat_i_[0])
                    quat_i_p_[j] += dx
                    f_p = angle_of_attack_all_rad_dimless(pos_i_[0], vel_i_[0], quat_i_p_, to, wind, units)
                    jac["quaternion"][iRow, (a+i)*4+j] = -(f_p - f_c) / dx / aoa_max
                to_p = to + dx
                f_p = angle_of_attack_all_rad_dimless(pos_i_[0], vel_i_[0], quat_i_[0], to_p, wind, units)
                jac["t"][iRow, i] = -(f_p - f_c) / dx / aoa_max

                iRow += 1

        # max-Q
        if section_name in condition["q_max_pa"]:
            q_max = condition["q_max_pa"][section_name]["value"]
            if condition["q_max_pa"][section_name]["range"] == "all":
                f_c = dynamic_pressure_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units)
                for k in range(n+1):
                    for j in range(3):
                        pos_i_p_ = deepcopy(pos_i_[k])
                        pos_i_p_[j] += dx
                        f_p = dynamic_pressure_pa_dimless(pos_i_p_, vel_i_[k], quat_i_[k], t_i_[k], wind, units)
                        jac["position"][iRow+k, (a+i+k)*3+j] = -(f_p - f_c[k]) / dx / q_max
                    for j in range(3):
                        vel_i_p_ = deepcopy(vel_i_[k])
                        vel_i_p_[j] += dx
                        f_p = dynamic_pressure_pa_dimless(pos_i_[k], vel_i_p_, quat_i_[k], t_i_[k], wind, units)
                        jac["velocity"][iRow+k, (a+i+k)*3+j] = -(f_p - f_c[k]) / dx / q_max
                to_p = to + dx
                t_i_p_[0] = to_p
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf-to_p) / 2.0 + (tf+to_p) / 2.0
                f_p = dynamic_pressure_array_dimless(pos_i_, vel_i_, quat_i_, t_i_p_, wind, units)
                jac["t"][iRow:iRow+n+1, i] = -(f_p - f_c) / dx / q_max

                tf_p = tf + dx
                t_i_p_[0] = to
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf_p-to) / 2.0 + (tf_p+to) / 2.0
                f_p = dynamic_pressure_array_dimless(pos_i_, vel_i_, quat_i_, t_i_p_, wind, units)
                jac["t"][iRow:iRow+n+1, i+1] = -(f_p - f_c) / dx / q_max

                iRow += n+1

            elif condition["q_max_pa"][section_name]["range"] == "initial":

                f_c = dynamic_pressure_pa_dimless(pos_i_[0], vel_i_[0], to, wind, units)
                for j in range(3):
                    pos_i_p_ = deepcopy(pos_i_[0])
                    pos_i_p_[j] += dx
                    f_p = dynamic_pressure_pa_dimless(pos_i_p_, vel_i_[0], to, wind, units)
                    jac["position"][iRow, (a+i)*3+j] = -(f_p - f_c) / dx / q_max
                for j in range(3):
                    vel_i_p_ = deepcopy(vel_i_[0])
                    vel_i_p_[j] += dx
                    f_p = dynamic_pressure_pa_dimless(pos_i_[0], vel_i_p_, to, wind, units)
                    jac["velocity"][iRow, (a+i)*3+j] = -(f_p - f_c) / dx / q_max
                to_p = to + dx
                f_p = dynamic_pressure_pa_dimless(pos_i_[0], vel_i_[0], to_p, wind, units)
                jac["t"][iRow, i] = -(f_p - f_c) / dx / q_max

                iRow += 1


        # max-Qalpha
        if section_name in condition["q-alpha_max_pa-deg"]:
            qalpha_max = condition["q-alpha_max_pa-deg"][section_name]["value"] * np.pi / 180.0
            if condition["q-alpha_max_pa-deg"][section_name]["range"] == "all":
                f_c = q_alpha_array_dimless(pos_i_, vel_i_, quat_i_, t_i_, wind, units)
                for k in range(n+1):
                    for j in range(3):
                        pos_i_p_ = deepcopy(pos_i_[k])
                        pos_i_p_[j] += dx
                        f_p = q_alpha_pa_rad_dimless(pos_i_p_, vel_i_[k], quat_i_[k], t_i_[k], wind, units)
                        jac["position"][iRow+k, (a+i+k)*3+j] = -(f_p - f_c[k]) / dx / qalpha_max
                    for j in range(3):
                        vel_i_p_ = deepcopy(vel_i_[k])
                        vel_i_p_[j] += dx
                        f_p = q_alpha_pa_rad_dimless(pos_i_[k], vel_i_p_, quat_i_[k], t_i_[k], wind, units)
                        jac["velocity"][iRow+k, (a+i+k)*3+j] = -(f_p - f_c[k]) / dx / qalpha_max
                    for j in range(4):
                        quat_i_p_ = deepcopy(quat_i_[k])
                        quat_i_p_[j] += dx
                        f_p = q_alpha_pa_rad_dimless(pos_i_[k], vel_i_[k], quat_i_p_, t_i_[k], wind, units)
                        jac["quaternion"][iRow+k, (a+i+k)*4+j] = -(f_p - f_c[k]) / dx / qalpha_max
                to_p = to + dx
                t_i_p_[0] = to_p
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf-to_p) / 2.0 + (tf+to_p) / 2.0
                f_p = q_alpha_array_dimless(pos_i_, vel_i_, quat_i_, t_i_p_, wind, units)
                jac["t"][iRow:iRow+n+1, i] = -(f_p - f_c) / dx / qalpha_max

                tf_p = tf + dx
                t_i_p_[0] = to
                t_i_p_[1:] = pdict["ps_params"][i]["tau"] * (tf_p-to) / 2.0 + (tf_p+to) / 2.0
                f_p = q_alpha_array_dimless(pos_i_, vel_i_, quat_i_, t_i_p_, wind, units)
                jac["t"][iRow:iRow+n+1, i+1] = -(f_p - f_c) / dx / qalpha_max

                iRow += n+1

            elif condition["q-alpha_max_pa-deg"][section_name]["range"] == "initial":
                
                f_c = q_alpha_pa_rad_dimless(pos_i_[0], vel_i_[0], quat_i_[0], to, wind, units)
                for j in range(3):
                    pos_i_p_ = deepcopy(pos_i_[0])
                    pos_i_p_[j] += dx
                    f_p = q_alpha_pa_rad_dimless(pos_i_p_, vel_i_[0], quat_i_[0], to, wind, units)
                    jac["position"][iRow, (a+i)*3+j] = -(f_p - f_c) / dx / qalpha_max
                for j in range(3):
                    vel_i_p_ = deepcopy(vel_i_[0])
                    vel_i_p_[j] += dx
                    f_p = q_alpha_pa_rad_dimless(pos_i_[0], vel_i_p_, quat_i_[0], to, wind, units)
                    jac["velocity"][iRow, (a+i)*3+j] = -(f_p - f_c) / dx / qalpha_max
                for j in range(4):
                    quat_i_p_ = deepcopy(quat_i_[0])
                    quat_i_p_[j] += dx
                    f_p = q_alpha_pa_rad_dimless(pos_i_[0], vel_i_[0], quat_i_p_, to, wind, units)
                    jac["quaternion"][iRow, (a+i)*4+j] = -(f_p - f_c) / dx / qalpha_max
                to_p = to + dx
                f_p = q_alpha_pa_rad_dimless(pos_i_[0], vel_i_[0], quat_i_[0], to_p, wind, units)
                jac["t"][iRow, i] = -(f_p - f_c) / dx / qalpha_max

                iRow += 1

    return jac


@jit(nopython=True)
def dynamic_pressure_array_dimless(pos, vel, t, wind, units):
    return np.array([dynamic_pressure_pa_dimless(pos[i], vel[i], t[i], wind, units) for i in range(len(t))])

@jit(nopython=True)
def aoa_zerolift_array_dimless(pos, vel, quat, t, wind, units):
    return np.array([angle_of_attack_all_rad_dimless(pos[i], vel[i], quat[i], t[i], wind, units) for i in range(len(t))])

@jit(nopython=True)
def q_alpha_array_dimless(pos, vel, quat, t, wind, units):
    return np.array([q_alpha_pa_rad_dimless(pos[i], vel[i], quat[i], t[i], wind, units) for i in range(len(t))])

@jit(nopython=True)
def roll_direction_array(pos, quat):
    return np.array([yb_r_dot(pos[i], quat[i]) for i in range(len(pos))])

@jit(nopython=True)
def yb_r_dot(pos_eci, quat_eci2body):
    yb_dir_eci = quatrot(conj(quat_eci2body), np.array([0.0, 1.0, 0.0]))
    return yb_dir_eci.dot(normalize(pos_eci))

@jit(nopython=True)
def q_alpha_pa_rad_dimless(pos_eci, vel_eci, quat, t, wind, units):
    return angle_of_attack_all_rad_dimless(pos_eci, vel_eci, quat, t, wind, units) * dynamic_pressure_pa_dimless(pos_eci, vel_eci, t, wind, units)

@jit(nopython=True)
def angle_of_attack_all_rad_dimless(pos_eci_e, vel_eci_e, quat, t_e, wind, units):

    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return angle_of_attack_all_rad(pos_eci, vel_eci, quat, t, wind)

@jit(nopython=True)
def angle_of_attack_all_rad(pos_eci, vel_eci, quat, t, wind):

    thrust_dir_eci = quatrot(conj(quat), np.array([1.0, 0.0, 0.0]))
    
    pos_llh = ecef2geodetic(pos_eci[0],pos_eci[1],pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])
        
    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)
    
    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci

    c_alpha = normalize(vel_air_eci).dot(normalize(thrust_dir_eci))
    
    if c_alpha >= 1.0 or norm(vel_air_eci) < 0.001:
        return 0.0
    else:
        return acos(c_alpha)

@jit(nopython=True)
def angle_of_attack_ab_rad_dimless(pos_eci_e, vel_eci_e, quat, t_e, wind, units):

    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return angle_of_attack_ab_rad(pos_eci, vel_eci, quat, t, wind)

@jit(nopython=True)
def angle_of_attack_ab_rad(pos_eci, vel_eci, quat, t, wind):

    pos_llh = ecef2geodetic(pos_eci[0],pos_eci[1],pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])
        
    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)
    
    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci
    
    vel_air_body = quatrot(quat, vel_air_eci)
    
    if vel_air_body[0] < 0.001:
        return np.zeros(2)
    else:
        alpha_z = atan2(vel_air_body[2], vel_air_body[0])
        alpha_y = atan2(vel_air_body[1], vel_air_body[0])
        return np.array((alpha_z, alpha_y))

    
@jit(nopython=True)
def dynamic_pressure_pa_dimless(pos_eci_e, vel_eci_e, t_e, wind, units):

    pos_eci = pos_eci_e * units[0]
    vel_eci = vel_eci_e * units[1]
    t = t_e * units[2]
    return dynamic_pressure_pa(pos_eci, vel_eci, t, wind) 

@jit(nopython=True)
def dynamic_pressure_pa(pos_eci, vel_eci, t, wind):

    pos_llh = ecef2geodetic(pos_eci[0],pos_eci[1],pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])
    rho = airdensity_at(altitude_m)
        
    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)
    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci
    
    return 0.5 * vel_air_eci.dot(vel_air_eci) * rho

    
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

def initialize_xdict_6DoF_2(x_init, pdict, condition, unitdict, mode='LGR', dt=0.005, flag_display=True):
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
        to = pdict["params"][i]["timeAt_sec"]
        tf = pdict["params"][i]["timeFinishAt_sec"]
        tau = pdict["ps_params"][i]["tau"]
        
        if mode=='LG' or mode=='LGR':
            tau_x = np.hstack((-1.0, tau))
        else:
            tau_x = tau
        
        time_nodes = np.hstack((time_nodes, tau*(tf-to)/2.0 + (tf+to)/2.0))
        time_x_nodes = np.hstack((time_x_nodes, tau_x*(tf-to)/2.0 + (tf+to)/2.0))


    time_knots = np.array([e["timeAt_sec"] for e in pdict["params"]])
    xdict["t"] = (time_knots / unitdict["t"]).ravel()

    # 現在位置と目標軌道から、適当に目標位置・速度を決める
    
    if condition["rf_m"] is not None:
        r_final = condition["rf_m"]
    else:
        if condition["hp_m"] is None or condition["ha_m"] is None:
            print("DESTINATION ORBIT NOT DETERMINED!!")
            sys.exit()
    print(r_final)
    
    u_nodes = np.vstack([[[0.0, pdict["params"][i]["pitchrate_dps"],pdict["params"][i]["yawrate_dps"]]] * pdict["ps_params"][i]["nodes"] for i in range(num_sections)])
    xdict["u"] = (u_nodes / unitdict["u"]).ravel()
        
    u_table = np.hstack((time_nodes.reshape(-1,1),u_nodes))
    
    x_nodes, _ = rocket_simulation(x_init, u_table, pdict, time_nodes[0], time_x_nodes, dt)
    
    xdict["mass"] = x_nodes[:,0] / unitdict["mass"]
    xdict["position"] = (x_nodes[:,1:4] / unitdict["position"]).ravel()
    xdict["velocity"] = (x_nodes[:,4:7] / unitdict["velocity"]).ravel()
    xdict["quaternion"] = (x_nodes[:,7:11]).ravel()
    
    if flag_display:
        display_6DoF(output_6DoF(xdict, unitdict, time_x_nodes, time_nodes, pdict))
    return xdict

def initialize_xdict_6DoF_from_file(x_ref, pdict, condition, unitdict, mode='LGL', flag_display=True):
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
        to = pdict["params"][i]["timeAt_sec"]
        tf = pdict["params"][i]["timeFinishAt_sec"]
        tau = pdict["ps_params"][i]["tau"]
        
        if mode=='LG' or mode=='LGR':
            tau_x = np.hstack((-1.0, tau))
        else:
            tau_x = tau
        
        time_nodes = np.hstack((time_nodes, tau*(tf-to)/2.0 + (tf+to)/2.0))
        time_x_nodes = np.hstack((time_x_nodes, tau_x*(tf-to)/2.0 + (tf+to)/2.0))


    time_knots = np.array([e["timeAt_sec"] for e in pdict["params"]])
    xdict["t"] = (time_knots / unitdict["t"]).ravel()    
    
    xdict["mass"] = (interp1d(x_ref["time"], x_ref["mass"], fill_value="extrapolate")(time_x_nodes) / unitdict["mass"]).ravel()
    xdict["position"] = (interp1d(x_ref["time"], x_ref[["pos_ECI_X", "pos_ECI_Y", "pos_ECI_Z"]], axis=0, fill_value="extrapolate")(time_x_nodes) / unitdict["position"]).ravel()
    xdict["velocity"] = (interp1d(x_ref["time"], x_ref[["vel_ECI_X", "vel_ECI_Y", "vel_ECI_Z"]], axis=0, fill_value="extrapolate")(time_x_nodes) / unitdict["velocity"]).ravel()
    xdict["quaternion"] = (interp1d(x_ref["time"], x_ref[["quat_i2b_0", "quat_i2b_1", "quat_i2b_2", "quat_i2b_3"]], axis=0, fill_value="extrapolate")(time_x_nodes)).ravel()
    xdict["u"] = (interp1d(x_ref["time"], x_ref[["rate_P", "rate_Q", "rate_R"]], axis=0, fill_value="extrapolate")(time_nodes) / unitdict["u"]).ravel()

    if flag_display:
        display_6DoF(output_6DoF(xdict, unitdict, time_x_nodes, time_nodes, pdict))
    return xdict


def output_6DoF(xdict, unitdict, tx_res, tu_res, pdict):
    
        
    N = len(tx_res)

    unit_mass= unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_u = unitdict["u"]
    unit_t = unitdict["t"]

    mass_ = xdict["mass"] * unit_mass
    pos_ = xdict["position"].reshape(-1,3) * unit_pos
    vel_ = xdict["velocity"].reshape(-1,3) * unit_vel
    quat_ = xdict["quaternion"].reshape(-1,4)
    
    u_ = xdict["u"].reshape(-1,3) * unit_u
    
    
    out = { "event"      : [""] * N,
            "time"       : tx_res.round(6),
            "stage"      : [""] * N,
            "section"    : np.zeros(N,dtype="i4"),
            "thrust"     : np.zeros(N),
            "mass"       : mass_,
            "lat"        : np.zeros(N),
            "lon"        : np.zeros(N),
            "lat_IIP"    : np.zeros(N),
            "lon_IIP"    : np.zeros(N),
            "alt"        : np.zeros(N),
            "ha"         : np.zeros(N),
            "hp"         : np.zeros(N),
            "inc"        : np.zeros(N),
            "argp"       : np.zeros(N),
            "asnd"       : np.zeros(N),
            "tanm"       : np.zeros(N),
            "pos_ECI_X"  : pos_[:,0],
            "pos_ECI_Y"  : pos_[:,1],
            "pos_ECI_Z"  : pos_[:,2],
            "vel_ECI_X"  : vel_[:,0],
            "vel_ECI_Y"  : vel_[:,1],
            "vel_ECI_Z"  : vel_[:,2],
            "vel_NED_X"  : np.zeros(N),
            "vel_NED_Y"  : np.zeros(N),
            "vel_NED_Z"  : np.zeros(N),
            "quat_i2b_0" : quat_[:,0],
            "quat_i2b_1" : quat_[:,1],
            "quat_i2b_2" : quat_[:,2],
            "quat_i2b_3" : quat_[:,3],
            "accel_X"    : np.zeros(N),
            "aero_X"     : np.zeros(N),
            "heading"    : np.zeros(N),
            "pitch"      : np.zeros(N),
            "roll"       : np.zeros(N),
            "vi"         : norm(vel_,axis=1),
            "fpvgd"      : np.zeros(N),
            "azvgd"      : np.zeros(N),
            "thrustvec_X": np.zeros(N),
            "thrustvec_Y": np.zeros(N),
            "thrustvec_Z": np.zeros(N),
            "rate_P"     : np.interp(tx_res, tu_res, u_[:,0]),
            "rate_Q"     : np.interp(tx_res, tu_res, u_[:,1]),
            "rate_R"     : np.interp(tx_res, tu_res, u_[:,2]),
            "vr"         : np.zeros(N),
            "va"         : np.zeros(N),
            "aoa_total"  : np.zeros(N),
            "aoa_alpha"  : np.zeros(N),
            "aoa_beta"   : np.zeros(N),
            "q"          : np.zeros(N),
            "q-alpha"    : np.zeros(N),
            "M"          : np.zeros(N)
        }
    
    section = 0
    out["event"][0] = pdict["params"][0]["name"]

    for i in range(N):
        
        mass = mass_[i]
        pos = pos_[i]
        vel = vel_[i]
        quat = normalize(quat_[i])
        t = tx_res[i]
        
        out["section"][i] = section
        out["stage"][i] = pdict["params"][section]["rocketStage"]
        thrust_vac_n = pdict["params"][section]["thrust_n"]
        massflow_kgps = pdict["params"][section]["massflow_kgps"]
        airArea_m2 = pdict["params"][section]["airArea_m2"]
        nozzleArea_m2 = pdict["params"][section]["nozzleArea_m2"]
        if i >= pdict["ps_params"][section]["index_start"] + pdict["ps_params"][section]["nodes"] + section:
            out["event"][i] = pdict["params"][section+1]["name"]
            section += 1

        pos_llh = eci2geodetic(pos, t)
        altitude_m = geopotential_altitude(pos_llh[2])
        out["lat"][i], out["lon"][i], out["alt"][i]  = pos_llh
        
        elem = orbital_elements(pos, vel)
        out["ha"][i] = elem[0] * (1.0 + elem[1]) - 6378137
        out["hp"][i] = elem[0] * (1.0 - elem[1]) - 6378137
        out["inc"][i], out["asnd"][i], out["argp"][i], out["tanm"][i] = elem[2:6]
        
        vel_ground_ecef = vel_eci2ecef(vel, pos, t)
        vel_ground_ned  = quatrot(quat_ecef2nedg(eci2ecef(pos, t)), vel_ground_ecef)
        out["vel_NED_X"][i], out["vel_NED_Y"][i], out["vel_NED_Z"][i] = vel_ground_ned
        vel_ned         = quatrot(quat_eci2nedg(pos, t), vel)
        vel_air_ned     = vel_ground_ned - wind_ned(altitude_m, pdict["wind_table"])
        out["vr"][i] = norm(vel_ground_ecef)
        
        out["azvgd"][i] = degrees(atan2(vel_ned[1], vel_ned[0]))
        out["fpvgd"][i] = degrees(asin(-vel_ned[2] / norm(vel_ned)))
        
        q = 0.5 * norm(vel_air_ned)**2 * airdensity_at(pos_llh[2])
        out["q"][i] = q
        
        aoa_all_deg = angle_of_attack_all_rad(pos, vel, quat, t, pdict["wind_table"]) * 180.0 / np.pi
        aoa_ab_deg = angle_of_attack_ab_rad(pos, vel, quat, t, pdict["wind_table"]) * 180.0 / np.pi
        
        out["aoa_total"][i] = aoa_all_deg
        out["q-alpha"][i] = aoa_all_deg * q
        out["aoa_alpha"][i], out["aoa_beta"][i] = aoa_ab_deg

        thrustdir_eci = quatrot(conj(quat), np.array([1.0, 0.0, 0.0]))
        out["thrustvec_X"][i], out["thrustvec_Y"][i], out["thrustvec_Z"][i] = thrustdir_eci
        euler = euler_from_quat(quat_nedg2body(quat, pos, t))
        out["heading"][i] = euler[0]
        out["pitch"][i]   = euler[1]
        out["roll"][i]    = euler[2]

        #####
        rho = airdensity_at(altitude_m)
        p = airpressure_at(altitude_m)
        
        #対気速度
        
        pos_ecef = eci2ecef(pos, t)
        vel_ecef = vel_eci2ecef(vel, pos, t)
        vel_wind_ned = wind_ned(altitude_m, pdict["wind_table"])
        
        vel_wind_eci = quatrot(quat_nedg2eci(pos, t), vel_wind_ned)
        vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci
        mach_number = norm(vel_air_eci) / speed_of_sound(altitude_m)
        out["M"][i] = mach_number
        airAxialForce_coeff = np.interp(mach_number, pdict["ca_table"][:,0], pdict["ca_table"][:,1])
        out["va"][i] = norm(vel_air_eci)        
        
        ret = np.zeros(11)
        
        aero_n_eci = 0.5 * rho * norm(vel_air_eci) * -vel_air_eci * airArea_m2 * airAxialForce_coeff
        aero_n_body = quatrot(quat, aero_n_eci)

        thrust_n = thrust_vac_n - nozzleArea_m2 * p
        out["thrust"][i] = thrust_n
        thrustdir_eci = quatrot(conj(quat), np.array([1.0, 0.0, 0.0]))
        thrust_n_eci = thrustdir_eci * thrust_n
        gravity_eci = gravity(pos)
        out["aero_X"][i] = aero_n_body[0]
        out["accel_X"][i] = (thrust_n + aero_n_body[0]) / mass
        
        out["lat_IIP"][i], out["lon_IIP"][i], _ = posLLH_IIP_FAA(pos_ecef, vel_ecef)

        acc_eci = gravity_eci + (thrust_n_eci + aero_n_eci) / mass
        
        #####
        
    return pd.DataFrame(out)

