import sys
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numba import jit
from utils import *
from PSfunctions import *
from USStandardAtmosphere import *
from coordinate import *
from tools.plot_output import display_6DoF
from tools.IIP import posLLH_IIP_FAA
from user_constraints import *

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
