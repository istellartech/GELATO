import sys
import numpy as np
import pandas as pd
from numba import jit
from utils import *
from USStandardAtmosphere import *
from coordinate import *
from tools.plot_output import display_6DoF


@jit(nopython=True)
def dynamics_mass(param):

    massflow_kgps = param[1]    
    
    return -massflow_kgps

@jit(nopython=True)
def dynamics_position(vel_eci):
    return vel_eci

@jit(nopython=True)
def dynamics_velocity(mass, pos_eci, vel_eci, quat_eci2body, t, param, wind, ca):
    
    acc_eci = np.zeros(vel_eci.shape)

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
        
    return acc_eci

@jit(nopython=True)
def dynamics_quaternion(quat_eci2body, u):

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
                zlt = pdict["params"][event_index]["do_zeroliftturn"]
                x[0] -= pdict["params"][event_index]["mass_jettison_kg"]
        
        u = np.array([np.interp(t, u_table[:,0], u_table[:,i+1]) for i in range(3)])
        x = runge_kutta_4d(lambda xa,ta:dynamics(xa, u, ta, param, zlt, wind, ca), x, t, dt)
        t = t + dt
            
        if zlt:
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


def equality_init(xdict, unitdict, condition):
    
    con = []
    mass_ = xdict["mass"] * unitdict["mass"]
    pos_ = xdict["position"].reshape(-1,3) * unitdict["position"]
    vel_ = xdict["velocity"].reshape(-1,3) * unitdict["velocity"]
    quat_ = xdict["quaternion"].reshape(-1,4)

    #initial condition
    if condition["OptimizationMode"] != "Payload":
        con.append((mass_[0] - condition["init"]["mass"]) / unitdict["mass"])
    con.append((pos_[0] - condition["init"]["position"]) / unitdict["position"])
    con.append((vel_[0] - condition["init"]["velocity"]) / unitdict["velocity"])
    con.append(quat_[0] - condition["init"]["quaternion"])

    return np.concatenate(con, axis=None)

def equality_time(xdict, pdict, unitdict, condition):
    con = []
    unit_t = unitdict["t"]
    
    t = xdict["t"] * unit_t

    num_sections = pdict["num_sections"]
    
    #knotting time
    con.append([(t[i] - pdict["params"][i]["timeAt_sec"]) / unit_t for i in range(num_sections+1) if pdict["params"][i]["timeFixed"]])
    
    return np.concatenate(con, axis=None)


def equality_dynamics_mass(xdict, pdict, unitdict):
    con = []

    unit_mass = unitdict["mass"]
    mass_ = xdict["mass"] * unit_mass
    t = xdict["t"] * unitdict["t"]

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
        
        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]

        lh = pdict["ps_params"][i]["D"].dot(mass_i_ / unit_mass)
        rh = np.full(n, -param[1] * (tf-to) / 2.0 / unit_mass) #dynamics_mass
        con.append(lh - rh)
            
            
    return np.concatenate(con, axis=None)


def equality_dynamics_position(xdict, pdict, unitdict):
    con = []

    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    pos_ = xdict["position"].reshape(-1,3) * unit_pos
    vel_ = xdict["velocity"].reshape(-1,3) * unit_vel
    t = xdict["t"] * unitdict["t"]

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

        lh = pdict["ps_params"][i]["D"].dot(pos_i_ / unit_pos)
        rh = vel_i_[1:] * (tf-to) / 2.0 / unit_pos #dynamics_position
        con.append((lh - rh).ravel())
                        
    return np.concatenate(con, axis=None)

def equality_dynamics_velocity(xdict, pdict, unitdict):
    con = []

    unit_mass = unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    mass_ = xdict["mass"] * unit_mass 
    pos_ = xdict["position"].reshape(-1,3) * unit_pos
    vel_ = xdict["velocity"].reshape(-1,3) * unit_vel
    quat_ = xdict["quaternion"].reshape(-1,4)
    t = xdict["t"] * unitdict["t"]

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
        t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
        
        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]

        wind = pdict["wind_table"]
        ca = pdict["ca_table"]

        lh = pdict["ps_params"][i]["D"].dot(vel_i_ / unit_vel)
        rh = dynamics_velocity(mass_i_[1:], pos_i_[1:], vel_i_[1:], quat_i_[1:], t_nodes, param, wind, ca) * (tf-to) / 2.0 / unit_vel
        con.append((lh - rh).ravel())
                        
    return np.concatenate(con, axis=None)

def equality_dynamics_quaternion(xdict, pdict, unitdict):
    con = []

    quat_ = xdict["quaternion"].reshape(-1,4)
    u_ = xdict["u"].reshape(-1,3) * unitdict["u"]
    t = xdict["t"] * unitdict["t"]

    num_sections = pdict["num_sections"]
    
    param = np.zeros(5)
    
    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        quat_i_ = quat_[a+i:b+i+1]
        u_i_ = u_[a:b]
        to = t[i]
        tf = t[i+1]
        # t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
        
        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]

        lh = pdict["ps_params"][i]["D"].dot(quat_i_)
        rh = dynamics_quaternion(quat_i_[1:], u_i_) * (tf-to) / 2.0
        con.append((lh - rh).ravel())
                        
    return np.concatenate(con, axis=None)


def equality_knot_LGR(xdict, pdict, unitdict):
    con = []

    unit_mass= unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]

    mass_ = xdict["mass"] * unit_mass
    pos_ = xdict["position"].reshape(-1,3) * unit_pos
    vel_ = xdict["velocity"].reshape(-1,3) * unit_vel
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
        con.append((mass_next_ - mass_final_ + pdict["params"][i+1]["mass_jettison_kg"]) / unit_mass)

        pos_next_ = pos_[b+i+1]
        pos_final_ = pos_i_[-1]
        con.append((pos_next_ - pos_final_) / unit_pos)

        vel_next_ = vel_[b+i+1]
        vel_final_ = vel_i_[-1]
        con.append((vel_next_ - vel_final_) / unit_vel)

        quat_next_ = quat_[b+i+1]
        quat_final_ = quat_i_[-1]
        con.append((quat_next_ - quat_final_))

    return np.concatenate(con, axis=None)


def equality_6DoF_LGR_terminal(xdict, pdict, unitdict, condition):
    con = []

    #unit_mass= unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    #unit_u = unitdict["u"]
    #unit_t = unitdict["t"]

    #mass_ = xdict["mass"] * unit_mass
    pos_ = xdict["position"].reshape(-1,3) * unit_pos
    vel_ = xdict["velocity"].reshape(-1,3) * unit_vel
    #quat_ = xdict["quaternion"].reshape(-1,4)
    
    #u_ = xdict["u"].reshape(-1,3) * unit_u
    #t = xdict["t"] * unit_t

    num_sections = pdict["num_sections"]
    
    
    param = np.zeros(5)
    
    i = num_sections - 1

    a = pdict["ps_params"][i]["index_start"]
    n = pdict["ps_params"][i]["nodes"]
    b = a + n
    pos_i_ = pos_[a+i:b+i+1]
    vel_i_ = vel_[a+i:b+i+1]
            
    param[0] = pdict["params"][i]["thrust_n"]
    param[1] = pdict["params"][i]["massflow_kgps"]
    param[2] = pdict["params"][i]["airArea_m2"]
    param[4] = pdict["params"][i]["nozzleArea_m2"]
            

    # terminal conditions

    pos_f = pos_i_[-1]
    vel_f = vel_i_[-1]

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


def equality_6DoF_rate(xdict, pdict, unitdict):
    con = []

    unit_pos = unitdict["position"]
    unit_u = unitdict["u"]

    pos_ = xdict["position"].reshape(-1,3) * unit_pos
    quat_ = xdict["quaternion"].reshape(-1,4)
    
    u_ = xdict["u"].reshape(-1,3) * unit_u

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
        if att == "zero-lift-turn" or att == "free":
            #zero-lift-turn: pitch/yaw free, roll hold
            con.append(u_i_[:,0])
        
        else:    
            # pitch/yaw rate constant
            if att != "pitch-yaw-free":
                con.append((u_i_[1:,1:] - u_i_[0,1:]).ravel())
            
            
            if pdict["params"][i]["hold_yaw"]:
                
                # yaw hold
                con.append(u_i_[0,2])
                con.append(u_i_[:,0])
                if pdict["params"][i]["hold_pitch"]:
                    # total attitude hold
                    con.append(u_i_[0,1])
            else:
                # roll constraint
                con.append(roll_direction_array(pos_i_[1:], quat_i_[1:]))
                
                if att == "same-rate":
                    # same pitch/yaw rate as previous section
                    uf_prev = u_[a-1]
                    con.append(u_i_[0,1:] - uf_prev[1:])

            
            
    return np.concatenate(con, axis=None)


def inequality_time(xdict, pdict):
    con = []
    t_normal = xdict["t"]

    for i in range(pdict["num_sections"]-1):
        if not (pdict["params"][i]["timeFixed"] and pdict["params"][i+1]["timeFixed"]):
            con.append(t_normal[i+1] - t_normal[i])

    return np.array(con)

def inequality_6DoF(xdict, pdict, unitdict, condition):
    
    con = []

    unit_mass= unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_u = unitdict["u"]
    unit_t = unitdict["t"]

    #mass_ = xdict["mass"] * unit_mass
    pos_ = xdict["position"].reshape(-1,3) * unit_pos
    vel_ = xdict["velocity"].reshape(-1,3) * unit_vel
    quat_ = xdict["quaternion"].reshape(-1,4)
    
    u_ = xdict["u"].reshape(-1,3) * unit_u
    t = xdict["t"] * unit_t

    num_sections = pdict["num_sections"]
    
    
    
    for i in range(num_sections-1):
        a = pdict["ps_params"][i]["index_start"]
        n = pdict["ps_params"][i]["nodes"]
        b = a + n
        #mass_i_ = mass_[a+i:b+i+1]
        pos_i_ = pos_[a+i:b+i+1]
        vel_i_ = vel_[a+i:b+i+1]
        quat_i_ = quat_[a+i:b+i+1]
        u_i_ = u_[a:b]
        to = t[i]
        tf = t[i+1]
        t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
        t_i_ = np.hstack((to, t_nodes))

        # kick turn
        if "kick" in pdict["params"][i]["attitude"]:
            con.append(-u_i_[:,1])
            #con.append(u_i_[:,1]+0.36)
        
        section_name = pdict["params"][i]["name"]

        # angle of attack
        if section_name in condition["aoa_max_deg"]:
            aoa_max = condition["aoa_max_deg"][section_name]["value"] * np.pi / 180.0
            if condition["aoa_max_deg"][section_name]["range"] == "all":
                con.append(1.0 - aoa_zerolift_array(pos_i_, vel_i_, quat_i_, t_i_, pdict["wind_table"]) / aoa_max)
            elif condition["aoa_max_deg"][section_name]["range"] == "initial":
                con.append(1.0 - angle_of_attack_all_rad(pos_i_[0], vel_i_[0], quat_i_[0], to, pdict["wind_table"]) / aoa_max)
        
        # max-Q
        if section_name in condition["q_max_pa"]:
            q_max = condition["q_max_pa"][section_name]["value"]
            if condition["q_max_pa"][section_name]["range"] == "all":
                con.append(1.0 - dynamic_pressure_array(pos_i_, vel_i_, t_i_, pdict["wind_table"]) / q_max)
            elif condition["q_max_pa"][section_name]["range"] == "initial":
                con.append(1.0 - dynamic_pressure_pa(pos_i_[0], vel_i_[0], to, pdict["wind_table"]) / q_max)

        # max-Qalpha
        if section_name in condition["q-alpha_max_pa-deg"]:
            qalpha_max = condition["q-alpha_max_pa-deg"][section_name]["value"] * np.pi / 180.0
            if condition["q-alpha_max_pa-deg"][section_name]["range"] == "all":
                con.append(1.0 - q_alpha_array(pos_i_, vel_i_, quat_i_, t_i_, pdict["wind_table"]) / qalpha_max)
            elif condition["q-alpha_max_pa-deg"][section_name]["range"] == "initial":
                con.append(1.0 - q_alpha_pa_rad(pos_i_[0], vel_i_[0], quat_i_[0], to, pdict["wind_table"]) / qalpha_max)

        
    return np.concatenate(con, axis=None)    

@jit(nopython=True)
def dynamic_pressure_array(pos, vel, t, wind):
    return np.array([dynamic_pressure_pa(pos[i], vel[i], t[i], wind) for i in range(len(t))])

@jit(nopython=True)
def aoa_zerolift_array(pos, vel, quat, t, wind):
    return np.array([angle_of_attack_all_rad(pos[i], vel[i], quat[i], t[i], wind) for i in range(len(t))])

@jit(nopython=True)
def q_alpha_array(pos, vel, quat, t, wind):
    return np.array([q_alpha_pa_rad(pos[i], vel[i], quat[i], t[i], wind) for i in range(len(t))])

@jit(nopython=True)
def roll_direction_array(pos, quat):
    return np.array([yb_r_dot(pos[i], quat[i]) for i in range(len(pos))])

@jit(nopython=True)
def yb_r_dot(pos_eci, quat_eci2body):
    yb_dir_eci = quatrot(conj(quat_eci2body), np.array([0.0, 1.0, 0.0]))
    return yb_dir_eci.dot(normalize(pos_eci))

@jit(nopython=True)
def q_alpha_pa_rad(pos_eci, vel_eci, quat, t, wind):
    return angle_of_attack_all_rad(pos_eci, vel_eci, quat, t, wind) * dynamic_pressure_pa(pos_eci, vel_eci, t, wind)

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
def dynamic_pressure_pa(pos_eci, vel_eci, t, wind):
        
    pos_llh = ecef2geodetic(pos_eci[0],pos_eci[1],pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])
    rho = airdensity_at(altitude_m)
        
    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)
    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci
    
    return 0.5 * vel_air_eci.dot(vel_air_eci) * rho

    

def cost_6DoF(xdict, condition):
    
    if condition["OptimizationMode"] == "Payload":
        return -xdict["mass"][0] #初期質量(無次元)を最大化
    else:
        return xdict["t"][-1] #到達時間を最小化(=余剰推進剤を最大化)


def initialize_xdict_6DoF_2(x_init, pdict, condition, unitdict, mode='LGL', dt=0.005, flag_display=True):
    
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
            "time"       : tx_res,
            "section"    : np.zeros(N,dtype="i4"),
            "thrust"     : np.zeros(N),
            "mass"       : mass_,
            "lat"        : np.zeros(N),
            "lon"        : np.zeros(N),
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
        if i >= pdict["ps_params"][section]["index_start"] + pdict["ps_params"][section]["nodes"] + section:
            out["event"][i] = pdict["params"][section+1]["name"]
            section += 1
        thrust_vac_n = pdict["params"][section]["thrust_n"]
        massflow_kgps = pdict["params"][section]["massflow_kgps"]
        airArea_m2 = pdict["params"][section]["airArea_m2"]
        nozzleArea_m2 = pdict["params"][section]["nozzleArea_m2"]

        pos_llh = eci2geodetic(pos, t)
        altitude_m = geopotential_altitude(pos_llh[2])
        out["lat"][i]  = pos_llh[0]
        out["lon"][i]  = pos_llh[1]
        out["alt"][i]  = pos_llh[2]
        
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
        
        acc_eci = gravity_eci + (thrust_n_eci + aero_n_eci) / mass
        
        #####
        
    return pd.DataFrame(out)

