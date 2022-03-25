import sys
import numpy as np
import pandas as pd
from numba import jit
from utils import *
from USStandardAtmosphere import *
from coordinate import *
import matplotlib.pyplot as plt


@jit('f8[:](f8[:],f8[:],f8,f8[:],f8[:,:],f8[:,:])',nopython=True)
def dynamics(x, u, t, param, wind, ca):

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
        x = runge_kutta_4d(lambda xa,ta:dynamics(xa, u, ta, param, wind, ca), x, t, dt)
        t = t + dt
            
        #if zlt:
        #    x[7:11] = zerolift_turn_correct(x, t, wind)
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


def equality_init(xdict, pdict, unit_xut, condition):
    con = []
    unit_x = unit_xut["x"]
    unit_u = unit_xut["u"]
    unit_t = unit_xut["t"]
    
    xx = xdict["xvars"].reshape(-1,pdict["num_states"]) * unit_xut["x"]
    uu = xdict["uvars"].reshape(-1,pdict["num_controls"]) * unit_xut["u"]
    t = xdict["t"] * unit_t

    num_sections = pdict["num_sections"]
    
    #initial condition
    con.append((xx[0,1:11] - condition["x_init"][1:11]) / unit_x[1:11])
    if not condition["OptimizationMode"]["Maximize initial mass"]:
        con.append((xx[0,0] - condition["x_init"][0]) / unit_x[0])

    return np.concatenate(con, axis=None)

def equality_time(xdict, pdict, unit_xut, condition):
    con = []
    unit_x = unit_xut["x"]
    unit_u = unit_xut["u"]
    unit_t = unit_xut["t"]
    
    xx = xdict["xvars"].reshape(-1,pdict["num_states"]) * unit_xut["x"]
    uu = xdict["uvars"].reshape(-1,pdict["num_controls"]) * unit_xut["u"]
    t = xdict["t"] * unit_t

    num_sections = pdict["num_sections"]
    
    #knotting time
    free_tf = condition["OptimizationMode"]["Maximize excess propellant mass"]
    con.append([(t[i] - pdict["params"][i]["timeAt_sec"]) / unit_t for i in range(num_sections+1) if pdict["params"][i]["timeFixed"] and not free_tf])
    
    return np.concatenate(con, axis=None)


def equality_6DoF_LG_diff(xdict, pdict, unit_xut, condition):
    con = []
    unit_x = unit_xut["x"]
    unit_u = unit_xut["u"]
    unit_t = unit_xut["t"]
    
    xx = xdict["xvars"].reshape(-1,pdict["num_states"]) * unit_xut["x"]
    uu = xdict["uvars"].reshape(-1,pdict["num_controls"]) * unit_xut["u"]
    t = xdict["t"] * unit_t

    num_sections = pdict["num_sections"]
    

    
    param = np.zeros(5)
    
    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        b = a + pdict["ps_params"][i]["nodes"]
        x = xx[a+i:b+i+1]
        u = uu[a:b]
        to = t[i]
        tf = t[i+1]
        t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
        
        # Pseudo-spectral constraints
        
        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]
        zlt = pdict["params"][i]["do_zeroliftturn"]
        
        ps = equality_ps_6DoF_LG(x, u, t_nodes, param, pdict["wind_table"], pdict["ca_table"], pdict["ps_params"][i]["D"], tf, to, unit_x)
        
        con.append(ps.ravel())

            
            
    return np.concatenate(con, axis=None)


def equality_6DoF_LG_knot(xdict, pdict, unit_xut, condition):
    con = []
    unit_x = unit_xut["x"]
    unit_u = unit_xut["u"]
    unit_t = unit_xut["t"]
    
    xx = xdict["xvars"].reshape(-1,pdict["num_states"]) * unit_xut["x"]
    uu = xdict["uvars"].reshape(-1,pdict["num_controls"]) * unit_xut["u"]
    t = xdict["t"] * unit_t

    num_sections = pdict["num_sections"]
    

    
    param = np.zeros(5)
    
    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        b = a + pdict["ps_params"][i]["nodes"]
        x = xx[a+i:b+i+1]
        u = uu[a:b]
        to = t[i]
        tf = t[i+1]
        t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
        
        # Pseudo-spectral constraints
        
        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]
        zlt = pdict["params"][i]["do_zeroliftturn"]
        

        if i < num_sections-1:
            # knotting constraints: 現在のsectionの末尾と次のsectionの先頭の連続性
            xoi_next = xx[b+i+1]
            xfi = x_final(x, u, t_nodes, param, pdict["wind_table"], pdict["ca_table"], pdict["ps_params"][i]["weight"], tf, to, unit_x)
            d_xx = (xoi_next - xfi) / unit_x
            uoi_next = uu[b+1]
            ufi = u[-1]
            d_uu = (uoi_next - ufi) / unit_u
            con.append(d_xx[1:7]) # pos, vel
            
            ALMA_mode = False
            if zlt and ALMA_mode:
                con.append(x[0,7:11] - zerolift_turn_correct(x[0],to, pdict["wind_table"])) # quatの不連続を許可
            else:
                con.append(d_xx[7:11]) # quat
            con.append((d_xx[0]) + pdict["params"][i+1]["mass_jettison_kg"] / unit_x[0]) # mass : 投棄物考慮

    return np.concatenate(con, axis=None)


def equality_6DoF_LG_terminal(xdict, pdict, unit_xut, condition):
    con = []
    unit_x = unit_xut["x"]
    unit_u = unit_xut["u"]
    unit_t = unit_xut["t"]
    
    xx = xdict["xvars"].reshape(-1,pdict["num_states"]) * unit_xut["x"]
    uu = xdict["uvars"].reshape(-1,pdict["num_controls"]) * unit_xut["u"]
    t = xdict["t"] * unit_t

    num_sections = pdict["num_sections"]
    
    
    param = np.zeros(5)
    
    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        b = a + pdict["ps_params"][i]["nodes"]
        x = xx[a+i:b+i+1]
        u = uu[a:b]
        to = t[i]
        tf = t[i+1]
        t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
        
        # Pseudo-spectral constraints
        
        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]
        zlt = pdict["params"][i]["do_zeroliftturn"]
                

        if i < num_sections-1:
            # knotting constraints: 現在のsectionの末尾と次のsectionの先頭の連続性
            ufi = u[-1]

        else:
            # terminal conditions
            xf = x_final(x, u, t_nodes, param, pdict["wind_table"], pdict["ca_table"], pdict["ps_params"][i]["weight"], tf, to, unit_x)
            
            pos_f = xf[1:4]
            vel_f = xf[4:7]
            elem = orbital_elements(pos_f, vel_f)
            
            if condition["hp_km"] is not None:
                hp = elem[0] * (1.0 - elem[1]) + 6378137
                con.append((hp - condition["hp_km"]*1000) / unit_x[1])
                
            if condition["ha_km"] is not None:
                ha = elem[0] * (1.0 + elem[1]) + 6378137
                con.append((ha - condition["ha_km"]*1000) / unit_x[1])
                
            if condition["rf_km"] is not None:
                rf = norm(pos_f)
                con.append((rf - condition["rf_km"]*1000) / unit_x[1])
                
            if condition["vtf_kmps"] is not None:
                vrf = vel_f.dot(normalize(pos_f))
                vtf = sqrt(norm(vel_f)**2 - vrf**2)
                con.append((vtf - condition["vtf_kmps"]*1000) / unit_x[4])
                
            if condition["vf_elev_deg"] is not None:
                cos_vf_angle = normalize(vel_f).dot(normalize(pos_f))
                con.append(cos(radians(90.0-condition["vf_elev_deg"])) - cos_vf_angle)
                
            if condition["vrf_kmps"] is not None:
                vrf = vel_f.dot(normalize(pos_f))
                con.append((vrf - condition["vrf_kmps"]*1000) / unit_x[4])
                
            if condition["inclination_deg"] is not None:
                con.append((elem[2] - condition["inclination_deg"]) / 90.0)            
        
        #rate constraint

            
            
    return np.concatenate(con, axis=None)


def equality_6DoF_LG_rate(xdict, pdict, unit_xut, condition):
    con = []
    unit_x = unit_xut["x"]
    unit_u = unit_xut["u"]
    unit_t = unit_xut["t"]
    
    xx = xdict["xvars"].reshape(-1,pdict["num_states"]) * unit_xut["x"]
    uu = xdict["uvars"].reshape(-1,pdict["num_controls"]) * unit_xut["u"]
    t = xdict["t"] * unit_t

    num_sections = pdict["num_sections"]
    
    param = np.zeros(5)
    
    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        b = a + pdict["ps_params"][i]["nodes"]
        x = xx[a+i:b+i+1]
        u = uu[a:b]
        to = t[i]
        tf = t[i+1]
        t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
        
        # Pseudo-spectral constraints
        
        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]
        zlt = pdict["params"][i]["do_zeroliftturn"]
        
        #rate constraint
        
        att = pdict["params"][i]["attitude"]
        if att == "zero-lift-turn":
            #zero-lift-turn: pitch/yaw free, roll hold
            con.append(u[:,0])
        
        else:    
            # pitch/yaw rate constant
            if att != "pitch-yaw-free":
                con.append((u[1:,1:] - u[0,1:]).ravel())
            
            
            if pdict["params"][i]["hold_yaw"]:
                
                # yaw hold
                con.append(u[0,2])
                con.append(u[:,0])
                if pdict["params"][i]["hold_pitch"]:
                    # total attitude hold
                    con.append(u[0,1])
            else:
                # roll constraint
                con.append(roll_direction(x[1:]))
                
                if att == "same-rate":
                    # same pitch/yaw rate as previous section
                    uf_prev = uu[a-1]
                    con.append(u[0,1:] - uf_prev[1:])

            
            
    return np.concatenate(con, axis=None)




def equality_6DoF_LG(xdict, pdict, unit_xut, condition):
    con = []
    unit_x = unit_xut["x"]
    unit_u = unit_xut["u"]
    unit_t = unit_xut["t"]
    
    xx = xdict["xvars"].reshape(-1,pdict["num_states"]) * unit_xut["x"]
    uu = xdict["uvars"].reshape(-1,pdict["num_controls"]) * unit_xut["u"]
    t = xdict["t"] * unit_t

    num_sections = pdict["num_sections"]
    
    #initial condition
    con.append((xx[0,1:11] - condition["x_init"][1:11]) / unit_x[1:11])
    if not condition["OptimizationMode"]["Maximize initial mass"]:
        con.append((xx[0,0] - condition["x_init"][0]) / unit_x[0])

    #knotting time
    free_tf = condition["OptimizationMode"]["Maximize excess propellant mass"]
    con.append([(t[i] - pdict["params"][i]["timeAt_sec"]) / unit_t for i in range(num_sections+1) if pdict["params"][i]["timeFixed"] and not free_tf])
    

    
    param = np.zeros(5)
    
    for i in range(num_sections):
        a = pdict["ps_params"][i]["index_start"]
        b = a + pdict["ps_params"][i]["nodes"]
        x = xx[a+i:b+i+1]
        u = uu[a:b]
        to = t[i]
        tf = t[i+1]
        t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
        
        # Pseudo-spectral constraints
        
        param[0] = pdict["params"][i]["thrust_n"]
        param[1] = pdict["params"][i]["massflow_kgps"]
        param[2] = pdict["params"][i]["airArea_m2"]
        param[4] = pdict["params"][i]["nozzleArea_m2"]
        zlt = pdict["params"][i]["do_zeroliftturn"]
        
        ps = equality_ps_6DoF_LG(x, u, t_nodes, param, pdict["wind_table"], pdict["ca_table"], pdict["ps_params"][i]["D"], tf, to, unit_x)
        
        con.append(ps.ravel())
        

        if i < num_sections-1:
            # knotting constraints: 現在のsectionの末尾と次のsectionの先頭の連続性
            xoi_next = xx[b+i+1]
            xfi = x_final(x, u, t_nodes, param, pdict["wind_table"], pdict["ca_table"], pdict["ps_params"][i]["weight"], tf, to, unit_x)
            d_xx = (xoi_next - xfi) / unit_x
            uoi_next = uu[b+1]
            ufi = u[-1]
            d_uu = (uoi_next - ufi) / unit_u
            con.append(d_xx[1:7]) # pos, vel
            
            ALMA_mode = False
            if zlt and ALMA_mode:
                con.append(x[0,7:11] - zerolift_turn_correct(x[0],to, pdict["wind_table"])) # quatの不連続を許可
            else:
                con.append(d_xx[7:11]) # quat
            con.append((d_xx[0]) + pdict["params"][i+1]["mass_jettison_kg"] / unit_x[0]) # mass : 投棄物考慮
        else:
            # terminal conditions
            xf = x_final(x, u, t_nodes, param, pdict["wind_table"], pdict["ca_table"], pdict["ps_params"][i]["weight"], tf, to, unit_x)
            
            pos_f = xf[1:4]
            vel_f = xf[4:7]
            elem = orbital_elements(pos_f, vel_f)
            
            if condition["hp_km"] is not None:
                hp = elem[0] * (1.0 - elem[1]) + 6378137
                con.append((hp - condition["hp_km"]*1000) / unit_x[1])
                
            if condition["ha_km"] is not None:
                ha = elem[0] * (1.0 + elem[1]) + 6378137
                con.append((ha - condition["ha_km"]*1000) / unit_x[1])
                
            if condition["rf_km"] is not None:
                rf = norm(pos_f)
                con.append((rf - condition["rf_km"]*1000) / unit_x[1])
                
            if condition["vtf_kmps"] is not None:
                vrf = vel_f.dot(normalize(pos_f))
                vtf = sqrt(norm(vel_f)**2 - vrf**2)
                con.append((vtf - condition["vtf_kmps"]*1000) / unit_x[4])
                
            if condition["vf_elev_deg"] is not None:
                cos_vf_angle = normalize(vel_f).dot(normalize(pos_f))
                con.append(cos(radians(90.0-condition["vf_elev_deg"])) - cos_vf_angle)
                
            if condition["vrf_kmps"] is not None:
                vrf = vel_f.dot(normalize(pos_f))
                con.append((vrf - condition["vrf_kmps"]*1000) / unit_x[4])
                
            if condition["inclination_deg"] is not None:
                con.append((elem[2] - condition["inclination_deg"]) / 90.0)            
        
        #rate constraint
        
        att = pdict["params"][i]["attitude"]
        if att == "zero-lift-turn":
            #zero-lift-turn: pitch/yaw free, roll hold
            con.append(u[:,0])
        
        else:    
            # pitch/yaw rate constant
            if att != "pitch-yaw-free":
                con.append((u[1:,1:] - u[0,1:]).ravel())
            
            
            if pdict["params"][i]["hold_yaw"]:
                
                # yaw hold
                con.append(u[0,2])
                con.append(u[:,0])
                if pdict["params"][i]["hold_pitch"]:
                    # total attitude hold
                    con.append(u[0,1])
            else:
                # roll constraint
                con.append(roll_direction(x[1:]))
                
                if att == "same-rate":
                    # same pitch/yaw rate as previous section
                    uf_prev = uu[a-1]
                    con.append(u[0,1:] - uf_prev[1:])

            
            
    return np.concatenate(con, axis=None)


@jit('f8[:,:](f8[:,:],f8[:,:],f8[:],f8[:],f8[:,:],f8[:,:],f8[:,:],f8,f8,f8[:])',nopython=True)
def equality_ps_6DoF_LG(x, u, t, param, wind, ca, D, tf, to, unit_x):

    lh = D.dot(x / unit_x)

    n = len(t)
    r = np.zeros((n, len(x[0])))
    for i in range(n):
        r[i] = dynamics(x[i+1], u[i], t[i], param, wind, ca)
    rh =  (tf-to) / 2.0 * r / unit_x
    ps = lh - rh

    return ps


@jit('f8[:](f8[:,:],f8[:,:],f8[:],f8[:],f8[:,:],f8[:,:],f8[:],f8,f8,f8[:])',nopython=True)
def x_final(x, u, t, param, wind, ca, weight, tf, to, unit_x):

    n = len(t)
    r = np.zeros((n, len(x[0])))
    for i in range(n):
        r[i] = dynamics(x[i+1], u[i], t[i], param, wind, ca)
    rh =  (tf-to) / 2.0 * r

    return x[0] + weight.dot(rh)


def inequality_6DoF_LG(xdict, pdict, unit_xut, condition):
    
    con = []
    
    xx = xdict["xvars"].reshape(-1,pdict["num_states"]) * unit_xut["x"]
    uu = xdict["uvars"].reshape(-1,pdict["num_controls"]) * unit_xut["u"]
    t = xdict["t"] * unit_xut["t"]
            
    for i in range(pdict["num_sections"]):
        
        a = pdict["ps_params"][i]["index_start"]
        b = a + pdict["ps_params"][i]["nodes"]
        x = xx[a+i:b+i+1]
        u = uu[a:b]
        to = t[i]
        tf = t[i+1]
        t_nodes = pdict["ps_params"][i]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0        

        # kick turn
        if "kick" in pdict["params"][i]["attitude"]:
            con.append(-u[:,1])
            #con.append(u[:,1]+0.36)
        
        # zerolift turn
        if pdict["params"][i]["do_zeroliftturn"]:
            if condition["aoa_max_deg"] is not None:
                con.append(-aoa_zerolift(x[1:], u, t_nodes, pdict["wind_table"]) + radians(condition["aoa_max_deg"]))
        
        # max-Q and max-Q-alpha
        if condition["q-alpha_max_kpa-deg"] is not None:
            qalpha_max = condition["q-alpha_max_kpa-deg"] * 1000 * np.pi / 180.0
            con.append(1.0 - q_alpha_zerolift(x[1:], u, t_nodes, pdict["wind_table"]) / qalpha_max)
        if condition["q_max_kpa"] is not None:
            q_max = condition["q_max_kpa"] * 1000
            con.append(1.0 - dynamic_pressure(x[1:], u, t_nodes, pdict["wind_table"]) / q_max)

    
        # dynamic pressure at SEP
        if pdict["params"][i]["rocketStage"] > pdict["params"][i-1]["rocketStage"]:
            if condition["q_sep_max_pa"] is not None:
                con.append(1.0 - dynamic_pressure_pa(x[0], 0.0, to, pdict["wind_table"]) / condition["q_sep_max_pa"])
        
    return np.concatenate(con, axis=None)    

@jit(nopython=True)
def dynamic_pressure(x, u, t, wind):
    return np.array([dynamic_pressure_pa(x[i], u[i], t[i], wind) for i in range(len(t))])

@jit(nopython=True)
def aoa_zerolift(x, u, t, wind):
    return np.array([angle_of_attack_all_rad(x[i], u[i], t[i], wind) for i in range(len(t))])

@jit(nopython=True)
def q_alpha_zerolift(x, u, t, wind):
    return np.array([angle_of_attack_all_rad(x[i], u[i], t[i], wind) * dynamic_pressure_pa(x[i], u[i], t[i], wind)  for i in range(len(t))])

@jit(nopython=True)
def roll_direction(x):
    return np.array([yb_r_dot(x[i]) for i in range(len(x))])

@jit(nopython=True)
def yb_r_dot(x):
    pos_eci = x[1:4]
    quat_eci2body = x[7:11]
    yb_dir_eci = quatrot(conj(quat_eci2body), np.array([0.0, 1.0, 0.0]))
    return yb_dir_eci.dot(normalize(pos_eci))

@jit(nopython=True)
def angle_of_attack_all_rad(x, u, t, wind):
    
    mass = x[0]
    pos_eci = x[1:4]
    vel_eci = x[4:7]
    thrust_dir_eci = quatrot(conj(x[7:11]), np.array([1.0, 0.0, 0.0]))
    
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
def angle_of_attack_ab_rad(x, u, t, wind):
    
    mass = x[0]
    pos_eci = x[1:4]
    vel_eci = x[4:7]
    quat_eci2body = x[7:11]
    
    pos_llh = ecef2geodetic(pos_eci[0],pos_eci[1],pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])
        
    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)
    
    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci
    
    vel_air_body = quatrot(quat_eci2body, vel_air_eci)
    
    if vel_air_body[0] < 0.001:
        return np.zeros(2)
    else:
        alpha_z = atan2(vel_air_body[2], vel_air_body[0])
        alpha_y = atan2(vel_air_body[1], vel_air_body[0])
        return np.array((alpha_z, alpha_y))

    
@jit(nopython=True)
def dynamic_pressure_pa(x, u, t, wind):
    
    mass = x[0]
    pos_eci = x[1:4]
    vel_eci = x[4:7]
    
    pos_llh = ecef2geodetic(pos_eci[0],pos_eci[1],pos_eci[2])
    altitude_m = geopotential_altitude(pos_llh[2])
    rho = airdensity_at(altitude_m)
        
    vel_ecef = vel_eci2ecef(vel_eci, pos_eci, t)
    vel_wind_ned = wind_ned(altitude_m, wind)
    vel_wind_eci = quatrot(quat_nedg2eci(pos_eci, t), vel_wind_ned)
    vel_air_eci = ecef2eci(vel_ecef, t) - vel_wind_eci
    
    return 0.5 * vel_air_eci.dot(vel_air_eci) * rho

    

def cost_6DoF_LG(xdict, condition):
    
    if condition["OptimizationMode"]["Maximize excess propellant mass"]:
        return xdict["t"][-1] #到達時間を最小化(=余剰推進剤を最大化)
    else:
        return -xdict["xvars"][0] #初期質量(無次元)を最大化


def initialize_xdict_6DoF_2(x_init, pdict, condition, unit_xut, mode='LGL', dt=0.005, flag_display=True):
    
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
    xdict["t"] = (time_knots / unit_xut["t"]).ravel()

    # 現在位置と目標軌道から、適当に目標位置・速度を決める
    
    if condition["rf_km"] is not None:
        r_final = condition["rf_km"]
    else:
        if condition["hp_km"] is None or condition["ha_km"] is None:
            print("DESTINATION ORBIT NOT DETERMINED!!")
            sys.exit()
    print(r_final)
    
    u_nodes = np.vstack([[[0.0, pdict["params"][i]["pitchrate_dps"],pdict["params"][i]["yawrate_dps"]]] * pdict["ps_params"][i]["nodes"] for i in range(num_sections)])
    xdict["uvars"] = (u_nodes / unit_xut["u"]).ravel()
        
    u_table = np.hstack((time_nodes.reshape(-1,1),u_nodes))
    
    x_nodes, _ = rocket_simulation(x_init, u_table, pdict, time_nodes[0], time_x_nodes, dt)
    
    xdict["xvars"] = (x_nodes / unit_xut["x"]).ravel()
    
    if flag_display:
        display_6DoF(output_6DoF(x_nodes, u_nodes, time_x_nodes, time_nodes, pdict))
    return xdict
    

def output_6DoF(x_res, u_res, tx_res, tu_res, pdict):
    
    
    
    N = len(tx_res)
    
    
    out = { "time"       : tx_res,
            "section"    : np.zeros(N),
            "thrust"     : np.zeros(N),
            "mass"       : x_res[:,0],
            "lat"        : np.zeros(N),
            "lon"        : np.zeros(N),
            "alt"        : np.zeros(N),
            "ha"         : np.zeros(N),
            "hp"         : np.zeros(N),
            "inc"        : np.zeros(N),
            "argp"       : np.zeros(N),
            "asnd"       : np.zeros(N),
            "tanm"       : np.zeros(N),
            "pos_ECI_X"  : x_res[:,1],
            "pos_ECI_Y"  : x_res[:,2],
            "pos_ECI_Z"  : x_res[:,3],
            "vel_ECI_X"  : x_res[:,4],
            "vel_ECI_Y"  : x_res[:,5],
            "vel_ECI_Z"  : x_res[:,6],
            "vel_NED_X"  : np.zeros(N),
            "vel_NED_Y"  : np.zeros(N),
            "vel_NED_Z"  : np.zeros(N),
            "quat_i2b_0" : x_res[:,7],
            "quat_i2b_1" : x_res[:,8],
            "quat_i2b_2" : x_res[:,9],
            "quat_i2b_3" : x_res[:,10],
            "accel_X"    : np.zeros(N),
            "aero_X"     : np.zeros(N),
            "heading"    : np.zeros(N),
            "pitch"      : np.zeros(N),
            "roll"       : np.zeros(N),
            "vi"         : norm(x_res[:,4:7],axis=1),
            "fpvgd"      : np.zeros(N),
            "azvgd"      : np.zeros(N),
            "thrustvec_X": np.zeros(N),
            "thrustvec_Y": np.zeros(N),
            "thrustvec_Z": np.zeros(N),
            "rate_P"     : np.interp(tx_res, tu_res, u_res[:,0]),
            "rate_Q"     : np.interp(tx_res, tu_res, u_res[:,1]),
            "rate_R"     : np.interp(tx_res, tu_res, u_res[:,2]),
            "vr"         : np.zeros(N),
            "va"         : np.zeros(N),
            "aoa_total"  : np.zeros(N),
            "aoa_alpha"  : np.zeros(N),
            "aoa_beta"   : np.zeros(N),
            "q"          : np.zeros(N),
            "M"          : np.zeros(N)
        }
    
    section = 0

    for i in range(N):
        
        x = x_res[i]
        t = tx_res[i]
        mass = x[0]
        pos = x[1:4]
        vel = x[4:7]
        quat = normalize(x[7:11])
        
        if t >= pdict["params"][section]["timeFinishAt_sec"]:
            section += 1
        thrust_vac_n = pdict["params"][section]["thrust_n"]
        massflow_kgps = pdict["params"][section]["massflow_kgps"]
        airArea_m2 = pdict["params"][section]["airArea_m2"]
        nozzleArea_m2 = pdict["params"][section]["nozzleArea_m2"]
        out["section"][i] = section

        pos_llh = eci2geodetic(pos, t)
        altitude_m = geopotential_altitude(pos_llh[2])
        out["lat"][i]  = pos_llh[0]
        out["lon"][i]  = pos_llh[1]
        out["alt"][i]  = pos_llh[2] / 1000.0
        
        elem = orbital_elements(pos, vel)
        out["ha"][i] = elem[0] * (1.0 + elem[1]) / 1000.0 - 6378.137
        out["hp"][i] = elem[0] * (1.0 - elem[1]) / 1000.0 - 6378.137
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
        
        aoa_all_deg = angle_of_attack_all_rad(x, 0.0, t, pdict["wind_table"]) * 180.0 / np.pi
        aoa_ab_deg = angle_of_attack_ab_rad(x, 0.0, t, pdict["wind_table"]) * 180.0 / np.pi
        
        out["aoa_total"][i] = aoa_all_deg
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
    
  

def display_6DoF(out, flag_savefig=False):
    
    plt.figure()
    plt.title('Altitude[km]')
        
    plt.plot(out["time"], out["alt"], '.-', lw=0.8, label='Altitude')
    plt.plot(out["time"], out["ha"], label='ha')
    plt.plot(out["time"], out["hp"], label='hp')

    plt.ylim([0,None])
    plt.xlim([0,None])
    plt.grid()
    plt.legend()
    if flag_savefig:
        plt.savefig('figures/Altitude.png')
    plt.show()

    plt.figure()
    plt.title('Orbital elements')
    plt.plot(out["time"], out.loc[:,["inc","asnd","argp"]], label=['i', 'Ω', 'ω'])
    plt.xlim([0,None])
    plt.ylim([-180,180])
    plt.grid()
    if flag_savefig:
        plt.savefig('figures/Orbital_Elements.png')
    plt.show()
    
    
    plt.figure()
    plt.title('Ground speed_NED')
    
    plt.plot(out["time"], out.loc[:,["vel_NED_X","vel_NED_Y","vel_NED_Z"]], '.-', lw=0.8, label=["N", "E", "D"])
    plt.xlim([0,None])
    plt.grid()
    if flag_savefig:
        plt.savefig('figures/Ground_Speed.png')
    plt.show()

    plt.figure()
    plt.title('Angle of attack')
    
    plt.plot(out["time"], out.loc[:,["aoa_alpha","aoa_beta"]], '.-', lw=0.8, label=["alpha", "beta"])
    plt.xlim([0,None])
    plt.ylim([-10,10])
    plt.grid()
    if flag_savefig:
        plt.savefig('figures/Angle_of_attack.png')
    plt.show()
    
    plt.figure()
    plt.title('Flight trajectory and velocity vector')
    i = int(len(out["time"]) / 10)
    
    pos_llh = out.loc[:,["lat","lon","alt"]].to_numpy('f8')
    vel_ned = out.loc[:,["vel_NED_X","vel_NED_Y","vel_NED_Z"]].to_numpy('f8')

    x = np.deg2rad(pos_llh[:,1])
    y = np.log(np.tan(np.deg2rad(45+pos_llh[:,0]/2)))
    
    plt.plot(x,y,lw=0.5)
    plt.quiver(x[::i],y[::i],vel_ned[::i,1],vel_ned[::i,0], scale=1.5e5)
    plt.axis('equal')
    plt.grid()
    if flag_savefig:
        plt.savefig('figures/Trajectory.png')
    plt.show()


    plt.figure()
    plt.title('Thrust vector (ECI)')
    
    plt.plot(out["time"], out.loc[:,["thrustvec_X","thrustvec_Y","thrustvec_Z"]], '.-', lw=0.8)
    plt.ylim([-1.0,1.0])
    plt.xlim([0, None])
    plt.grid()
    if flag_savefig:
        plt.savefig('figures/Thrust_vector.png')
    plt.show()
    
    
    plt.figure()
    plt.title('Euler Angle and Velocity Direction')
    
    
    plt.plot(out["time"], out.loc[:,["heading","pitch","roll"]], '.-', lw=0.8, label=["body azimuth", "body elevation", "body roll"])
    plt.plot(out["time"], out["azvgd"], lw=1, label="vel direction")
    plt.plot(out["time"], out["fpvgd"], lw=1, label="vel elevation")
    plt.xlim([0,None])
    plt.ylim([-180,180])
    plt.grid()
    plt.legend()
    if flag_savefig:
        plt.savefig('figures/Euler_angle.png')
    plt.show()
    
def display_3d(out):
    lim = 6378 + 2500

    x_km = out["pos_ECI_X"].to_numpy() / 1000.0
    y_km = out["pos_ECI_Y"].to_numpy() / 1000.0
    z_km = out["pos_ECI_Z"].to_numpy() / 1000.0

    thetas = np.linspace(0, np.pi, 20)
    phis = np.linspace(0, np.pi*2, 20)

    xs = 6378 * np.outer(np.sin(thetas),np.sin(phis))
    ys = 6378 * np.outer(np.sin(thetas),np.cos(phis))
    zs = 6357 * np.outer(np.cos(thetas),np.ones_like(phis))

    plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((1,1,1))

    ax.view_init(elev=15, azim=150)

    ax.plot_wireframe(xs,ys,zs, color='c', lw=0.2)
    ax.plot(x_km,y_km,z_km, color='r')

    ax.plot([0,2000],[0,0],[0,0],color='r',lw=1)
    ax.plot([0,0],[0,2000],[0,0],color='g',lw=1)
    ax.plot([0,0],[0,0],[0,2000],color='b',lw=1)


    ax.set_xlabel('X[km]')
    ax.set_ylabel('Y[km]')
    ax.set_zlabel('Z[km]')