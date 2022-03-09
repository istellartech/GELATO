import numpy as np
from numpy.linalg import norm
from math import sin,cos,tan,asin,acos,atan,atan2,degrees,radians,sqrt
from numba import jit
import pymap3d as pm
from utils import *

@jit('f8[:](f8[:],f8[:])',nopython=True)
def quatmult(q, p):
    qp0 = q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3]
    qp1 = q[1]*p[0] + q[0]*p[1] - q[3]*p[2] + q[2]*p[3]
    qp2 = q[2]*p[0] + q[3]*p[1] + q[0]*p[2] - q[1]*p[3]
    qp3 = q[3]*p[0] - q[2]*p[1] + q[1]*p[2] + q[0]*p[3]
    return np.array([qp0, qp1, qp2, qp3])

@jit('f8[:](f8[:])',nopython=True)
def conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

@jit('f8[:](f8[:])',nopython=True)
def normalize(v):
    return v / norm(v)


@jit('f8[:](f8[:],f8[:])',nopython=True)
def quatrot(q, v):
    vq  = np.array((0.0, v[0], v[1], v[2]))
    rvq = quatmult(conj(q), quatmult(vq, q))
    return rvq[1:4]

@jit(nopython=True)
def dcm_from_quat(q):
    C = np.zeros((3,3))
    C[0,0] = q[0]**2+q[1]**2-q[2]**2-q[3]**2
    C[0,1] = 2.0 * (q[1]*q[2] + q[0]*q[3])
    C[0,2] = 2.0 * (q[1]*q[3] - q[0]*q[2])

    C[1,0] = 2.0 * (q[1]*q[2] - q[0]*q[3])
    C[1,1] = q[0]**2-q[1]**2+q[2]**2-q[3]**2
    C[1,2] = 2.0 * (q[2]*q[3] + q[0]*q[1])

    C[2,0] = 2.0 * (q[1]*q[3] + q[0]*q[2])
    C[2,1] = 2.0 * (q[2]*q[3] - q[0]*q[1])
    C[2,2] = q[0]**2-q[1]**2-q[2]**2+q[3]**2

    return C

@jit(nopython=True)
def quat_from_dcm(C):
    if (1.0 + C[0,0] + C[1,1] + C[2,2]) < 0.0:
        print("quaternion conversion error")
        return np.array((1.0, 0.0, 0.0, 0.0))

    q0 = 0.5 * sqrt(1.0 + C[0,0] + C[1,1] + C[2,2])
    q1 = 0.25 / q0 * (C[1,2] - C[2,1])
    q2 = 0.25 / q0 * (C[2,0] - C[0,2])
    q3 = 0.25 / q0 * (C[0,1] - C[1,0])

    return np.array((q0, q1, q2, q3))   

@jit('f8[:](f8[:],f8)',nopython=True)
def ecef2eci(xyz_in, t):
    omega_earth_rps = 7.2921151467e-5
    xyz_out = np.zeros(3)
    xyz_out[0] = xyz_in[0] * cos(omega_earth_rps * t) - xyz_in[1] * sin(omega_earth_rps * t)
    xyz_out[1] = xyz_in[0] * sin(omega_earth_rps * t) + xyz_in[1] * cos(omega_earth_rps * t)
    xyz_out[2] = xyz_in[2]
    return xyz_out


@jit('f8[:](f8[:],f8)',nopython=True)
def eci2ecef(xyz_in, t):
    omega_earth_rps = 7.2921151467e-5
    xyz_out = np.zeros(3)
    xyz_out[0] = xyz_in[0] * cos(omega_earth_rps * t) + xyz_in[1] * sin(omega_earth_rps * t)
    xyz_out[1] =-xyz_in[0] * sin(omega_earth_rps * t) + xyz_in[1] * cos(omega_earth_rps * t)
    xyz_out[2] = xyz_in[2]
    return xyz_out


@jit('f8[:](f8[:],f8[:],f8)',nopython=True)
def vel_ecef2eci(vel_in, pos_in, t):
    
    omega_earth_rps = 7.2921151467e-5
    pos_eci = ecef2eci(pos_in, t)
    vel_ground_eci = ecef2eci(vel_in, t)
    
    vel_rotation_eci = np.cross(np.array([0, 0, omega_earth_rps]), pos_eci)
    
    return vel_ground_eci + vel_rotation_eci


@jit('f8[:](f8[:],f8[:],f8)',nopython=True)
def vel_eci2ecef(vel_in, pos_in, t):
    
    omega_earth_rps = 7.2921151467e-5
    
    vel_rotation_eci = np.cross(np.array([0, 0, omega_earth_rps]), pos_in)
    vel_ground_eci = vel_in - vel_rotation_eci
    
    return eci2ecef(vel_ground_eci, t)

@jit('f8[:](f8)',nopython=True)
def quat_eci2ecef(t):
    omega_earth_rps = 7.2921151467e-5
    return np.array([cos(omega_earth_rps*t/2.0), 0.0, 0.0, sin(omega_earth_rps*t/2.0)])

@jit('f8[:](f8)',nopython=True)
def quat_ecef2eci(t):
    return conj(quat_eci2ecef(t))

@jit('f8[:](f8[:])',nopython=True)
def quat_ecef2nedc(pos_ecef):
    l = atan2(pos_ecef[1], pos_ecef[0])
    p = asin(pos_ecef[2] / norm(pos_ecef))
    c_hl = cos(l/2.0)
    s_hl = sin(l/2.0)
    c_hp = cos(p/2.0)
    s_hp = sin(p/2.0)
    
    quat_ecef2ned = np.zeros(4)
    quat_ecef2ned[0] = c_hl * (c_hp - s_hp) / sqrt(2.0)
    quat_ecef2ned[1] = s_hl * (c_hp + s_hp) / sqrt(2.0)
    quat_ecef2ned[2] =-c_hl * (c_hp + s_hp) / sqrt(2.0)
    quat_ecef2ned[3] = s_hl * (c_hp - s_hp) / sqrt(2.0)
    
    return quat_ecef2ned


@jit('f8[:](f8[:])',nopython=True)
def quat_ecef2nedg(pos_ecef):
    
    p,l,_ = ecef2geodetic(pos_ecef[0], pos_ecef[1], pos_ecef[2])
    p = radians(p)
    l = radians(l)
    
    c_hl = cos(l/2.0)
    s_hl = sin(l/2.0)
    c_hp = cos(p/2.0)
    s_hp = sin(p/2.0)
    
    quat_ecef2ned = np.zeros(4)
    quat_ecef2ned[0] = c_hl * (c_hp - s_hp) / sqrt(2.0)
    quat_ecef2ned[1] = s_hl * (c_hp + s_hp) / sqrt(2.0)
    quat_ecef2ned[2] =-c_hl * (c_hp + s_hp) / sqrt(2.0)
    quat_ecef2ned[3] = s_hl * (c_hp - s_hp) / sqrt(2.0)
    
    return quat_ecef2ned

@jit('f8[:](f8[:])',nopython=True)
def quat_nedg2ecef(pos_ecef):
    return conj(quat_ecef2nedg(pos_ecef))

@jit('f8[:](f8[:])',nopython=True)
def quat_nedc2ecef(pos_ecef):
    return conj(quat_ecef2nedc(pos_ecef))


@jit('f8[:](f8[:],f8)',nopython=True)
def quat_eci2nedg(pos, t):
    return quatmult(quat_eci2ecef(t), quat_ecef2nedg(eci2ecef(pos,t)))

@jit('f8[:](f8[:],f8)',nopython=True)
def quat_eci2nedc(pos, t):
    return quatmult(quat_eci2ecef(t), quat_ecef2nedc(eci2ecef(pos,t)))


@jit('f8[:](f8[:],f8)',nopython=True)
def quat_nedg2eci(pos, t):
    return conj(quat_eci2nedg(pos, t))

@jit('f8[:](f8[:],f8)',nopython=True)
def quat_nedc2eci(pos, t):
    return conj(quat_eci2nedc(pos, t))


@jit('f8[:](f8[:],f8[:])',nopython=True)
def quat_eci2guide(pos, vel):
    
    z_dir = normalize(-pos)
    x_dir = normalize(np.cross(vel, pos))
    y_dir = np.cross(z_dir,x_dir)
    
    q0 = 0.5 * sqrt(1.0 + x_dir[0] + y_dir[1] + z_dir[2])
    q1 = 0.25 / q0 * (y_dir[2] - z_dir[1])
    q2 = 0.25 / q0 * (z_dir[0] - x_dir[2])
    q3 = 0.25 / q0 * (x_dir[1] - y_dir[0])
    
    return normalize(np.array((q0, q1, q2, q3)))


@jit('f8[:](f8[:],f8[:],f8[:])',nopython=True)
def quat_guide2body(pos, vel, quat_eci2body):
    return quatmult(conj(quat_eci2guide(pos, vel)), quat_eci2body)



@jit('f8[:](f8,f8,f8)',nopython=True)
def quat_from_euler(az, el, ro):
    qz = np.array([cos(radians(az/2)), 0.0, 0.0, sin(radians(az/2))])
    qy = np.array([cos(radians(el/2)), 0.0, sin(radians(el/2)), 0.0])
    qx = np.array([cos(radians(ro/2)), sin(radians(ro/2)), 0.0, 0.0])

    return quatmult(qz, quatmult(qy, qx))

@jit(nopython=True)
def gravity(pos):
    x,y,z = pos
    
    a = 6378137
    f = 1.0 / 298.257223563
    mu = 3.986004418e14
    J2 = 1.082628e-3
    
    r = norm(pos)
    p2 = x**2+y**2
    
    fx = mu * (-x / r**3 + J2 * a**2 * x / r**7 * (6.0 * z**2 - 1.5 * p2))
    fy = mu * (-y / r**3 + J2 * a**2 * y / r**7 * (6.0 * z**2 - 1.5 * p2))
    fz = mu * (-z / r**3 + J2 * a**2 * z / r**7 * (3.0 * z**2 - 4.5 * p2))

    return np.array([fx, fy, fz])

@jit('f8[:](f8[:],f8[:],f8)',nopython=True)
def quat_nedg2body(quat_eci2body, pos, t):
    q = quat_eci2nedg(pos, t)
    return quatmult(conj(q), quat_eci2body)

@jit('f8[:](f8[:])',nopython=True)
def euler_from_quat(q):
    if 2.0*(q[0]*q[2]-q[3]*q[1]) >= 1.0:
        el = np.pi/2
        az = 0.0
        ro = 0.0
    else:
        az = atan2(2.0 * (q[0]*q[3]+q[1]*q[2]), 1.0-2.0*(q[2]**2+q[3]**2))
        el =  asin(2.0 * (q[0]*q[2]-q[3]*q[1]))
        ro = atan2(2.0 * (q[0]*q[1]+q[2]*q[3]), 1.0-2.0*(q[1]**2+q[2]**2))
    return np.rad2deg(np.array([az, el, ro]))

@jit(nopython=True)
def euler_from_dcm(C):
    el = asin(-C[0,2])
    if cos(el) < 0.0001:
        az = 0.0
        ro = 0.0
    else:
        az = atan2(C[0,1], C[0,0])
        ro = atan2(C[1,2], C[2,2])
    return np.rad2deg(np.array([az, el, ro]))

@jit(nopython=True)
def dcm_from_thrustvector(pos_eci, u):

    xb_dir = normalize(u)
    if u.dot(normalize(pos_eci)) > 0.999999:
        yb_dir = normalize(np.cross(np.array([0.0, 0.0, 1.0]), u))
    else:
        yb_dir = normalize(np.cross(u, normalize(pos_eci)))
    zb_dir = np.cross(u, yb_dir)

    return np.vstack((xb_dir, yb_dir, zb_dir))  

def eci2geodetic(pos_in, t):
    pos_ecef = eci2ecef(pos_in, t)
    return pm.ecef2geodetic(pos_ecef[0],pos_ecef[1],pos_ecef[2])


@jit('f8[:](f8[:],f8[:])', nopython=True)
def orbital_elements(r_eci, v_eci):
    GMe = 3.986004418e14
    
    nr = normalize(r_eci)
    v_r = nr.dot(v_eci)
    v_n = norm(v_eci - v_r * normalize(nr))
    
    c_eci = np.cross(r_eci, v_eci) # orbit plane vector
    f_eci = np.cross(v_eci, c_eci) - GMe * normalize(r_eci) # Laplace vector
    
    c1_eci = normalize(c_eci)
    
    inclination_rad = acos(c1_eci.dot(np.array([0.0,0.0,1.0])))
    
    if inclination_rad > 1e-10:
        ascending_node_rad = atan2(c1_eci[0], -c1_eci[1])
        n_eci = np.array([cos(ascending_node_rad), sin(ascending_node_rad), 0.0]) # direction of ascending node
        argument_perigee = acos(n_eci.dot(normalize(f_eci)))
        if f_eci[2] < 0:
            argument_perigee *= -1.0
    else:
        ascending_node_rad = 0.0
        argument_perigee = atan2(f_eci[1], f_eci[0])
    
    p = norm(c_eci)**2 / GMe # semi-latus rectum
    e = norm(f_eci) / GMe # eccentricity
    a = p / (1.0 - e**2) # semi-major axis
    
    true_anomaly_rad = acos(normalize(f_eci).dot(normalize(r_eci)))
    if v_eci.dot(r_eci) < 0.0:
        true_anomaly_rad = 2.0*np.pi - true_anomaly_rad
        
    return np.array([a, e, degrees(inclination_rad), degrees(ascending_node_rad), degrees(argument_perigee), degrees(true_anomaly_rad)])

@jit(nopython=True)
def posvel_from_orbital_elements(elem):
    
    a = elem[0]
    e = elem[1]
    inclination_rad = radians(elem[2])
    ascending_node_rad = radians(elem[3])
    argument_perigee_rad = radians(elem[4])
    true_anomaly_rad = radians(elem[5])
    
    