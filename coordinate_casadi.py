#
# Coordinate transforms and orbital mechanics — CasADi symbolic versions.
#
# All functions accept and return CasADi MX/SX expressions so that
# CasADi can compute exact derivatives via automatic differentiation.
#

import numpy as np
import casadi as ca

# --- Constants ---
OMEGA_EARTH = 7.2921159e-5  # Earth rotation rate [rad/s]
MU_EARTH = 3.986004418e14  # gravitational parameter [m^3/s^2]
R_EARTH_A = 6378137.0  # WGS84 semi-major axis [m]
R_EARTH_B = 6356752.314245  # WGS84 semi-minor axis [m]
E2 = 1.0 - (R_EARTH_B / R_EARTH_A) ** 2  # first eccentricity squared
EP2 = (R_EARTH_A / R_EARTH_B) ** 2 - 1.0  # second eccentricity squared


# ================================================================
# Quaternion operations  (q = [w, x, y, z])
# ================================================================


def quatmult(q1, q2):
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return ca.vertcat(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def conj(q):
    """Quaternion conjugate."""
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])


def quatrot(q, v):
    """Rotate a vector with coordinate transformation quaternion.
    Calculates conj(q) * [0,v] * q.  Given a quaternion q that represents
    transformation from A-frame to B-frame, this returns the representation
    of vector v (given in A-frame) in B-frame."""
    v_q = ca.vertcat(0, v[0], v[1], v[2])
    r = quatmult(conj(q), quatmult(v_q, q))
    return r[1:4]


# ================================================================
# ECI / ECEF conversions
# ================================================================


def eci2ecef(pos_eci, t):
    """Rotate ECI position to ECEF at time t [s]."""
    theta = OMEGA_EARTH * t
    c, s = ca.cos(theta), ca.sin(theta)
    return ca.vertcat(
        c * pos_eci[0] + s * pos_eci[1],
        -s * pos_eci[0] + c * pos_eci[1],
        pos_eci[2],
    )


def ecef2eci(pos_ecef, t):
    """Rotate ECEF position to ECI at time t [s]."""
    theta = OMEGA_EARTH * t
    c, s = ca.cos(theta), ca.sin(theta)
    return ca.vertcat(
        c * pos_ecef[0] - s * pos_ecef[1],
        s * pos_ecef[0] + c * pos_ecef[1],
        pos_ecef[2],
    )


def vel_eci2ecef(vel_eci, pos_eci, t):
    """Convert ECI velocity to ECEF velocity."""
    # v_ecef = R(t) * (v_eci - omega x r_eci)
    omega_cross = ca.vertcat(
        -OMEGA_EARTH * pos_eci[1],
        OMEGA_EARTH * pos_eci[0],
        0,
    )
    v_rel = vel_eci - omega_cross
    return eci2ecef(v_rel, t)


# ================================================================
# Geodetic conversions  (Bowring single-iteration, accurate to ~1 mm)
# ================================================================


def ecef2geodetic(pos_ecef):
    """ECEF → (lat_rad, lon_rad, alt_m).  Bowring's method."""
    x, y, z = pos_ecef[0], pos_ecef[1], pos_ecef[2]
    p = ca.sqrt(x**2 + y**2)
    theta = ca.atan2(z * R_EARTH_A, p * R_EARTH_B)

    lat = ca.atan2(
        z + EP2 * R_EARTH_B * ca.sin(theta) ** 3,
        p - E2 * R_EARTH_A * ca.cos(theta) ** 3,
    )
    lon = ca.atan2(y, x)

    sin_lat = ca.sin(lat)
    cos_lat = ca.cos(lat)
    N = R_EARTH_A / ca.sqrt(1 - E2 * sin_lat**2)
    alt = p / cos_lat - N

    return lat, lon, alt


def eci2geodetic(pos_eci, t):
    """ECI → (lat_deg, lon_deg, alt_m)."""
    pos_ecef = eci2ecef(pos_eci, t)
    lat_r, lon_r, alt = ecef2geodetic(pos_ecef)
    return ca.vertcat(lat_r * 180 / ca.pi, lon_r * 180 / ca.pi, alt)


def geodetic2ecef_np(lat_deg, lon_deg, alt_m):
    """Geodetic → ECEF (NumPy, for constant values)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    N = R_EARTH_A / np.sqrt(1 - E2 * sin_lat**2)
    x = (N + alt_m) * cos_lat * np.cos(lon)
    y = (N + alt_m) * cos_lat * np.sin(lon)
    z = (N * (1 - E2) + alt_m) * sin_lat
    return np.array([x, y, z])


# ================================================================
# NED frame helpers
# ================================================================


def ned_basis_ecef(lat_rad, lon_rad):
    """North and East unit vectors in ECEF, given geodetic lat/lon in radians."""
    sl, cl = ca.sin(lat_rad), ca.cos(lat_rad)
    sn, cn = ca.sin(lon_rad), ca.cos(lon_rad)
    north = ca.vertcat(-sl * cn, -sl * sn, cl)
    east = ca.vertcat(-sn, cn, 0)
    return north, east


# ================================================================
# Gravity
# ================================================================


def gravity_eci(pos_eci):
    """Gravitational acceleration in ECI [m/s^2].
    Uses JGM-3 geopotential model with J2 zonal harmonic."""
    _J2 = 1.082628e-3
    x, y, z = pos_eci[0], pos_eci[1], pos_eci[2]
    r = ca.norm_2(pos_eci)
    p2 = x**2 + y**2
    fx = MU_EARTH * (
        -x / r**3 + _J2 * R_EARTH_A**2 * x / r**7 * (6.0 * z**2 - 1.5 * p2)
    )
    fy = MU_EARTH * (
        -y / r**3 + _J2 * R_EARTH_A**2 * y / r**7 * (6.0 * z**2 - 1.5 * p2)
    )
    fz = MU_EARTH * (
        -z / r**3 + _J2 * R_EARTH_A**2 * z / r**7 * (3.0 * z**2 - 4.5 * p2)
    )
    return ca.vertcat(fx, fy, fz)


# ================================================================
# Orbital mechanics
# ================================================================


def angular_momentum(pos, vel):
    """Specific angular momentum magnitude [m^2/s]."""
    h = ca.cross(pos, vel)
    return ca.norm_2(h)


def orbit_energy(pos, vel):
    """Specific orbital energy [m^2/s^2]."""
    return 0.5 * ca.dot(vel, vel) - MU_EARTH / ca.norm_2(pos)


def inclination_rad(pos, vel):
    """Orbital inclination [rad]."""
    h = ca.cross(pos, vel)
    cos_inc = ca.fmax(-1.0, ca.fmin(1.0, h[2] / ca.norm_2(h)))
    return ca.acos(cos_inc)


def angular_momentum_from_altitude(alt_perigee, alt_apogee):
    """Target angular momentum from perigee/apogee altitudes [m]."""
    r_p = R_EARTH_A + alt_perigee
    r_a = R_EARTH_A + alt_apogee
    a = (r_p + r_a) / 2.0
    p = a * (1.0 - ((r_a - r_p) / (r_a + r_p)) ** 2)
    return np.sqrt(MU_EARTH * p)


def orbit_energy_from_altitude(alt_perigee, alt_apogee):
    """Target specific energy from perigee/apogee altitudes [m]."""
    r_p = R_EARTH_A + alt_perigee
    r_a = R_EARTH_A + alt_apogee
    a = (r_p + r_a) / 2.0
    return -MU_EARTH / (2.0 * a)
