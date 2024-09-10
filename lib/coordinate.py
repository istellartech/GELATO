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

from math import sin, cos, asin, acos, atan2, degrees, radians, sqrt
import numpy as np
from numpy.linalg import norm


def quatmult(q, p):
    """Multiplies two quaternions."""
    qp0 = q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3]
    qp1 = q[1] * p[0] + q[0] * p[1] - q[3] * p[2] + q[2] * p[3]
    qp2 = q[2] * p[0] + q[3] * p[1] + q[0] * p[2] - q[1] * p[3]
    qp3 = q[3] * p[0] - q[2] * p[1] + q[1] * p[2] + q[0] * p[3]
    return np.array([qp0, qp1, qp2, qp3])


def conj(q):
    """Returns conjugate of quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def normalize(v):
    """Normalizes given vector.
    Args:
        v (ndarray) : The input vector.
    Returns:
        ndarray : The normalized vector.
    """
    return v / norm(v)


def quatrot(q, v):
    """Rotates a vector with coordinate transformation quaternion.
    This function calculates conj(q) * v * q, where the sign * means
    quaternion product.
    Args:
        q (ndarray) : The quaternion that represents transformation
        from A-frame to B-frame.
        v (ndarray) : The representation of vector in A-frame.
    Returns:
        ndarray : The representation of the input vector in B-frame.
    """
    vq = np.array((0.0, v[0], v[1], v[2]))
    rvq = quatmult(conj(q), quatmult(vq, q))
    return rvq[1:4]


def dcm_from_quat(q):
    """Converts a quaternion to a direction cosine matrix (DCM)."""
    C = np.zeros((3, 3))
    C[0, 0] = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    C[0, 1] = 2.0 * (q[1] * q[2] + q[0] * q[3])
    C[0, 2] = 2.0 * (q[1] * q[3] - q[0] * q[2])

    C[1, 0] = 2.0 * (q[1] * q[2] - q[0] * q[3])
    C[1, 1] = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    C[1, 2] = 2.0 * (q[2] * q[3] + q[0] * q[1])

    C[2, 0] = 2.0 * (q[1] * q[3] + q[0] * q[2])
    C[2, 1] = 2.0 * (q[2] * q[3] - q[0] * q[1])
    C[2, 2] = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2

    return C


def quat_from_dcm(C):
    """Converts a direction cosine matrix (DCM) into a quaternion."""
    if (1.0 + C[0, 0] + C[1, 1] + C[2, 2]) < 0.0:
        print("quaternion conversion error")
        return np.array((1.0, 0.0, 0.0, 0.0))

    q0 = 0.5 * sqrt(1.0 + C[0, 0] + C[1, 1] + C[2, 2])
    q1 = 0.25 / q0 * (C[1, 2] - C[2, 1])
    q2 = 0.25 / q0 * (C[2, 0] - C[0, 2])
    q3 = 0.25 / q0 * (C[0, 1] - C[1, 0])

    return np.array((q0, q1, q2, q3))


def ecef2geodetic(x, y, z):
    """Converts a position in ECEF frame into a geodetic coordinates.
    Args:
        x (float64) : X-coordinate of the position [m]
        y (float64) : Y-coordinate of the position [m]
        z (float64) : Z-coordinate of the position [m]
    Returns:
        ndarray : The combination of latitude [deg], longitude [deg]
        and altitude [m] of the input position in WGS84.
    """

    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = (a**2 - b**2) / a**2
    ep2 = (a**2 - b**2) / b**2

    p = sqrt(x**2 + y**2)
    theta = atan2(z * a, p * b)

    lat = atan2(z + ep2 * b * sin(theta) ** 3, p - e2 * a * cos(theta) ** 3)
    lon = atan2(y, x)
    N = a / sqrt(1.0 - e2 * sin(lat) ** 2)
    alt = p / cos(lat) - N

    return np.array((degrees(lat), degrees(lon), alt))


def geodetic2ecef(lat, lon, alt):
    """Converts a position in geodetic coordinates into an ECEF frame.
    Args:
        lat (float64) : latitude of the position [deg]
        lon (float64) : longitude of the position [deg]
        alt (float64) : geometric altitude of the position [m]
    Returns:
        ndarray : The position vector of the input position in
        ECEF frame.
    """

    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = (a**2 - b**2) / a**2

    N = a / sqrt(1.0 - e2 * sin(radians(lat)) ** 2)

    x = (N + alt) * cos(radians(lat)) * cos(radians(lon))
    y = (N + alt) * cos(radians(lat)) * sin(radians(lon))
    z = (N * (1 - e2) + alt) * sin(radians(lat))

    return np.array((x, y, z))


def ecef2geodetic_sphere(x, y, z):
    """Converts a position in ECEF frame into a spherical coordinates.
    This function is DEPRECATED.
    Args:
        x (float64) : X-coordinate of the position [m]
        y (float64) : Y-coordinate of the position [m]
        z (float64) : Z-coordinate of the position [m]
    Returns:
        ndarray : The combination of latitude [deg], longitude [deg]
        and altitude [m] of the input position in spherical
        coordinates.
    """
    r_Earth = 6378137.0
    lat = degrees(atan2(z, sqrt(x**2 + y**2)))
    lon = degrees(atan2(y, x))
    alt = sqrt(x**2 + y**2 + z**2) - r_Earth
    return np.array((lat, lon, alt))


def geodetic2ecef_sphere(lat, lon, alt):
    """Converts a position in spherical coordinates into an ECEF frame.
    This function is DEPRECATED.
    Args:
        lat (float64) : latitude of the position [deg]
        lon (float64) : longitude of the position [deg]
        alt (float64) : geometric altitude of the position [m]
    Returns:
        ndarray : The position vector of the input position in ECEF
        frame.
    """

    r_Earth = 6378137.0
    z = (alt + r_Earth) * sin(radians(lat))
    y = (alt + r_Earth) * cos(radians(lat)) * sin(radians(lon))
    x = (alt + r_Earth) * cos(radians(lat)) * cos(radians(lon))
    return np.array((x, y, z))


def ecef2eci(xyz_in, t):
    """Converts a position in ECEF coordinates into an ECI frame.
    Args:
        xyz_int (ndarray) : position vector in the ECEF frame
        t (float64) : time from the epoch (the time when the ECEF
        frame and the ECI frame coincides)
    Returns:
        ndarray : The position vector of the input position in the
        ECI frame.
    """

    omega_earth_rps = 7.2921151467e-5
    xyz_out = np.zeros(3)
    xyz_out[0] = xyz_in[0] * cos(omega_earth_rps * t) - xyz_in[1] * sin(
        omega_earth_rps * t
    )
    xyz_out[1] = xyz_in[0] * sin(omega_earth_rps * t) + xyz_in[1] * cos(
        omega_earth_rps * t
    )
    xyz_out[2] = xyz_in[2]
    return xyz_out


def eci2ecef(xyz_in, t):
    """Converts a position in ECI coordinates into an ECEF frame.
    Args:
        xyz_int (ndarray) : position vector in the ECI frame
        t (float64) : time from the epoch (the time when the ECEF
        frame and the ECI frame coincides)
    Returns:
        ndarray : The position vector of the input position in the
        ECEF frame.
    """

    omega_earth_rps = 7.2921151467e-5
    xyz_out = np.zeros(3)
    xyz_out[0] = xyz_in[0] * cos(omega_earth_rps * t) + xyz_in[1] * sin(
        omega_earth_rps * t
    )
    xyz_out[1] = -xyz_in[0] * sin(omega_earth_rps * t) + xyz_in[1] * cos(
        omega_earth_rps * t
    )
    xyz_out[2] = xyz_in[2]
    return xyz_out


def vel_ecef2eci(vel_in, pos_in, t):
    """Converts a ground velocity vector in ECEF coordinates into
    an inertial velocity vector in ECI frame.
    Args:
        vel_in (ndarray) : ground velocity vector in the ECEF frame
        pos_in (ndarray) : position vector in the ECEF frame
        t (float64) : time from the epoch (the time when the ECEF
        frame and the ECI frame coincides)
    Returns:
        ndarray : The inertial velocity vector in the ECI frame.
    """

    omega_earth_rps = 7.2921151467e-5
    pos_eci = ecef2eci(pos_in, t)
    vel_ground_eci = ecef2eci(vel_in, t)

    vel_rotation_eci = np.cross(np.array([0, 0, omega_earth_rps]), pos_eci)

    return vel_ground_eci + vel_rotation_eci


def vel_eci2ecef(vel_in, pos_in, t):
    """Converts an inertial velocity vector in ECI coordinates into
    a ground velocity vector in ECEF frame.
    Args:
        vel_in (ndarray) : inertial velocity vector in the ECI frame
        pos_in (ndarray) : position vector in the ECI frame
        t (float64) : time from the epoch (the time when the ECEF
        frame and the ECI frame coincides)
    Returns:
        ndarray : The ground velocity vector in the ECEF frame.
    """

    omega_earth_rps = 7.2921151467e-5

    vel_rotation_eci = np.cross(np.array([0, 0, omega_earth_rps]), pos_in)
    vel_ground_eci = vel_in - vel_rotation_eci

    return eci2ecef(vel_ground_eci, t)


def quat_eci2ecef(t):
    """Returns coordinates transformation quaternion from the ECI
    frame to the ECEF frame.
    Args:
        t (float64) : time from the epoch (the time when the ECEF
        frame and the ECI frame coincides)
    Returns:
        ndarray : The coordinates transformation quaternion.
    """

    omega_earth_rps = 7.2921151467e-5
    return np.array(
        [cos(omega_earth_rps * t / 2.0), 0.0, 0.0, sin(omega_earth_rps * t / 2.0)]
    )


def quat_ecef2eci(t):
    """Returns coordinates transformation quaternion from the ECEF
    frame to the ECI frame.
    Args:
        t (float64) : time from the epoch (the time when the ECEF
        frame and the ECI frame coincides)
    Returns:
        ndarray : The coordinates transformation quaternion.
    """

    return conj(quat_eci2ecef(t))


def quat_ecef2nedc(pos_ecef):
    """(DEPRECATED) Returns coordinates transformation quaternion from
    the ECEF frame to the local spherical North-East-Down frame.
    Args:
        pos_ecef (ndarray) : position in the ECEF frame
    Returns:
        ndarray : The coordinates transformation quaternion.
    """

    l = atan2(pos_ecef[1], pos_ecef[0])
    p = asin(pos_ecef[2] / norm(pos_ecef))
    c_hl = cos(l / 2.0)
    s_hl = sin(l / 2.0)
    c_hp = cos(p / 2.0)
    s_hp = sin(p / 2.0)

    quat_ecef2ned = np.zeros(4)
    quat_ecef2ned[0] = c_hl * (c_hp - s_hp) / sqrt(2.0)
    quat_ecef2ned[1] = s_hl * (c_hp + s_hp) / sqrt(2.0)
    quat_ecef2ned[2] = -c_hl * (c_hp + s_hp) / sqrt(2.0)
    quat_ecef2ned[3] = s_hl * (c_hp - s_hp) / sqrt(2.0)

    return quat_ecef2ned


def quat_ecef2nedg(pos_ecef):
    """Returns coordinates transformation quaternion from the ECEF
    frame to the WGS84 local North-East-Down frame.
    Args:
        pos_ecef (ndarray) : position in the ECEF frame
    Returns:
        ndarray : The coordinates transformation quaternion.
    """

    p, l, _ = ecef2geodetic(pos_ecef[0], pos_ecef[1], pos_ecef[2])
    p = radians(p)
    l = radians(l)

    c_hl = cos(l / 2.0)
    s_hl = sin(l / 2.0)
    c_hp = cos(p / 2.0)
    s_hp = sin(p / 2.0)

    quat_ecef2ned = np.zeros(4)
    quat_ecef2ned[0] = c_hl * (c_hp - s_hp) / sqrt(2.0)
    quat_ecef2ned[1] = s_hl * (c_hp + s_hp) / sqrt(2.0)
    quat_ecef2ned[2] = -c_hl * (c_hp + s_hp) / sqrt(2.0)
    quat_ecef2ned[3] = s_hl * (c_hp - s_hp) / sqrt(2.0)

    return quat_ecef2ned


def quat_nedg2ecef(pos_ecef):
    """Returns coordinates transformation quaternion from the WGS84
    local North-East-Down frame to the ECEF frame.
    Args:
        pos_ecef (ndarray) : position in the ECEF frame
    Returns:
        ndarray : The coordinates transformation quaternion.
    """

    return conj(quat_ecef2nedg(pos_ecef))


def quat_nedc2ecef(pos_ecef):
    """(DEPRECATED) Returns coordinates transformation quaternion
    from the local spherical North-East-Down frame to the ECEF frame.
    Args:
        pos_ecef (ndarray) : position in the ECEF frame
    Returns:
        ndarray : The coordinates transformation quaternion.
    """

    return conj(quat_ecef2nedc(pos_ecef))


def quat_eci2nedg(pos, t):
    """Returns coordinates transformation quaternion from the ECI
    frame to the WGS84 local North-East-Down frame.
    Args:
        pos (ndarray) : position in the ECI frame
        t (float64) : time from the epoch (the time when the ECEF
        frame and the ECI frame coincides)
    Returns:
        ndarray : The coordinates transformation quaternion.
    """

    return quatmult(quat_eci2ecef(t), quat_ecef2nedg(eci2ecef(pos, t)))


def quat_eci2nedc(pos, t):
    """(DEPRECATED) Returns coordinates transformation quaternion
    from the ECI frame to the local spherical North-East-Down frame.
    Args:
        pos (ndarray) : position in the ECI frame
        t (float64) : time from the epoch (the time when the ECEF
        frame and the ECI frame coincides)
    Returns:
        ndarray : The coordinates transformation quaternion.
    """

    return quatmult(quat_eci2ecef(t), quat_ecef2nedc(eci2ecef(pos, t)))


def quat_nedg2eci(pos, t):
    """Returns coordinates transformation quaternion from the WGS84
    local North-East-Down frame to the ECI frame.
    Args:
        pos (ndarray) : position in the ECI frame
        t (float64) : time from the epoch (the time when the ECEF
        frame and the ECI frame coincides)
    Returns:
        ndarray : The coordinates transformation quaternion.
    """

    return conj(quat_eci2nedg(pos, t))


def quat_nedc2eci(pos, t):
    """(DEPRECATED) Returns coordinates transformation quaternion from
    the local spherical North-East-Down frame to the ECI frame.
    Args:
        pos (ndarray) : position in the ECI frame
        t (float64) : time from the epoch (the time when the ECEF frame
        and the ECI frame coincides)
    Returns:
        ndarray : The coordinates transformation quaternion.
    """

    return conj(quat_eci2nedc(pos, t))


def quat_from_euler(az, el, ro):
    """Converts Euler angles into a coordinate transformation
    quaternion.
    The sequence of rotation is Z-Y-X (yaw-pitch-roll).
    Args:
        az (float64) : yaw angle [deg]
        el (float64) : pitch angle [deg]
        ro (float64) : roll angle [deg]
    Returns:
        ndarray : The coordinates transformation quaternion.
    """
    qz = np.array([cos(radians(az / 2)), 0.0, 0.0, sin(radians(az / 2))])
    qy = np.array([cos(radians(el / 2)), 0.0, sin(radians(el / 2)), 0.0])
    qx = np.array([cos(radians(ro / 2)), sin(radians(ro / 2)), 0.0, 0.0])

    return quatmult(qz, quatmult(qy, qx))


def gravity(pos):
    """Calculates gravity acceleration vector at the given position.
    This function uses JGM-3 geopotential model.
    J2 factor is considered and higher zonal coefficients and the all
    tesseral coefficients are ignored.
    Args:
        pos (ndarray) : position in the ECI or ECEF frame [m]
    Returns:
        ndarray: The gravity acceleration at the given point in the same
        frame as the input position [m/s2]
    """
    x, y, z = pos

    a = 6378137
    f = 1.0 / 298.257223563
    mu = 3.986004418e14
    J2 = 1.082628e-3

    r = norm(pos)
    p2 = x**2 + y**2

    fx = mu * (-x / r**3 + J2 * a**2 * x / r**7 * (6.0 * z**2 - 1.5 * p2))
    fy = mu * (-y / r**3 + J2 * a**2 * y / r**7 * (6.0 * z**2 - 1.5 * p2))
    fz = mu * (-z / r**3 + J2 * a**2 * z / r**7 * (3.0 * z**2 - 4.5 * p2))

    return np.array([fx, fy, fz])


def quat_nedg2body(quat_eci2body, pos, t):
    """Returns coordinates transformation quaternion from the WGS84
    local North-East-Down frame to the body frame.
    Args:
        quat_eci2body : coordinates transformation quaternion from the
        ECI frame to the body frame
        pos (ndarray) : position in the ECI frame
        t (float64) : time from the epoch (the time when the ECEF
        frame and the ECI frame coincides)
    Returns:
        ndarray : The coordinates transformation quaternion.
    """

    q = quat_eci2nedg(pos, t)
    return quatmult(conj(q), quat_eci2body)


def euler_from_quat(q):
    """Calculates Euler angles from a coordinate transformation
    quaternion.
    The sequence of rotation is Z-Y-X (yaw-pitch-roll).
    Args:
        q (ndarray) : the coordinates transformation quaternion
    Returns:
        ndarray : yaw-pitch-roll Euler angles [deg]
    """
    if 2.0 * (q[0] * q[2] - q[3] * q[1]) >= 1.0:
        el = np.pi / 2
        az = 0.0
        ro = 0.0
    else:
        az = atan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
        )
        el = asin(2.0 * (q[0] * q[2] - q[3] * q[1]))
        ro = atan2(
            2.0 * (q[0] * q[1] + q[2] * q[3]), 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
        )
    if az < 0.0:
        az += 2.0 * np.pi
    return np.rad2deg(np.array([az, el, ro]))


def euler_from_dcm(C):
    """Calculates Euler angles from a direction cosine matrix.
    The sequence of rotation is Z-Y-X (yaw-pitch-roll).
    Args:
        C (ndarray) : the direction cosine matrix
    Returns:
        ndarray : yaw-pitch-roll Euler angles [deg]
    """
    el = asin(-C[0, 2])
    if cos(el) < 0.0001:
        az = 0.0
        ro = 0.0
    else:
        az = atan2(C[0, 1], C[0, 0])
        ro = atan2(C[1, 2], C[2, 2])
    if az < 0.0:
        az += 2.0 * np.pi
    return np.rad2deg(np.array([az, el, ro]))


def dcm_from_thrustvector(pos_eci, u):
    """Calculates direction cosine matrix(DCM) from the position
    and the direction of body axis.
    The roll angle is assumed to be 0 degree.
    Args:
        pos_eci (ndarray) : the position in the ECI frame
        u (ndarray) : the direction of body axis in the ECI frame
    Returns:
        ndarray : the DCM from the ECI frame to the body frame
    """

    xb_dir = normalize(u)
    pos_dir = normalize(pos_eci)
    if u[0] * pos_dir[0] + u[1] * pos_dir[1] + u[2] * pos_dir[2] >= 1.0:
        yb_dir = normalize(np.cross(np.array([0.0, 0.0, 1.0]), u))
    else:
        yb_dir = normalize(np.cross(u, pos_dir))
    zb_dir = np.cross(u, yb_dir)

    return np.vstack((xb_dir, yb_dir, zb_dir))


def eci2geodetic(pos_in, t):
    """Converts a position in ECI frame into a WGS84 geodetic(latitude,
    longitude, altitude) coordinates.
    Args:
        x (float64) : X-coordinate of the position [m]
        y (float64) : Y-coordinate of the position [m]
        z (float64) : Z-coordinate of the position [m]
        t (float64) : time from the epoch (the time when the ECEF
        frame and the ECI frame coincides)
    Returns:
        ndarray : The combination of latitude [deg], longitude [deg]
        and altitude [m] of the input position in WGS84.
    """

    pos_ecef = eci2ecef(pos_in, t)
    return ecef2geodetic(pos_ecef[0], pos_ecef[1], pos_ecef[2])


def orbital_elements(r_eci, v_eci):
    """Calculates orbital elements from position and velocity vectors.
    Args:
        r_eci (float64) : position vector in ECI frame [m]
        v_eci (float64) : inertial velocity vector in ECI frame [m]
    Returns:
        ndarray : orbital elements (semi-major axis[m], eccentricity,
        inclination[deg], longitude of ascending node[deg],
        argument of perigee[deg], true anomaly[deg]).
    """
    GMe = 3.986004418e14

    nr = normalize(r_eci)

    c_eci = np.cross(r_eci, v_eci)  # orbit plane vector
    f_eci = np.cross(v_eci, c_eci) - GMe * nr  # Laplace vector

    c1_eci = normalize(c_eci)
    f1_eci = normalize(f_eci)

    inclination_rad = acos(c1_eci[2])

    if inclination_rad > 1e-10:
        ascending_node_rad = atan2(c1_eci[0], -c1_eci[1])
        n_eci = np.array(
            [cos(ascending_node_rad), sin(ascending_node_rad), 0.0]
        )  # direction of ascending node
        argument_perigee = acos(n_eci[0] * f1_eci[0] + n_eci[1] * f1_eci[1])
        if f_eci[2] < 0:
            argument_perigee *= -1.0
    else:
        ascending_node_rad = 0.0
        argument_perigee = atan2(f_eci[1], f_eci[0])

    p = norm(c_eci) ** 2 / GMe  # semi-latus rectum
    e = norm(f_eci) / GMe  # eccentricity
    a = p / (1.0 - e**2)  # semi-major axis

    true_anomaly_rad = acos(f1_eci[0] * nr[0] + f1_eci[1] * nr[1] + f1_eci[2] * nr[2])
    if v_eci[0] * r_eci[0] + v_eci[1] * r_eci[1] + v_eci[2] * r_eci[2] < 0.0:
        true_anomaly_rad = 2.0 * np.pi - true_anomaly_rad

    if ascending_node_rad < 0.0:
        ascending_node_rad += 2.0 * np.pi
    if argument_perigee < 0.0:
        argument_perigee += 2.0 * np.pi
    if true_anomaly_rad < 0.0:
        true_anomaly_rad += 2.0 * np.pi

    return np.array(
        [
            a,
            e,
            degrees(inclination_rad),
            degrees(ascending_node_rad),
            degrees(argument_perigee),
            degrees(true_anomaly_rad),
        ]
    )


def angular_momentum_vec(r, v):
    """Calculates angular momentum vector from position and velocity vectors.
    Args:
        r (ndarray) : position vector in ECI frame [m]
        v (ndarray) : inertial velocity vector in ECI frame [m]
    Returns:
        ndarray : angular momentum vector [m^2/s]
    """
    return np.cross(r, v)


def angular_momentum(r, v):
    """Calculates angular momentum from position and velocity vectors.
    Args:
        r (ndarray) : position vector in ECI frame [m]
        v (ndarray) : inertial velocity vector in ECI frame [m]
    Returns:
        float64 : angular momentum [m^2/s]
    """
    return norm(angular_momentum_vec(r, v))


def inclination_cosine(r, v):
    """Calculates cosine of inclination from position and velocity vectors.
    Args:
        r (ndarray) : position vector in ECI frame [m]
        v (ndarray) : inertial velocity vector in ECI frame [m]
    Returns:
        float64 : cosine of inclination
    """
    return angular_momentum_vec(r, v)[2] / angular_momentum(r, v)


def inclination_rad(r, v):
    """Calculates inclination from position and velocity vectors.
    Args:
        r (ndarray) : position vector in ECI frame [m]
        v (ndarray) : inertial velocity vector in ECI frame [m]
    Returns:
        float64 : inclination [rad]
    """
    return acos(inclination_cosine(r, v))


def laplace_vector(r, v):
    """Calculates Laplace vector from position and velocity vectors.
    Args:
        r (ndarray) : position vector in ECI frame [m]
        v (ndarray) : inertial velocity vector in ECI frame [m]
    Returns:
        ndarray : Laplace vector
    """
    h = angular_momentum_vec(r, v)
    return np.cross(v, h) - 3.986004418e14 * r / norm(r)


def orbit_energy(r, v):
    """Calculates specific orbital energy.
    Args:
        r (ndarray) : position vector in ECI frame [m]
        v (ndarray) : inertial velocity vector in ECI frame [m]
    Returns:
        float64 : specific orbital energy [J/kg]
    """
    return 0.5 * norm(v) ** 2 - 3.986004418e14 / norm(r)


def angular_momentum_from_altitude(ha, hp):
    """Calculates angular momentum from altitude.
    Args:
        ha (float64) : altitude of the apogee [m]
        hp (float64) : altitude of the perigee [m]
    Returns:
        float64 : angular momentum [m^2/s]
    """
    ra = 6378137 + ha
    rp = 6378137 + hp
    a = (ra + rp) / 2
    vp = sqrt(3.986004418e14 * (2 / rp - 1 / a))
    return rp * vp


def orbit_energy_from_altitude(ha, hp):
    """Calculates specific orbital energy from altitude.
    Args:
        ha (float64) : altitude of the apogee [m]
        hp (float64) : altitude of the perigee [m]
    Returns:
        float64 : specific orbital energy [J/kg]
    """
    ra = 6378137 + ha
    rp = 6378137 + hp
    a = (ra + rp) / 2
    return -3.986004418e14 / 2 / a
