#
# Dynamics equations — CasADi symbolic versions.
#
# Provides acceleration and quaternion-derivative functions
# that operate on CasADi symbolic expressions, enabling
# automatic differentiation of the NLP constraints.
#

import casadi as ca
from coordinate_casadi import (
    gravity_eci,
    quatrot,
    conj,
    quatmult,
    eci2ecef,
    ecef2geodetic,
    ecef2eci,
    ned_basis_ecef,
    OMEGA_EARTH,
)
from atmosphere_casadi import geopotential_altitude


def _compute_acceleration_impl(
    mass,
    pos_eci,
    vel_eci,
    quat,
    t_phys,
    thrust_vac,
    ref_area,
    nozzle_area,
    density_fn,
    pressure_fn,
    sound_speed_fn,
    wind_n_fn,
    wind_e_fn,
    ca_fn,
):
    """Core acceleration computation (pure symbolic, no Function wrapping)."""
    # --- Gravity ---
    grav = gravity_eci(pos_eci)

    # --- Geodetic altitude (for atmosphere lookup) ---
    pos_ecef = eci2ecef(pos_eci, t_phys)
    lat_r, lon_r, alt_m = ecef2geodetic(pos_ecef)
    alt_gp = geopotential_altitude(alt_m)
    alt_clamp = ca.fmax(0, ca.fmin(alt_gp, 89000))

    rho = density_fn(alt_clamp)
    p_atm = pressure_fn(alt_clamp)
    a_sound = sound_speed_fn(alt_clamp)

    # --- Thrust (body +X axis) ---
    thrust_mag = thrust_vac - nozzle_area * p_atm
    thrust_dir = quatrot(conj(quat), ca.vertcat(1, 0, 0))
    thrust_eci = thrust_dir * thrust_mag

    # --- Aerodynamic force ---
    # Wind: NED → ECEF → ECI
    alt_wind = ca.fmax(0, ca.fmin(alt_gp, 89000))
    w_n = wind_n_fn(alt_wind)
    w_e = wind_e_fn(alt_wind)
    north_ecef, east_ecef = ned_basis_ecef(lat_r, lon_r)
    wind_ecef = w_n * north_ecef + w_e * east_ecef
    wind_eci = ecef2eci(wind_ecef, t_phys)

    # Earth-rotation velocity at pos_eci
    omega_cross = ca.vertcat(-OMEGA_EARTH * pos_eci[1], OMEGA_EARTH * pos_eci[0], 0)

    # Airspeed in ECI
    vel_air = vel_eci - omega_cross - wind_eci
    vel_air_mag = ca.norm_2(vel_air)

    # Mach → drag coefficient
    mach = vel_air_mag / a_sound
    ca_coeff = ca_fn(ca.fmax(0, ca.fmin(mach, 25)))

    # Axial drag: F = 0.5 * rho * |v_air| * (-v_air) * S * CA
    aero_eci = 0.5 * rho * vel_air_mag * (-vel_air) * ref_area * ca_coeff

    # --- Total acceleration ---
    return grav + (thrust_eci + aero_eci) / mass


def compute_acceleration(
    mass,
    pos_eci,
    vel_eci,
    quat,
    t_phys,
    thrust_vac,
    ref_area,
    nozzle_area,
    density_fn,
    pressure_fn,
    sound_speed_fn,
    wind_n_fn,
    wind_e_fn,
    ca_fn,
):
    """Compute translational acceleration in ECI frame [m/s^2].

    Thin wrapper for use in non-mapped contexts (e.g. aero constraints).
    """
    return _compute_acceleration_impl(
        mass,
        pos_eci,
        vel_eci,
        quat,
        t_phys,
        thrust_vac,
        ref_area,
        nozzle_area,
        density_fn,
        pressure_fn,
        sound_speed_fn,
        wind_n_fn,
        wind_e_fn,
        ca_fn,
    )


def quaternion_derivative(quat, u_pitch_deg, u_yaw_deg, unit_u):
    """Quaternion kinematic equation.

    dq/dt = 0.5 * q ⊗ [0, 0, omega_pitch, omega_yaw]

    Args:
        quat:        ECI-to-body quaternion [w,x,y,z] (4,)
        u_pitch_deg: pitch rate [deg/s] (normalized value * unit_u gives deg/s)
        u_yaw_deg:   yaw rate [deg/s]
        unit_u:      unit for angular rate (typically 1.0)

    Returns:
        dq/dt (4,)
    """
    omega_pitch = u_pitch_deg * unit_u * ca.pi / 180.0
    omega_yaw = u_yaw_deg * unit_u * ca.pi / 180.0
    omega_quat = ca.vertcat(0, 0, omega_pitch, omega_yaw)
    return 0.5 * quatmult(quat, omega_quat)


def angle_of_attack(pos_eci, vel_eci, quat, t_phys, wind_n_fn, wind_e_fn):
    """Total angle of attack [rad].

    Angle between body X-axis and airspeed vector.
    """
    pos_ecef = eci2ecef(pos_eci, t_phys)
    lat_r, lon_r, alt_m = ecef2geodetic(pos_ecef)
    alt_gp = geopotential_altitude(alt_m)
    alt_clamp = ca.fmax(0, ca.fmin(alt_gp, 89000))

    w_n = wind_n_fn(alt_clamp)
    w_e = wind_e_fn(alt_clamp)
    north_ecef, east_ecef = ned_basis_ecef(lat_r, lon_r)
    wind_ecef = w_n * north_ecef + w_e * east_ecef
    wind_eci = ecef2eci(wind_ecef, t_phys)

    omega_cross = ca.vertcat(-OMEGA_EARTH * pos_eci[1], OMEGA_EARTH * pos_eci[0], 0)
    vel_air = vel_eci - omega_cross - wind_eci
    vel_air_dir = vel_air / ca.norm_2(vel_air)

    body_x = quatrot(conj(quat), ca.vertcat(1, 0, 0))
    # Use atan2(||cross||, dot) instead of acos(dot) to avoid infinite
    # derivative at AOA=0 and AOA=pi
    cross_vec = ca.cross(body_x, vel_air_dir)
    sin_aoa = ca.norm_2(cross_vec)
    cos_aoa = ca.dot(body_x, vel_air_dir)
    return ca.atan2(sin_aoa, cos_aoa)


def dynamic_pressure(pos_eci, vel_eci, t_phys, wind_n_fn, wind_e_fn, density_fn):
    """Dynamic pressure [Pa]."""
    pos_ecef = eci2ecef(pos_eci, t_phys)
    lat_r, lon_r, alt_m = ecef2geodetic(pos_ecef)
    alt_gp = geopotential_altitude(alt_m)
    alt_clamp = ca.fmax(0, ca.fmin(alt_gp, 89000))

    rho = density_fn(alt_clamp)

    w_n = wind_n_fn(alt_clamp)
    w_e = wind_e_fn(alt_clamp)
    north_ecef, east_ecef = ned_basis_ecef(lat_r, lon_r)
    wind_ecef = w_n * north_ecef + w_e * east_ecef
    wind_eci = ecef2eci(wind_ecef, t_phys)

    omega_cross = ca.vertcat(-OMEGA_EARTH * pos_eci[1], OMEGA_EARTH * pos_eci[0], 0)
    vel_air = vel_eci - omega_cross - wind_eci
    return 0.5 * rho * ca.dot(vel_air, vel_air)
