#
# Analytical IIP (Instantaneous Impact Point) — CasADi symbolic version.
#
# Implements the FAA IIP algorithm (AC 401.5-1 Appendix C, (v))
# with spherical Earth assumption (no iteration).
# All operations are CasADi-symbolic, so AD gives exact gradients.
#

import casadi as ca
from coordinate_casadi import MU_EARTH, R_EARTH_A, OMEGA_EARTH, E2


def iip_latlon(pos_eci, vel_eci, t_phys):
    """Compute IIP latitude and longitude analytically (no iteration).

    Uses Kepler f-and-g series propagation to find the point where
    the ballistic (thrust-off) trajectory intersects a sphere of
    radius R_EARTH_A.  Earth rotation during flight is accounted for.

    Args:
        pos_eci:  ECI position [m]  (3-vector, CasADi MX/SX)
        vel_eci:  ECI velocity [m/s] (3-vector, CasADi MX/SX)
        t_phys:   time since epoch [s] (scalar, CasADi MX/SX)

    Returns:
        (lat_deg, lon_deg): geodetic latitude and east longitude
                            of impact point [deg]
    """
    mu = MU_EARTH
    R = R_EARTH_A  # spherical Earth radius for intersection

    r0 = ca.norm_2(pos_eci)
    v0_sq = ca.dot(vel_eci, vel_eci)

    # (v)-(E): e * cos(E0)
    eps_cos = r0 * v0_sq / mu - 1.0

    # (v)-(F): semi-major axis
    a = r0 / (1.0 - eps_cos)

    # (v)-(G): e * sin(E0)
    eps_sin = ca.dot(pos_eci, vel_eci) / ca.sqrt(mu * a)

    # (v)-(H): e^2
    eps2 = eps_cos**2 + eps_sin**2

    # (v)-(I): e * cos(E_impact), with r_impact = R
    eps_k_cos = (a - R) / a

    # (v)-(J): e * sin(E_impact) — negative (descending branch)
    eps_k_sin = -ca.sqrt(ca.fmax(1e-30, eps2 - eps_k_cos**2))

    # (v)-(K),(L): cos/sin of (E_impact - E0)
    delta_eps_cos = (eps_k_cos * eps_cos + eps_k_sin * eps_sin) / eps2
    delta_eps_sin = (eps_k_sin * eps_cos - eps_k_cos * eps_sin) / eps2

    # (v)-(M),(N): f-and-g series (Kepler propagation)
    a32_over_mu = ca.sqrt(a**3 / mu)
    f = (delta_eps_cos - eps_cos) / (1.0 - eps_cos)
    g = (delta_eps_sin + eps_sin - eps_k_sin) * a32_over_mu

    # (v)-(O): impact position in ECI  [r_impact = f * r0 + g * v0]
    r_impact = f * pos_eci + g * vel_eci
    E_k = r_impact[0]
    F_k = r_impact[1]
    G_k = r_impact[2]
    r_impact_mag = ca.norm_2(r_impact)

    # (v)-(R),(S): flight time from current position to impact
    delta_eps = ca.atan2(delta_eps_sin, delta_eps_cos)
    time_flight = (delta_eps + eps_sin - eps_k_sin) * a32_over_mu

    # (v)-(T): geocentric latitude at impact
    phi_geocentric = ca.asin(G_k / r_impact_mag)

    # (v)-(U): geodetic latitude (ellipsoid correction)
    phi_geodetic = ca.atan2(ca.tan(phi_geocentric), 1.0 - E2)

    # (v)-(V): east longitude (with Earth rotation correction)
    lambda_impact = ca.atan2(F_k, E_k) - OMEGA_EARTH * (t_phys + time_flight)

    lat_deg = phi_geodetic * 180.0 / ca.pi
    lon_deg = lambda_impact * 180.0 / ca.pi

    return lat_deg, lon_deg
