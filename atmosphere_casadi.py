#
# US Standard Atmosphere 1976 — CasADi interpolant version.
#
# Pre-computes density, pressure, and speed-of-sound tables,
# then wraps them in CasADi linear interpolants for use in
# symbolic expressions.
#

import casadi as ca
import numpy as np
import lib.USStandardAtmosphere as usatm

# --- Physical constants ---
_R_EARTH_ATMO = 6356766.0  # Earth radius for geopotential altitude [m]


def _build_tables():
    """Pre-compute atmosphere tables indexed by geometric altitude.

    Fine grid 0–90 km (100 m steps), coarse grid 90–1000 km (1 km steps).
    Values are evaluated at the corresponding geopotential altitude.
    """
    alt_geo_fine = np.arange(0, 90001, 100, dtype=np.float64)
    alt_geo_coarse = np.arange(91000, 1100001, 1000, dtype=np.float64)
    alt_geo = np.concatenate([alt_geo_fine, alt_geo_coarse])
    n = len(alt_geo)
    rho = np.zeros(n)
    pres = np.zeros(n)
    sos = np.zeros(n)

    for idx, h in enumerate(alt_geo):
        h_gp = usatm.geopotential_altitude(h)
        rho[idx] = usatm.airdensity_at(h_gp)
        pres[idx] = usatm.airpressure_at(h_gp)
        sos[idx] = usatm.speed_of_sound(h_gp)

    return alt_geo, rho, pres, sos


# Build tables once at module load
_ALT, _RHO, _PRES, _SOS = _build_tables()


def create_atmosphere_interpolants():
    """Create CasADi interpolant functions for density, pressure, speed of sound.

    Returns:
        (density_fn, pressure_fn, sound_speed_fn) — each takes geometric altitude [m]
    """
    density_fn = ca.interpolant("atmo_density", "linear", [_ALT], _RHO)
    pressure_fn = ca.interpolant("atmo_pressure", "linear", [_ALT], _PRES)
    sound_speed_fn = ca.interpolant("atmo_sos", "linear", [_ALT], _SOS)
    return density_fn, pressure_fn, sound_speed_fn
