#
# US Standard Atmosphere 1976 — CasADi interpolant version.
#
# Pre-computes density, pressure, and speed-of-sound tables,
# then wraps them in CasADi linear interpolants for use in
# symbolic expressions.
#

import casadi as ca
import numpy as np

# --- Physical constants ---
_R_AIR = 287.0531  # specific gas constant for dry air [J/(kg·K)]
_G0 = 9.80665  # standard gravity [m/s^2]
_GAMMA = 1.4  # heat capacity ratio
_R_EARTH_ATMO = 6356766.0  # Earth radius for geopotential altitude [m]

# --- Layer definitions (geopotential altitude) ---
_H_BASE = [0, 11000, 20000, 32000, 47000, 51000, 71000, 84852]
_T_BASE = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
_LAPSE = [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0]


def _build_tables():
    """Pre-compute atmosphere tables from 0 to 90 km (100 m resolution)."""
    altitudes = np.arange(0, 90001, 100, dtype=np.float64)
    n = len(altitudes)
    rho = np.zeros(n)
    pres = np.zeros(n)
    sos = np.zeros(n)

    # Base pressures at each layer boundary
    p_base = [101325.0]
    for layer in range(len(_H_BASE) - 1):
        h_b = _H_BASE[layer]
        T_b = _T_BASE[layer]
        L = _LAPSE[layer]
        p_b = p_base[-1]
        dh = _H_BASE[layer + 1] - h_b
        if abs(L) > 1e-12:
            p_next = p_b * (T_b / (T_b + L * dh)) ** (_G0 / (_R_AIR * L))
        else:
            p_next = p_b * np.exp(-_G0 * dh / (_R_AIR * T_b))
        p_base.append(p_next)

    for idx, h in enumerate(altitudes):
        # Find layer
        layer = 0
        for k in range(len(_H_BASE) - 1):
            if h >= _H_BASE[k]:
                layer = k
        h_b = _H_BASE[layer]
        T_b = _T_BASE[layer]
        L = _LAPSE[layer]
        p_b = p_base[layer]
        dh = h - h_b
        T = T_b + L * dh

        if abs(L) > 1e-12:
            p = p_b * (T_b / T) ** (_G0 / (_R_AIR * L))
        else:
            p = p_b * np.exp(-_G0 * dh / (_R_AIR * T_b))

        rho[idx] = p / (_R_AIR * T)
        pres[idx] = p
        sos[idx] = np.sqrt(_GAMMA * _R_AIR * T)

    return altitudes, rho, pres, sos


def geopotential_altitude(h_geometric):
    """Convert geometric altitude to geopotential altitude (CasADi compatible)."""
    return _R_EARTH_ATMO * h_geometric / (_R_EARTH_ATMO + h_geometric)


# Build tables once at module load
_ALT, _RHO, _PRES, _SOS = _build_tables()


def create_atmosphere_interpolants():
    """Create CasADi interpolant functions for density, pressure, speed of sound.

    Returns:
        (density_fn, pressure_fn, sound_speed_fn) — each takes geopotential altitude [m]
    """
    density_fn = ca.interpolant("atmo_density", "linear", [_ALT], _RHO)
    pressure_fn = ca.interpolant("atmo_pressure", "linear", [_ALT], _PRES)
    sound_speed_fn = ca.interpolant("atmo_sos", "linear", [_ALT], _SOS)
    return density_fn, pressure_fn, sound_speed_fn
