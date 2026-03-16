#
# CasADi NLP builder for rocket trajectory optimization.
#
# Replaces pyoptsparse + hand-written Jacobians with CasADi Opti.
# All gradients are computed automatically via AD.
#

import os
import shutil

import casadi as ca
import numpy as np

from atmosphere_casadi import create_atmosphere_interpolants
from coordinate_casadi import (
    angular_momentum,
    angular_momentum_from_altitude,
    eci2ecef,
    eci2geodetic,
    geodetic2ecef_np,
    inclination_rad,
    orbit_energy,
    orbit_energy_from_altitude,
)
from dynamics_casadi import (
    angle_of_attack,
    compute_acceleration,
    dynamic_pressure,
    quaternion_derivative,
)
from iip_casadi import iip_latlon

# ================================================================
# Interpolant creation helpers
# ================================================================


def _make_wind_interpolants(wind_table):
    """Create CasADi interpolants for wind north/east components."""
    alt = wind_table[:, 0].copy()
    wn = wind_table[:, 1].copy()
    we = wind_table[:, 2].copy()
    # Ensure strictly increasing grid (deduplicate, sort)
    _, idx = np.unique(alt, return_index=True)
    alt, wn, we = alt[idx], wn[idx], we[idx]
    # Extend to cover the full altitude range if needed
    if alt[-1] < 200000:
        alt = np.append(alt, 200000)
        wn = np.append(wn, 0.0)
        we = np.append(we, 0.0)
    fn_n = ca.interpolant("wind_n", "bspline", [alt], wn)
    fn_e = ca.interpolant("wind_e", "bspline", [alt], we)
    return fn_n, fn_e


def _make_ca_interpolant(ca_table):
    """Create CasADi interpolant for axial force coefficient."""
    mach = ca_table[:, 0].copy()
    coeff = ca_table[:, 1].copy()
    return ca.interpolant("ca_coeff", "bspline", [mach], coeff)


# ================================================================
# Main builder
# ================================================================


def build_and_solve(
    pdict, condition, unitdict, xdict_init, solver_options, solver_name="ipopt",
    output_dir="output",
):
    """Build CasADi Opti NLP and solve.

    Args:
        pdict:           problem parameters (events, sections, wind, CA, etc.)
        condition:       flight conditions (init, terminal, constraints)
        unitdict:        normalization units
        xdict_init:      initial guess (flat numpy arrays)
        solver_options:  dict of solver options (IPOPT or SNOPT)
        solver_name:     "ipopt" or "snopt"

    Returns:
        xStar:      dict of optimized variables (same format as pyoptsparse xStar)
        solve_stats: dict with timing info
    """
    ps = pdict["ps_params"]
    N = pdict["N"]
    M = pdict["M"]
    num_sections = pdict["num_sections"]

    unit_m = unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_u = unitdict["u"]
    unit_t = unitdict["t"]

    # --- Create interpolants ---
    density_fn, pressure_fn, sound_speed_fn = create_atmosphere_interpolants()
    wind_n_fn, wind_e_fn = _make_wind_interpolants(pdict["wind_table"])
    ca_fn = _make_ca_interpolant(pdict["ca_table"])

    # ============================================================
    # CasADi Opti
    # ============================================================
    opti = ca.Opti()

    # Decision variables (all normalised)
    mass = opti.variable(M)
    pos = opti.variable(M, 3)
    vel = opti.variable(M, 3)
    quat = opti.variable(M, 4)
    u = opti.variable(N, 2)
    t = opti.variable(num_sections + 1)

    # Bounds
    opti.subject_to(opti.bounded(1e-9, mass, 2.0))
    opti.subject_to(opti.bounded(-10.0, pos, 10.0))
    opti.subject_to(opti.bounded(-20.0, vel, 20.0))
    opti.subject_to(opti.bounded(-1.0, quat, 1.0))
    opti.subject_to(opti.bounded(-30, u, 30))
    opti.subject_to(opti.bounded(0.0, t, 1.5))

    # ============================================================
    # Objective
    # ============================================================
    opti.minimize(-mass[0])

    # ============================================================
    # 1. Initial conditions
    # ============================================================
    if condition["OptimizationMode"] != "Payload":
        opti.subject_to(mass[0] == condition["init"]["mass"] / unit_m)
    opti.subject_to(pos[0, :].T == condition["init"]["position"] / unit_pos)
    opti.subject_to(vel[0, :].T == condition["init"]["velocity"] / unit_vel)
    opti.subject_to(quat[0, :].T == condition["init"]["quaternion"])

    # ============================================================
    # 2. Time constraints
    # ============================================================
    time_is_constrained = [False] * (num_sections + 1)
    opti.subject_to(t[0] == 0.0)  # initial time is always 0
    time_is_constrained[0] = True

    for i in range(1, num_sections + 1):
        p = pdict["params"][i]
        mode = p.get("time_constraint_mode", "free")
        value = p.get("time_constraint_value")
        tr = p["time_ref"]

        if mode == "fixed":
            # Absolute time
            opti.subject_to(t[i] == value / unit_t)
            time_is_constrained[i] = True

        elif mode == "relative":
            if tr in pdict["event_index"]:
                i_ref = pdict["event_index"][tr]
                if i_ref != i:
                    opti.subject_to(t[i] - t[i_ref] == value / unit_t)
                    time_is_constrained[i] = True

    # Forward-time ordering with minimum section duration
    min_section_dt = pdict.get("min_section_duration", 1.0) / unit_t
    for i in range(num_sections):
        if not time_is_constrained[i] or not time_is_constrained[i + 1]:
            opti.subject_to(t[i + 1] >= t[i] + min_section_dt)

    # ============================================================
    # 3. Dynamics constraints (per section)
    # ============================================================
    for sec in range(num_sections):
        ua, ub, xa, xb, n = ps.get_index(sec)
        D = ps.D(sec)  # (n, n+1)  NumPy
        tau = ps.tau(sec)

        to_s, tf_s = t[sec], t[sec + 1]
        dt_half = (tf_s - to_s) * unit_t / 2.0

        p = pdict["params"][sec]
        engine_on = p["engineOn"]
        thrust_vac = p["thrust"]
        massflow = p["massflow"]
        ref_area = p["reference_area"]
        nozzle_area = p["nozzle_area"]

        # Slices
        m_s = mass[xa:xb]  # (n+1,)
        r_s = pos[xa:xb, :]  # (n+1, 3)
        v_s = vel[xa:xb, :]  # (n+1, 3)
        q_s = quat[xa:xb, :]  # (n+1, 4)
        u_s = u[ua:ub, :]  # (n, 2)

        # Time at each LGR node (physical seconds).
        t_nodes_phys = []
        for k in range(n):
            t_nodes_phys.append(
                (tau[k] * (tf_s - to_s) / 2 + (tf_s + to_s) / 2) * unit_t
            )

        # --- Mass dynamics ---
        # Raw collocation form: D @ m - rhs * dt_half = 0
        D_m = ca.MX(D)  # promote to CasADi
        if engine_on:
            lhs_m = D_m @ m_s
            rhs_val = -massflow / unit_m * dt_half
            for k in range(n):
                opti.subject_to(lhs_m[k] == rhs_val)
        else:
            for k in range(1, n + 1):
                opti.subject_to(m_s[k] == m_s[0])

        # --- Position dynamics ---
        # D @ r - v * unit_vel/unit_pos * dt_half = 0
        for dim in range(3):
            lhs_r = D_m @ r_s[:, dim]
            for k in range(n):
                rhs_r = v_s[k + 1, dim] * unit_vel * dt_half / unit_pos
                opti.subject_to(lhs_r[k] == rhs_r)

        # --- Velocity dynamics ---
        lhs_v = [D_m @ v_s[:, dim] for dim in range(3)]
        for k in range(n):
            mk = m_s[k + 1] * unit_m
            rk = r_s[k + 1, :].T * unit_pos
            vk = v_s[k + 1, :].T * unit_vel
            qk = q_s[k + 1, :].T
            tk = t_nodes_phys[k]

            acc = compute_acceleration(
                mk,
                rk,
                vk,
                qk,
                tk,
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

            for dim in range(3):
                rhs_v = acc[dim] / unit_vel * dt_half
                opti.subject_to(lhs_v[dim][k] == rhs_v)

        # --- Quaternion dynamics ---
        att = p["attitude"]
        if att in ("hold", "vertical"):
            for k in range(1, n + 1):
                for dim in range(4):
                    opti.subject_to(q_s[k, dim] == q_s[0, dim])
        else:
            lhs_q = [D_m @ q_s[:, dim] for dim in range(4)]
            for k in range(n):
                qk = q_s[k + 1, :].T
                dqdt = quaternion_derivative(qk, u_s[k, 0], u_s[k, 1], unit_u)
                for dim in range(4):
                    rhs_q = dqdt[dim] * dt_half
                    opti.subject_to(lhs_q[dim][k] == rhs_q)

    # ============================================================
    # 4. Knot continuity
    # ============================================================
    section_sep_map = {}  # section_sep_index → (section_ig_index, stage)
    for key, stage in pdict["RocketStage"].items():
        if stage["separation_at"] is not None:
            sec_ig = next(
                i
                for i, v in enumerate(pdict["params"])
                if v["name"] == stage["ignition_at"]
            )
            sec_sep = next(
                i
                for i, v in enumerate(pdict["params"])
                if v["name"] == stage["separation_at"]
            )
            section_sep_map[sec_sep] = (sec_ig, stage)

            mass_stage = (
                stage["mass_dry"]
                + stage["mass_propellant"]
                + sum(item["mass"] for item in stage["dropMass"].values())
            )
            idx_ig = ps.index_start_x(sec_ig)
            idx_sep = ps.index_start_x(sec_sep)
            opti.subject_to(mass[idx_ig] - mass[idx_sep] == mass_stage / unit_m)

    for i in range(1, num_sections):
        xa_i = ps.index_start_x(i)
        # Mass continuity
        if i not in section_sep_map:
            jettison = pdict["params"][i]["mass_jettison"]
            opti.subject_to(mass[xa_i] - mass[xa_i - 1] == -jettison / unit_m)
        # Position, velocity, quaternion continuity
        for dim in range(3):
            opti.subject_to(pos[xa_i, dim] == pos[xa_i - 1, dim])
            opti.subject_to(vel[xa_i, dim] == vel[xa_i - 1, dim])
        for dim in range(4):
            opti.subject_to(quat[xa_i, dim] == quat[xa_i - 1, dim])

    # ============================================================
    # 5. Terminal orbit conditions
    # ============================================================
    pos_f = pos[-1, :].T * unit_pos
    vel_f = vel[-1, :].T * unit_vel

    if (
        condition["altitude_perigee"] is not None
        and condition["altitude_apogee"] is not None
    ):
        c_target = angular_momentum_from_altitude(
            condition["altitude_perigee"], condition["altitude_apogee"]
        )
        e_target = orbit_energy_from_altitude(
            condition["altitude_perigee"], condition["altitude_apogee"]
        )
    else:
        from math import cos, radians

        c_target = condition["radius"] * condition["vel_tangential_geocentric"]
        vf = condition["vel_tangential_geocentric"] / cos(
            radians(condition["flightpath_vel_inertial_geocentric"])
        )
        e_target = vf**2 / 2.0 - 3.986004418e14 / condition["radius"]

    c_val = angular_momentum(pos_f, vel_f)
    e_val = orbit_energy(pos_f, vel_f)
    opti.subject_to(e_val / e_target - 1.0 == 0)
    opti.subject_to(c_val / c_target - 1.0 == 0)

    if condition["inclination"] is not None:
        inc_target = np.radians(condition["inclination"])
        opti.subject_to(inclination_rad(pos_f, vel_f) - inc_target == 0)

    # ============================================================
    # 6. Angular rate constraints
    # ============================================================
    for sec in range(num_sections):
        ua, ub, xa, xb, n = ps.get_index(sec)
        u_s = u[ua:ub, :]
        att = pdict["params"][sec]["attitude"]

        if att in ("hold", "vertical"):
            for k in range(n):
                opti.subject_to(u_s[k, 0] == 0)
                opti.subject_to(u_s[k, 1] == 0)

        elif att in ("kick-turn", "pitch"):
            for k in range(1, n):
                opti.subject_to(u_s[k, 0] == u_s[0, 0])
            for k in range(n):
                opti.subject_to(u_s[k, 1] == 0)

        elif att == "pitch-yaw":
            for k in range(1, n):
                opti.subject_to(u_s[k, 0] == u_s[0, 0])
                opti.subject_to(u_s[k, 1] == u_s[0, 1])

        elif att == "same-rate":
            u_prev = u[ua - 1, :]
            for k in range(n):
                opti.subject_to(u_s[k, 0] == u_prev[0])
                opti.subject_to(u_s[k, 1] == u_prev[1])

        elif att in ("zero-lift-turn", "free"):
            pass  # no constraint on angular rates

        # --- Rate limit (pitch / yaw) ---
        rate_limit = pdict["params"][sec].get("rate_limit", {})
        if "pitch" in rate_limit:
            limit_pitch = rate_limit["pitch"]  # deg/s
            for k in range(n):
                opti.subject_to(u_s[k, 0] * unit_u <= limit_pitch)
                opti.subject_to(-u_s[k, 0] * unit_u <= limit_pitch)
        if "yaw" in rate_limit:
            limit_yaw = rate_limit["yaw"]  # deg/s
            for k in range(n):
                opti.subject_to(u_s[k, 1] * unit_u <= limit_yaw)
                opti.subject_to(-u_s[k, 1] * unit_u <= limit_yaw)

    # ============================================================
    # 7. Kick-turn positivity (pitch rate must be negative → −u >= 0)
    # ============================================================
    for sec in range(num_sections - 1):
        if "kick" in pdict["params"][sec]["attitude"]:
            ua, ub, xa, xb, n = ps.get_index(sec)
            for k in range(n):
                opti.subject_to(-u[ua + k, 0] * unit_u >= 0)

    # ============================================================
    # 8. Propellant mass constraints
    # ============================================================
    for key, stage in pdict["RocketStage"].items():
        sec_ig = next(
            i
            for i, v in enumerate(pdict["params"])
            if v["name"] == stage["ignition_at"]
        )
        sec_co = next(
            i for i, v in enumerate(pdict["params"]) if v["name"] == stage["cutoff_at"]
        )
        m_ig = mass[ps.index_start_x(sec_ig)]
        m_co = mass[ps.index_start_x(sec_co)]
        d_mass = stage["mass_propellant"]
        if stage["dropMass"]:
            d_mass += sum(item["mass"] for item in stage["dropMass"].values())
        opti.subject_to(-m_ig + m_co + d_mass / unit_m >= 0)

    # ============================================================
    # 9. Aerodynamic constraints (AOA, Q, Q-alpha)
    # ============================================================
    for sec in range(num_sections - 1):
        section_name = pdict["params"][sec]["name"]
        ua, ub, xa, xb, n = ps.get_index(sec)
        to_s, tf_s = t[sec], t[sec + 1]
        tau = ps.tau(sec)

        def _phys_time(k):
            return (tau[k] * (tf_s - to_s) / 2 + (tf_s + to_s) / 2) * unit_t

        def _phys_time_boundary():
            return to_s * unit_t

        # --- AOA constraint ---
        if section_name in condition.get("AOA_max", {}):
            aoa_max_rad = condition["AOA_max"][section_name]["value"] * np.pi / 180.0
            rng = condition["AOA_max"][section_name]["range"]

            if rng == "all":
                indices = range(n + 1)
            else:  # "initial"
                indices = [0]

            for ki in indices:
                if ki == 0:
                    rk = pos[xa, :].T * unit_pos
                    vk = vel[xa, :].T * unit_vel
                    qk = quat[xa, :].T
                    tk = _phys_time_boundary()
                else:
                    rk = pos[xa + ki, :].T * unit_pos
                    vk = vel[xa + ki, :].T * unit_vel
                    qk = quat[xa + ki, :].T
                    tk = _phys_time(ki - 1)

                aoa = angle_of_attack(rk, vk, qk, tk, wind_n_fn, wind_e_fn)
                opti.subject_to(aoa <= aoa_max_rad)

        # --- Dynamic pressure constraint ---
        if section_name in condition.get("dynamic_pressure_max", {}):
            q_max = condition["dynamic_pressure_max"][section_name]["value"]
            rng = condition["dynamic_pressure_max"][section_name]["range"]

            if rng == "all":
                indices = range(n + 1)
            else:
                indices = [0]

            for ki in indices:
                if ki == 0:
                    rk = pos[xa, :].T * unit_pos
                    vk = vel[xa, :].T * unit_vel
                    tk = _phys_time_boundary()
                else:
                    rk = pos[xa + ki, :].T * unit_pos
                    vk = vel[xa + ki, :].T * unit_vel
                    tk = _phys_time(ki - 1)

                q = dynamic_pressure(rk, vk, tk, wind_n_fn, wind_e_fn, density_fn)
                opti.subject_to(q / q_max - 1.0 <= 0)

        # --- Q-alpha constraint ---
        if section_name in condition.get("Q_alpha_max", {}):
            qa_max = condition["Q_alpha_max"][section_name]["value"] * np.pi / 180.0
            rng = condition["Q_alpha_max"][section_name]["range"]

            if rng == "all":
                indices = range(n + 1)
            else:
                indices = [0]

            for ki in indices:
                if ki == 0:
                    rk = pos[xa, :].T * unit_pos
                    vk = vel[xa, :].T * unit_vel
                    qk = quat[xa, :].T
                    tk = _phys_time_boundary()
                else:
                    rk = pos[xa + ki, :].T * unit_pos
                    vk = vel[xa + ki, :].T * unit_vel
                    qk = quat[xa + ki, :].T
                    tk = _phys_time(ki - 1)

                aoa = angle_of_attack(rk, vk, qk, tk, wind_n_fn, wind_e_fn)
                q = dynamic_pressure(rk, vk, tk, wind_n_fn, wind_e_fn, density_fn)
                opti.subject_to(q * aoa / qa_max - 1.0 <= 0)

    # ============================================================
    # 10. Waypoint constraints (posLLH + IIP)
    # ============================================================
    if "waypoint" in condition:
        for sec in range(num_sections - 1):
            section_name = pdict["params"][sec]["name"]
            if section_name not in condition["waypoint"]:
                continue

            wp = condition["waypoint"][section_name]
            xa_i = ps.index_start_x(sec)
            r_phys = pos[xa_i, :].T * unit_pos
            v_phys = vel[xa_i, :].T * unit_vel
            t_phys = t[sec] * unit_t

            posLLH = eci2geodetic(r_phys, t_phys)  # [lat_deg, lon_deg, alt_m]

            # Latitude
            if "lat" in wp:
                if "exact" in wp["lat"]:
                    opti.subject_to((posLLH[0] - wp["lat"]["exact"]) / 90.0 == 0)
                if "min" in wp["lat"]:
                    opti.subject_to((posLLH[0] - wp["lat"]["min"]) / 90.0 >= 0)
                if "max" in wp["lat"]:
                    opti.subject_to(-(posLLH[0] - wp["lat"]["max"]) / 90.0 >= 0)

            # Longitude
            if "lon" in wp:
                if "exact" in wp["lon"]:
                    opti.subject_to((posLLH[1] - wp["lon"]["exact"]) / 180.0 == 0)
                if "min" in wp["lon"]:
                    opti.subject_to((posLLH[1] - wp["lon"]["min"]) / 180.0 >= 0)
                if "max" in wp["lon"]:
                    opti.subject_to(-(posLLH[1] - wp["lon"]["max"]) / 180.0 >= 0)

            # Altitude
            if "altitude" in wp:
                if "exact" in wp["altitude"]:
                    opti.subject_to(posLLH[2] / wp["altitude"]["exact"] - 1.0 == 0)
                if "min" in wp["altitude"]:
                    opti.subject_to(posLLH[2] / wp["altitude"]["min"] - 1.0 >= 0)
                if "max" in wp["altitude"]:
                    opti.subject_to(-(posLLH[2] / wp["altitude"]["max"] - 1.0) >= 0)

            # IIP constraints (analytical Kepler propagation, no callback)
            if "lat_IIP" in wp or "lon_IIP" in wp:
                iip_lat, iip_lon = iip_latlon(r_phys, v_phys, t_phys)

                if "lat_IIP" in wp:
                    if "exact" in wp["lat_IIP"]:
                        opti.subject_to((iip_lat - wp["lat_IIP"]["exact"]) / 90.0 == 0)
                    if "min" in wp["lat_IIP"]:
                        opti.subject_to((iip_lat - wp["lat_IIP"]["min"]) / 90.0 >= 0)
                    if "max" in wp["lat_IIP"]:
                        opti.subject_to(-(iip_lat - wp["lat_IIP"]["max"]) / 90.0 >= 0)

                if "lon_IIP" in wp:
                    if "exact" in wp["lon_IIP"]:
                        opti.subject_to((iip_lon - wp["lon_IIP"]["exact"]) / 180.0 == 0)
                    if "min" in wp["lon_IIP"]:
                        opti.subject_to((iip_lon - wp["lon_IIP"]["min"]) / 180.0 >= 0)
                    if "max" in wp["lon_IIP"]:
                        opti.subject_to(-(iip_lon - wp["lon_IIP"]["max"]) / 180.0 >= 0)

    # ============================================================
    # 11. Antenna elevation constraints
    # ============================================================
    if "antenna" in condition:
        for ant_data in condition["antenna"].values():
            posECEF_ANT = geodetic2ecef_np(
                ant_data["lat"], ant_data["lon"], ant_data["altitude"]
            )

            for sec in range(num_sections - 1):
                section_name = pdict["params"][sec]["name"]
                if section_name not in ant_data.get("elevation_min", {}):
                    continue

                elev_min_deg = ant_data["elevation_min"][section_name]
                xa_i = ps.index_start_x(sec)
                r_phys = pos[xa_i, :].T * unit_pos
                t_phys = t[sec] * unit_t

                # Compute sin(elevation)
                pos_ecef = eci2ecef(r_phys, t_phys)
                direction = pos_ecef - posECEF_ANT
                direction_n = direction / ca.norm_2(direction)
                # Vertical at antenna (up = -down in NED)
                lat_ant = np.radians(ant_data["lat"])
                lon_ant = np.radians(ant_data["lon"])
                sl, cl = np.sin(lat_ant), np.cos(lat_ant)
                sn, cn = np.sin(lon_ant), np.cos(lon_ant)
                up_ecef = np.array([-sl * cn, -sl * sn, cl]) * (-1)  # NED down → up
                # Actually: vertical up in ECEF at geodetic location
                up_ecef = np.array([cl * cn, cl * sn, sl])

                sin_elev = ca.dot(direction_n, up_ecef)
                sin_elev_min = np.sin(np.radians(elev_min_deg))
                opti.subject_to(sin_elev >= sin_elev_min)

    # ============================================================
    # Initial guess
    # ============================================================
    opti.set_initial(mass, xdict_init["mass"])
    opti.set_initial(pos, xdict_init["position"].reshape(-1, 3))
    opti.set_initial(vel, xdict_init["velocity"].reshape(-1, 3))
    opti.set_initial(quat, xdict_init["quaternion"].reshape(-1, 4))
    opti.set_initial(u, xdict_init["u"].reshape(-1, 2))
    opti.set_initial(t, xdict_init["t"])

    # ============================================================
    # Solver
    # ============================================================
    p_opts = {}

    if solver_name == "snopt":
        # ----------------------------------------------------------
        # SNOPT (SQP)
        # ----------------------------------------------------------
        snopt_opts = dict(solver_options)
        snopt_opts.setdefault("Major iterations limit", 2000)
        snopt_opts.setdefault("Minor iterations limit", 500)
        snopt_opts.setdefault("Major feasibility tolerance", 1e-6)
        snopt_opts.setdefault("Major optimality tolerance", 1e-6)
        snopt_opts.setdefault("Minor feasibility tolerance", 1e-6)
        snopt_opts.setdefault("Scale option", 0)

        opti.solver("snopt", {"snopt": snopt_opts}, {})
        print("Solving with SNOPT...")
        print("  (iteration log: solver.out)")
        try:
            sol = opti.solve()
            print("SNOPT: Converged.")
        except RuntimeError:
            print("WARNING: SNOPT did not converge — returning best iterate")
            sol = opti.debug

        # Move solver.out to output directory with mission name
        snopt_out_dest = os.path.join(
            output_dir, f"{pdict['name']}-SNOPT-print.out"
        )
        try:
            with open("solver.out") as f:
                lines = f.readlines()
            printing = False
            for line in lines:
                if "SNOPTC EXIT" in line or "SNOPTC INFO" in line:
                    printing = True
                if printing:
                    print(line, end="")
                    if line.strip() == "":
                        break
            shutil.move("solver.out", snopt_out_dest)
            print(f"  (solver log: {snopt_out_dest})")
        except FileNotFoundError:
            pass

    else:
        # ----------------------------------------------------------
        # IPOPT (interior point)
        # ----------------------------------------------------------
        ipopt_options = dict(solver_options)
        ipopt_options.setdefault("hessian_approximation", "limited-memory")
        opti.solver("ipopt", p_opts, ipopt_options)
        print("Solving with IPOPT...")
        try:
            sol = opti.solve()
            print("IPOPT: Converged.")
        except RuntimeError:
            print("WARNING: IPOPT did not converge — returning best iterate")
            sol = opti.debug

    # ============================================================
    # Extract solution in the same format as pyoptsparse xStar
    # ============================================================
    xStar = {
        "mass": np.array(sol.value(mass)).flatten(),
        "position": np.array(sol.value(pos)).flatten(),
        "velocity": np.array(sol.value(vel)).flatten(),
        "quaternion": np.array(sol.value(quat)).flatten(),
        "u": np.array(sol.value(u)).flatten(),
        "t": np.array(sol.value(t)).flatten(),
    }

    solve_stats = {
        "solver": f"{solver_name.upper()} (CasADi)",
        "stats": sol.stats() if hasattr(sol, "stats") else {"return_status": "debug"},
    }

    return xStar, solve_stats
