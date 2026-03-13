#
# The MIT License
#
# Copyright (c) 2022-2026 Interstellar Technologies Inc.
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

import json
import os
import sys
import time

import numpy as np
import pandas as pd

# --- Path setup ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from initialize import initialize_xdict_from_file, initialize_xdict_from_simulation
from lib.coordinate import (
    ecef2eci,
    geodetic2ecef,
    quat_eci2nedg,
    quat_from_euler,
    quatmult,
    vel_ecef2eci,
)
from output_result import output_result
from problem_builder import build_and_solve
from psmethod import PSparams

version = "1.0.0"


def main():
    # ============================================================
    # Load settings
    # ============================================================

    mission_name = sys.argv[1]
    mission_dir = os.path.dirname(os.path.abspath(mission_name))

    with open(mission_name, "r") as f:
        settings = json.load(f)


    def _resolve(path):
        """Resolve a path relative to the settings JSON file directory."""
        if os.path.isabs(path):
            return path
        return os.path.join(mission_dir, path)


    wind = pd.read_csv(_resolve(settings["Wind file"]))
    wind["wind_n"] = wind["wind_speed[m/s]"] * -np.cos(np.radians(wind["direction[deg]"]))
    wind["wind_e"] = wind["wind_speed[m/s]"] * -np.sin(np.radians(wind["direction[deg]"]))
    wind_table = wind[["altitude[m]", "wind_n", "wind_e"]].to_numpy()

    ca_table = pd.read_csv(_resolve(settings["CA file"])).to_numpy()

    stages = settings["RocketStage"]
    launch_conditions = settings["LaunchCondition"]
    terminal_conditions = settings["TerminalCondition"]

    # ============================================================
    # Reconstruct dropMass per stage
    # ============================================================
    for key, stage in stages.items():
        if "dropMass" not in stage:
            stage["dropMass"] = {}

    if "DropMass" in settings:
        for dm_name, dm_data in settings["DropMass"].items():
            sep_section = next(
                (
                    sec
                    for sec in settings["sections"]
                    if sec["name"] == dm_data["separation_at"]
                ),
                None,
            )
            if sep_section:
                stages[sep_section["rocket_stage"]]["dropMass"][dm_name] = dm_data

    # ============================================================
    # Build events DataFrame
    # ============================================================

    t_init = 0.0
    launchsite_ecef = np.array(
        geodetic2ecef(
            launch_conditions["lat"],
            launch_conditions["lon"],
            launch_conditions["altitude"],
        )
    )
    launchsite_eci = ecef2eci(launchsite_ecef, t_init)

    events_data = []
    for sec in settings["sections"]:
        tc = sec.get("time constraint", {})
        tc_mode = tc.get("mode", "free")
        tc_ref = tc.get("reference point")
        time_ref = "" if tc_mode == "free" else (tc_ref if tc_ref else sec["name"])

        events_data.append(
            {
                "time": sec["initial guess"]["time"],
                "time_ref": time_ref,
                "time_constraint_mode": tc_mode,
                "time_constraint_value": tc.get("value"),
                "rocketStage": int(sec["rocket_stage"]),
                "engineOn": sec["engine_mode"] != "off",
                "thrust": sec["thrust_vac"],
                "Isp_vac": sec["Isp_vac"],
                "nozzle_area": sec.get("nozzle_area", 0),
                "attitude": sec["attitude constraint"],
                "pitchrate_init": sec["initial guess"]["pitch_rate"],
                "yawrate_init": sec["initial guess"]["yaw_rate"],
                "num_nodes": sec["num_nodes"],
                "rate_limit": sec.get("rate_limit", {}),
            }
        )

    events = pd.DataFrame(events_data, index=[sec["name"] for sec in settings["sections"]])

    # Flight constraints
    flight_constraint = {
        "AOA_max": {},
        "dynamic_pressure_max": {},
        "Q_alpha_max": {},
        "waypoint": {},
        "antenna": {},
    }
    for sec in settings["sections"]:
        name = sec["name"]
        aoa = sec.get("AOA constraint", {})
        if aoa:
            flight_constraint["AOA_max"][name] = {
                "value": aoa["value"],
                "range": aoa["range"],
            }
        dyn_q = sec.get("dynamic pressure constraint", {})
        if dyn_q:
            flight_constraint["dynamic_pressure_max"][name] = {
                "value": dyn_q["value"],
                "range": dyn_q["range"],
            }
        qa = sec.get("Q-alpha constraint", {})
        if qa:
            flight_constraint["Q_alpha_max"][name] = {
                "value": qa["value"],
                "range": qa["range"],
            }
        wp = sec.get("waypoint constraint", {})
        if wp:
            wp_old = {}
            for wp_key, wp_val in wp.items():
                wp_old[wp_key] = {wp_val["mode"]: wp_val["value"]}
            flight_constraint["waypoint"][name] = wp_old
        ant = sec.get("antenna constraint", {})
        if ant:
            for ant_name, ant_data in ant.items():
                if ant_name not in flight_constraint["antenna"]:
                    flight_constraint["antenna"][ant_name] = {
                        "lat": ant_data["lat"],
                        "lon": ant_data["lon"],
                        "altitude": ant_data["altitude"],
                        "elevation_min": {},
                    }
                flight_constraint["antenna"][ant_name]["elevation_min"][name] = ant_data[
                    "elevation_min"
                ]

    num_sections = len(events) - 1

    events["timeduration"] = -events["time"].diff(-1)
    events.iloc[-1, events.columns.get_loc("timeduration")] = 9000.0
    events["timeFinishAt"] = events["time"] + events["timeduration"]
    events["mass_jettison"] = 0.0

    for key, stage in stages.items():
        if stage["separation_at"] in events.index:
            events.at[stage["separation_at"], "mass_jettison"] = stage["mass_dry"]
        if stage["dropMass"]:
            for dm_key, item in stage["dropMass"].items():
                if item["separation_at"] in events.index:
                    events.at[item["separation_at"], "mass_jettison"] = item["mass"]

    events["massflow"] = 0.0
    events["reference_area"] = 0.0
    for i in events.index:
        stage = stages[str(events.at[i, "rocketStage"])]
        events.at[i, "reference_area"] = stage["reference_area"]
        if events.at[i, "engineOn"]:
            events.at[i, "massflow"] = (
                events.at[i, "thrust"] / events.at[i, "Isp_vac"] / 9.80665
            )

    # ============================================================
    # Build pdict
    # ============================================================

    pdict = settings
    pdict["params"] = events.to_dict("records")
    nodes = events["num_nodes"][:-1]
    for i, event_name in enumerate(events.index):
        pdict["params"][i]["name"] = event_name
    pdict["event_index"] = {val["name"]: i for i, val in enumerate(pdict["params"])}

    N = sum(nodes)
    pdict["ps_params"] = PSparams(nodes)
    pdict["wind_table"] = wind_table
    pdict["ca_table"] = ca_table
    pdict["N"] = N
    pdict["M"] = N + num_sections
    pdict["num_sections"] = num_sections

    # ============================================================
    # Initial state
    # ============================================================

    r_init = launchsite_eci
    v_init = vel_ecef2eci(np.zeros(3), launchsite_ecef, t_init)
    quat_init = quatmult(
        quat_eci2nedg(r_init, t_init),
        quat_from_euler(launch_conditions["flight_azimuth_init"], 90.0, 0.0),
    )
    m_init = sum(s["mass_dry"] + s["mass_propellant"] for s in stages.values())
    if settings["OptimizationMode"] != "Payload":
        m_init += settings["mass_payload"]
    x_init = np.hstack((m_init, r_init, v_init, quat_init))

    unit_R = 6378137
    unit_V = 1000.0
    unit_m = m_init
    unit_u = 1.0
    unit_t = pdict["params"][-1]["time"]

    unitdict = {
        "mass": unit_m,
        "position": unit_R,
        "velocity": unit_V,
        "u": unit_u,
        "t": unit_t,
    }

    condition = {**terminal_conditions, **flight_constraint}
    condition["init"] = {
        "mass": m_init,
        "position": r_init,
        "velocity": v_init,
        "quaternion": quat_init,
        "u": np.zeros(2),
    }
    condition["flight_azimuth_init"] = launch_conditions["flight_azimuth_init"]
    condition["OptimizationMode"] = settings["OptimizationMode"]

    # ============================================================
    # Initial guess
    # ============================================================

    if settings.get("Initial trajectory file") is not None:
        x_ref = pd.read_csv(_resolve(settings["Initial trajectory file"]))
        xdict_init = initialize_xdict_from_file(x_ref, pdict, condition, unitdict, False)
    else:
        xdict_init = initialize_xdict_from_simulation(
            x_init, pdict, condition, unitdict, 0.1, False
        )

    # ============================================================
    # Solve
    # ============================================================

    solver_name = settings.get("solver", "IPOPT").upper()

    output_dir = os.path.join(mission_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    if solver_name == "SNOPT":
        solver_options = settings.get("SNOPT_options", {})
        # Note: "Print file" and "Summary file" are Fortran-level options that
        # CasADi's SNOPT plugin cannot forward.  Including them corrupts SNOPT's
        # I/O unit configuration, suppressing all iteration output.
        for _drop_key in ("Print file", "Summary file"):
            solver_options.pop(_drop_key, None)
    else:
        solver_name = "IPOPT"
        solver_options = settings.get("IPOPT_options", {})
        solver_options.setdefault("tol", 1e-6)
        solver_options.setdefault("max_iter", 2000)
        solver_options.setdefault("print_level", 5)
        if "output_file" not in solver_options:
            solver_options["output_file"] = os.path.join(
                output_dir, f"{settings['name']}-IPOPT.out"
            )

    t_start = time.time()
    xStar, solve_stats = build_and_solve(
        pdict,
        condition,
        unitdict,
        xdict_init,
        solver_options,
        solver_name=solver_name.lower(),
        output_dir=output_dir,
    )
    t_elapsed = time.time() - t_start

    # ============================================================
    # Post-processing (same logic as original)
    # ============================================================

    for i, p in enumerate(pdict["params"]):
        p["time"] = xStar["t"][i] * unitdict["t"]

    tu_res = np.array([])
    tx_res = np.array([])
    for i in range(num_sections):
        to = xStar["t"][i]
        tf = xStar["t"][i + 1]
        tau_x = np.hstack((-1.0, pdict["ps_params"].tau(i)))
        tu_res = np.hstack(
            (
                tu_res,
                (pdict["ps_params"].tau(i) * (tf - to) / 2 + (tf + to) / 2) * unitdict["t"],
            )
        )
        tx_res = np.hstack(
            (
                tx_res,
                (tau_x * (tf - to) / 2 + (tf + to) / 2) * unitdict["t"],
            )
        )

    m_res = xStar["mass"] * unitdict["mass"]

    res_info = []
    res_info.append(f"GELATO: GENERIC LAUNCH TRAJECTORY OPTIMIZER v{version}\n\n")
    res_info.append(f"Input file name : {mission_name}\n\n")
    res_info.append(f"initial mass    : {m_res[0]:10.3f} kg\n")
    res_info.append(f"final mass      : {m_res[-1]:10.3f} kg\n")

    mass_drop = sum(item["mass"] for item in settings.get("DropMass", {}).values())
    res_info.append(
        f"payload         : {m_res[0] - m_init - mass_drop:10.3f} kg\n\n"
    )
    res_info.append(f"optTime         : {t_elapsed:11.6f}\n\n")

    stats = solve_stats["stats"]
    converged = stats.get("return_status", "Unknown")
    res_info.append(f"{converged}\n")

    print("".join(res_info[1:]))
    opt_result_path = os.path.join(output_dir, f"{settings['name']}-optResult.txt")
    with open(opt_result_path, mode="w") as fout:
        fout.write("".join(res_info))

    out = output_result(xStar, unitdict, tx_res, tu_res, pdict)
    traj_result_path = os.path.join(output_dir, f"{settings['name']}-trajectoryResult.csv")
    out.to_csv(traj_result_path, index=False)

    print(f"Results saved to {traj_result_path}")


if __name__ == "__main__":
    main()
