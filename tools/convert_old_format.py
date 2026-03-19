#!/usr/bin/env python3
"""Convert old GELATO settings format to new format.

Old format (Before v0.9.0):
  - settings.json  : top-level params + FlightConstraint (AOA/dynP/Q-alpha/waypoint/antenna)
                     RocketStage contains Isp_vac and dropMass
  - events.csv     : one row per section (time, thrust, attitude, etc.)

New format (After v1.0.0):
  - settings.json  : single file with "sections" array (events + per-section constraints)
                     RocketStage without Isp_vac/dropMass; top-level DropMass

Usage:
    python tools/convert_old_format.py <old_settings.json> <output.json>
"""

import csv
import json
import sys
from pathlib import Path


def _num(v):
    """Return int if value is a whole number, else float."""
    f = float(v)
    return int(f) if f == int(f) else f


def convert(old_json_path: Path, out_json_path: Path) -> None:
    with open(old_json_path, encoding="utf-8") as f:
        old = json.load(f)

    # Locate events CSV (relative to the old JSON's directory)
    csv_name = old.get("Event setting file", "")
    if not csv_name:
        raise ValueError('"Event setting file" key not found in old JSON')
    csv_path = old_json_path.parent / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Events CSV not found: {csv_path}")

    with open(csv_path, encoding="utf-8", newline="") as f:
        events = list(csv.DictReader(f))

    # Absolute time lookup: section_name → float seconds
    time_map = {e["name"].strip(): float(e["time"]) for e in events}

    # ── FlightConstraint decomposition ──────────────────────────────────────────
    fc = old.get("FlightConstraint", {})

    aoa_map = fc.get("AOA_max", {})  # {sec: {value, range}}
    dynp_map = fc.get("dynamic_pressure_max", {})
    qa_map = fc.get("Q_alpha_max", {})

    # waypoint: {sec: {wp_key: {min|max|exact: value}}}
    wp_fc = fc.get("waypoint", {})

    # antenna: {station: {lat,lon,altitude,elevation_min:{sec: val}}}
    # → reorganise by section: {sec: {station: {lat,lon,altitude,elevation_min: val}}}
    ant_by_sec: dict = {}
    for stn, sdata in fc.get("antenna", {}).items():
        for sec_name, elev_val in sdata.get("elevation_min", {}).items():
            ant_by_sec.setdefault(sec_name, {})[stn] = {
                "lat": sdata["lat"],
                "lon": sdata["lon"],
                "altitude": sdata["altitude"],
                "elevation_min": elev_val,
            }

    # ── Build sections ───────────────────────────────────────────────────────────
    first_event_name = events[0]["name"].strip() if events else ""
    sections = []
    for ev in events:
        name = ev["name"].strip()
        time_val = float(ev["time"])
        time_ref = ev.get("time_ref", "").strip()
        rocket_stage = ev["rocketStage"].strip()
        engine_on = ev["engineOn"].strip().upper() == "TRUE"
        thrust = _num(ev["thrust"])
        nozzle_area = _num(ev["nozzle_area"])
        attitude = ev["attitude"].strip()
        pitch_rate = _num(ev["pitchrate_init"])
        yaw_rate = _num(ev["yawrate_init"])
        num_nodes = int(ev["num_nodes"])

        isp_vac = _num(old["RocketStage"][rocket_stage].get("Isp_vac", 0))

        # ── Time constraint ──────────────────────────────────────────────────────
        # time_ref == ''                → free
        # time_ref == first_event_name  → fixed (絶対時刻); 最初のイベント自身は reference point = null
        # time_ref == other             → relative (value = time - time_of_ref)
        if not time_ref:
            tc = {"mode": "free", "value": None, "reference point": None}
        elif time_ref == first_event_name:
            ref_point = None if name == first_event_name else first_event_name
            tc = {"mode": "fixed", "value": _num(time_val), "reference point": ref_point}
        else:
            rel = time_val - time_map[time_ref]
            tc = {"mode": "relative", "value": _num(rel), "reference point": time_ref}

        # ── Per-section constraints ──────────────────────────────────────────────
        aoa_con = {"mode": "max", **aoa_map[name]} if name in aoa_map else {}
        dyn_con = {"mode": "max", **dynp_map[name]} if name in dynp_map else {}
        qa_con = {"mode": "max", **qa_map[name]} if name in qa_map else {}

        # waypoint: {wp_key: {min|max|exact: value}} → {wp_key: {mode: ..., value: ...}}
        wp_con = {}
        for wk, wv in wp_fc.get(name, {}).items():
            for mode_key, val in wv.items():
                wp_con[wk] = {"mode": mode_key, "value": val}

        sections.append(
            {
                "name": name,
                "num_nodes": num_nodes,
                "rocket_stage": rocket_stage,
                "engine_mode": "full" if engine_on else "off",
                "thrust_vac": thrust,
                "Isp_vac": isp_vac,
                "throttle": [1, 1] if engine_on else [0, 0],
                "nozzle_area": nozzle_area,
                "attitude constraint": attitude,
                "initial guess": {
                    "time": _num(time_val),
                    "pitch_rate": pitch_rate,
                    "yaw_rate": yaw_rate,
                },
                "time constraint": tc,
                "AOA constraint": aoa_con,
                "dynamic pressure constraint": dyn_con,
                "Q-alpha constraint": qa_con,
                "waypoint constraint": wp_con,
                "antenna constraint": ant_by_sec.get(name, {}),
            }
        )

    # ── RocketStage: remove Isp_vac / dropMass; collect DropMass ────────────────
    rocket_stage_new = {}
    drop_mass = {}
    first_stage = True
    for sk, sv in sorted(old["RocketStage"].items()):
        stage_data = {
            k: v for k, v in sv.items() if k not in ("Isp_vac", "dropMass")
        }
        if not first_stage:
            stage_data.setdefault("aero_enabled", False)
        first_stage = False
        rocket_stage_new[sk] = stage_data
        for dm_name, dm_val in sv.get("dropMass", {}).items():
            drop_mass[dm_name] = dm_val

    # ── Solver options ───────────────────────────────────────────────────────────
    ipopt_options = old.get("IPOPT") or old.get("IPOPT_")
    snopt_options = old.get("SNOPT") or old.get("SNOPT_")
    # drop incompatible options (e.g. "Start"in SNOPT, "linear_solver" in IPOPT)
    if snopt_options:
        snopt_options = {
            k: v
            for k, v in snopt_options.items()
            if k not in ("Start", "Elastic weight", "Scale option")
        }
    if ipopt_options:
        ipopt_options = {k: v for k, v in ipopt_options.items() if k != "linear_solver"}
    solver = "SNOPT" if snopt_options else "IPOPT"

    # ── Assemble new JSON ────────────────────────────────────────────────────────
    new_data: dict = {
        "name": old.get("name", ""),
        "Wind file": old.get("Wind file", ""),
        "CA file": old.get("CA file", ""),
        "Initial trajectory file": old.get("Initial trajectory file", ""),
        "OptimizationMode": old.get("OptimizationMode", "Payload"),
        "RocketStage": rocket_stage_new,
        "DropMass": drop_mass,
        "mass_payload": old.get("mass_payload", 0.0),
        "LaunchCondition": old["LaunchCondition"],
        "TerminalCondition": old["TerminalCondition"],
        "solver": solver,
    }
    if ipopt_options:
        new_data["IPOPT_options"] = ipopt_options
    if snopt_options:
        new_data["SNOPT_options"] = snopt_options
    new_data["sections"] = sections

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Written : {out_json_path}")
    print(f'Sections: {len(sections)} ({", ".join(s["name"] for s in sections)})')


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <old_settings.json> <output.json>")
        sys.exit(1)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
