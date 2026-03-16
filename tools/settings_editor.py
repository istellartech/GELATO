#!/usr/bin/env python3
"""GELATO Settings JSON Editor - Web GUI.

Usage:
    python tools/settings_editor.py [--port PORT]

Opens a browser-based editor for creating and editing GELATO settings JSON files.
"""

import json
import os
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)
app.json.sort_keys = False

TOOLS_DIR = Path(__file__).resolve().parent
BASE_DIR = str(TOOLS_DIR.parent)

# ── Templates ────────────────────────────────────────────────────────────────

SECTION_TEMPLATE = {
    "name": "NEW_SECTION",
    "num_nodes": 5,
    "rocket_stage": "1",
    "engine_mode": "full",
    "thrust_vac": 0,
    "Isp_vac": 300,
    "throttle": [1.0, 1.0],
    "nozzle_area": 0,
    "attitude constraint": "pitch-yaw",
    "initial guess": {"time": 0.0, "pitch_rate": 0.0, "yaw_rate": 0.0},
    "time constraint": {"mode": "free", "value": None, "reference point": None},
    "AOA constraint": {},
    "dynamic pressure constraint": {},
    "Q-alpha constraint": {},
    "waypoint constraint": {},
    "antenna constraint": {},
}

DEFAULT_TEMPLATE = {
    "name": "new_project",
    "Wind file": "",
    "CA file": "",
    "Initial trajectory file": "",
    "OptimizationMode": "Payload",
    "RocketStage": {
        "1": {
            "mass_dry": 1000.0,
            "mass_propellant": 10000.0,
            "reference_area": 1.0,
            "ignition_at": "LIFTOFF",
            "cutoff_at": "MECO",
            "separation_at": "SEP1",
        }
    },
    "DropMass": {},
    "mass_payload": 0.0,
    "LaunchCondition": {
        "lon": 0.0,
        "lat": 0.0,
        "altitude": 0.0,
        "flight_azimuth_init": 90.0,
    },
    "TerminalCondition": {
        "altitude_perigee": 200000,
        "altitude_apogee": 200000,
        "inclination": None,
        "radius": None,
        "vel_tangential_geocentric": None,
        "vel_radial_geocentric": None,
        "flightpath_vel_inertial_geocentric": None,
    },
    "solver": "IPOPT",
    "IPOPT_options": {
        "linear_solver": "mumps",
        "tol": 1e-6,
        "acceptable_tol": 1e-4,
        "max_iter": 2000,
    },
    "SNOPT_options": {
        "Start": "Cold",
        "Time limit": 1200,
        "Major feasibility tolerance": 1e-6,
        "Minor feasibility tolerance": 1e-6,
        "Major optimality tolerance": 1e-6,
        "Scale option": 0,
    },
    "sections": [
        {
            "name": "LIFTOFF",
            "num_nodes": 5,
            "rocket_stage": "1",
            "engine_mode": "full",
            "thrust_vac": 100000,
            "Isp_vac": 300,
            "throttle": [1.0, 1.0],
            "nozzle_area": 0.5,
            "attitude constraint": "vertical",
            "initial guess": {"time": 0.0, "pitch_rate": 0.0, "yaw_rate": 0.0},
            "time constraint": {"mode": "fixed", "value": 0.0, "reference point": None},
            "AOA constraint": {},
            "dynamic pressure constraint": {},
            "Q-alpha constraint": {},
            "waypoint constraint": {},
            "antenna constraint": {},
        },
        {
            "name": "MECO",
            "num_nodes": 2,
            "rocket_stage": "1",
            "engine_mode": "off",
            "thrust_vac": 0,
            "Isp_vac": 300,
            "throttle": [0.0, 0.0],
            "nozzle_area": 0,
            "attitude constraint": "hold",
            "initial guess": {"time": 100.0, "pitch_rate": 0.0, "yaw_rate": 0.0},
            "time constraint": {"mode": "free", "value": None, "reference point": None},
            "AOA constraint": {},
            "dynamic pressure constraint": {},
            "Q-alpha constraint": {},
            "waypoint constraint": {},
            "antenna constraint": {},
        },
        {
            "name": "SEP1",
            "num_nodes": 2,
            "rocket_stage": "1",
            "engine_mode": "off",
            "thrust_vac": 0,
            "Isp_vac": 300,
            "throttle": [0.0, 0.0],
            "nozzle_area": 0,
            "attitude constraint": "hold",
            "initial guess": {"time": 105.0, "pitch_rate": 0.0, "yaw_rate": 0.0},
            "time constraint": {
                "mode": "relative",
                "value": 5.0,
                "reference point": "MECO",
            },
            "AOA constraint": {},
            "dynamic pressure constraint": {},
            "Q-alpha constraint": {},
            "waypoint constraint": {},
            "antenna constraint": {},
        },
    ],
}


# ── Validation ───────────────────────────────────────────────────────────────

VALID_ATTITUDE = [
    "vertical",
    "kick-turn",
    "pitch",
    "pitch-yaw",
    "same-rate",
    "zero-lift-turn",
    "hold",
    "free",
]
VALID_ENGINE_MODE = ["full", "off"]
VALID_TIME_MODE = ["fixed", "free", "relative"]
VALID_WP_KEY = ["lat", "lon", "altitude", "lat_IIP", "lon_IIP", "downrange"]
SUPPORTED_WP_KEY = ["lat", "lon", "altitude", "lat_IIP", "lon_IIP"]
LEGACY_WP_KEY = ["downrange"]
VALID_WP_MODE = ["exact", "min", "max"]


def validate_settings(data):
    """Validate a settings JSON dict. Returns list of {level, path, message}."""
    issues = []

    def err(path, msg):
        issues.append({"level": "error", "path": path, "message": msg})

    def warn(path, msg):
        issues.append({"level": "warning", "path": path, "message": msg})

    if not isinstance(data, dict):
        err("", "Root must be a JSON object")
        return issues

    # Required top-level keys
    required = [
        "name",
        "OptimizationMode",
        "RocketStage",
        "mass_payload",
        "LaunchCondition",
        "TerminalCondition",
        "solver",
        "sections",
    ]
    for key in required:
        if key not in data:
            err(key, f"Required key '{key}' is missing")

    # Solver
    solver = data.get("solver")
    if solver not in ("IPOPT", "SNOPT"):
        err("solver", "Must be 'IPOPT' or 'SNOPT'")
    if solver == "IPOPT" and "IPOPT_options" not in data:
        err("IPOPT_options", "IPOPT_options required when solver is IPOPT")
    if solver == "SNOPT" and "SNOPT_options" not in data:
        err("SNOPT_options", "SNOPT_options required when solver is SNOPT")

    # Collect section names
    sections = data.get("sections", [])
    section_names = [s.get("name", "") for s in sections if isinstance(s, dict)]

    if len(sections) < 2:
        err("sections", "At least 2 sections required")

    # Check section name uniqueness
    seen_names = set()
    for i, name in enumerate(section_names):
        if not name:
            err(f"sections[{i}].name", "Section name cannot be empty")
        elif name in seen_names:
            err(f"sections[{i}].name", f"Duplicate section name '{name}'")
        seen_names.add(name)

    # RocketStage validation
    stages = data.get("RocketStage", {})
    stage_keys = set(stages.keys()) if isinstance(stages, dict) else set()
    if isinstance(stages, dict):
        for sk, sv in stages.items():
            if not isinstance(sv, dict):
                continue
            for ref_key in ("ignition_at", "cutoff_at", "separation_at"):
                ref_val = sv.get(ref_key, "")
                if ref_val and ref_val not in section_names:
                    err(
                        f"RocketStage.{sk}.{ref_key}",
                        f"References section '{ref_val}' which does not exist",
                    )

    # DropMass validation
    drop_mass = data.get("DropMass", {})
    if isinstance(drop_mass, dict):
        for dm_name, dm_val in drop_mass.items():
            if not isinstance(dm_val, dict):
                continue
            sep = dm_val.get("separation_at", "")
            if sep and sep not in section_names:
                err(
                    f"DropMass.{dm_name}.separation_at",
                    f"References section '{sep}' which does not exist",
                )

    # LaunchCondition
    lc = data.get("LaunchCondition", {})
    if isinstance(lc, dict):
        for fld in ("lon", "lat", "altitude", "flight_azimuth_init"):
            if fld not in lc:
                err(f"LaunchCondition.{fld}", f"Required field '{fld}' missing")

    # Validate each section
    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            err(f"sections[{i}]", "Section must be an object")
            continue
        prefix = f"sections[{i}]"

        # rocket_stage
        rs = sec.get("rocket_stage", "")
        if rs not in stage_keys:
            err(f"{prefix}.rocket_stage", f"Stage '{rs}' not in RocketStage")

        # engine_mode
        em = sec.get("engine_mode", "")
        if em not in VALID_ENGINE_MODE:
            err(f"{prefix}.engine_mode", f"Invalid engine_mode '{em}'")

        # attitude constraint
        att = sec.get("attitude constraint", "")
        if att not in VALID_ATTITUDE:
            err(f"{prefix}.attitude constraint", f"Invalid attitude '{att}'")

        # throttle
        thr = sec.get("throttle", [])
        if (
            not isinstance(thr, list)
            or len(thr) != 2
            or not all(isinstance(v, (int, float)) for v in thr)
        ):
            err(f"{prefix}.throttle", "Must be [min, max] array of 2 numbers")
        elif thr[0] > thr[1]:
            err(f"{prefix}.throttle", "throttle[0] must be <= throttle[1]")

        # time constraint
        tc = sec.get("time constraint", {})
        if isinstance(tc, dict) and tc:
            tc_mode = tc.get("mode", "")
            if tc_mode not in VALID_TIME_MODE:
                err(f"{prefix}.time constraint.mode", f"Invalid mode '{tc_mode}'")
            if tc_mode in ("fixed", "relative"):
                ref = tc.get("reference point")
                if ref is not None and ref not in section_names:
                    err(
                        f"{prefix}.time constraint.reference point",
                        f"References section '{ref}' which does not exist",
                    )

        # waypoint constraint
        wp = sec.get("waypoint constraint", {})
        if isinstance(wp, dict):
            for wk, wv in wp.items():
                if wk not in VALID_WP_KEY:
                    warn(f"{prefix}.waypoint constraint.{wk}", f"Unusual key '{wk}'")
                elif wk in LEGACY_WP_KEY:
                    warn(
                        f"{prefix}.waypoint constraint.{wk}",
                        f"'{wk}' is not supported by the current CasADi solver",
                    )
                if isinstance(wv, dict):
                    wm = wv.get("mode", "")
                    if wm not in VALID_WP_MODE:
                        err(
                            f"{prefix}.waypoint constraint.{wk}.mode",
                            f"Invalid mode '{wm}'",
                        )

        # antenna constraint
        ant = sec.get("antenna constraint", {})
        if isinstance(ant, dict):
            for aname, aval in ant.items():
                if isinstance(aval, dict):
                    for af in ("lat", "lon", "altitude", "elevation_min"):
                        if af not in aval:
                            err(
                                f"{prefix}.antenna constraint.{aname}.{af}",
                                f"Missing '{af}'",
                            )

    return issues


# ── Flask Routes ─────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory(TOOLS_DIR / "static", "editor.html")


@app.route("/api/template")
def api_template():
    return jsonify(DEFAULT_TEMPLATE)


@app.route("/api/section_template")
def api_section_template():
    return jsonify(SECTION_TEMPLATE)


@app.route("/api/load", methods=["POST"])
def api_load():
    body = request.get_json()
    fpath = body.get("path", "")
    if not os.path.isabs(fpath):
        fpath = os.path.join(BASE_DIR, fpath)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({"ok": True, "data": data, "path": fpath})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


def _order_settings(data):
    """Reorder settings keys to match the GUI tab/field order."""
    if not isinstance(data, dict):
        return data

    TOP_KEY_ORDER = [
        "name", "Wind file", "CA file", "Initial trajectory file",
        "OptimizationMode", "RocketStage", "DropMass", "mass_payload",
        "LaunchCondition", "TerminalCondition", "solver",
        "IPOPT_options", "SNOPT_options", "sections",
    ]
    STAGE_KEY_ORDER = [
        "mass_dry", "mass_propellant", "reference_area",
        "ignition_at", "cutoff_at", "separation_at",
    ]
    LC_KEY_ORDER = ["lon", "lat", "altitude", "flight_azimuth_init"]
    TC_KEY_ORDER = [
        "altitude_perigee", "altitude_apogee", "inclination", "radius",
        "vel_tangential_geocentric", "vel_radial_geocentric",
        "flightpath_vel_inertial_geocentric",
    ]
    IPOPT_KEY_ORDER = ["linear_solver", "tol", "acceptable_tol", "max_iter"]
    SNOPT_KEY_ORDER = [
        "Start", "Time limit",
        "Major feasibility tolerance", "Minor feasibility tolerance",
        "Major optimality tolerance", "Scale option",
    ]
    SECTION_KEY_ORDER = [
        "name", "num_nodes", "rocket_stage", "engine_mode",
        "thrust_vac", "Isp_vac", "throttle", "nozzle_area",
        "attitude constraint", "initial guess", "time constraint",
        "AOA constraint", "dynamic pressure constraint", "Q-alpha constraint",
        "waypoint constraint", "antenna constraint",
    ]
    IG_KEY_ORDER = ["time", "pitch_rate", "yaw_rate"]
    TIME_CON_KEY_ORDER = ["mode", "value", "reference point"]

    def _ordered(d, key_order):
        ordered = {}
        for k in key_order:
            if k in d:
                ordered[k] = d[k]
        for k in d:
            if k not in ordered:
                ordered[k] = d[k]
        return ordered

    result = _ordered(data, TOP_KEY_ORDER)

    if isinstance(result.get("RocketStage"), dict):
        result["RocketStage"] = {
            k: _ordered(v, STAGE_KEY_ORDER) if isinstance(v, dict) else v
            for k, v in result["RocketStage"].items()
        }

    if isinstance(result.get("DropMass"), dict):
        result["DropMass"] = {
            k: _ordered(v, ["mass", "separation_at"]) if isinstance(v, dict) else v
            for k, v in result["DropMass"].items()
        }

    if isinstance(result.get("LaunchCondition"), dict):
        result["LaunchCondition"] = _ordered(result["LaunchCondition"], LC_KEY_ORDER)

    if isinstance(result.get("TerminalCondition"), dict):
        result["TerminalCondition"] = _ordered(result["TerminalCondition"], TC_KEY_ORDER)

    if isinstance(result.get("IPOPT_options"), dict):
        result["IPOPT_options"] = _ordered(result["IPOPT_options"], IPOPT_KEY_ORDER)

    if isinstance(result.get("SNOPT_options"), dict):
        result["SNOPT_options"] = _ordered(result["SNOPT_options"], SNOPT_KEY_ORDER)

    if isinstance(result.get("sections"), list):
        ordered_sections = []
        for sec in result["sections"]:
            if not isinstance(sec, dict):
                ordered_sections.append(sec)
                continue
            sec = _ordered(sec, SECTION_KEY_ORDER)
            if isinstance(sec.get("initial guess"), dict):
                sec["initial guess"] = _ordered(sec["initial guess"], IG_KEY_ORDER)
            if isinstance(sec.get("time constraint"), dict):
                sec["time constraint"] = _ordered(
                    sec["time constraint"], TIME_CON_KEY_ORDER
                )
            ordered_sections.append(sec)
        result["sections"] = ordered_sections

    return result


@app.route("/api/save", methods=["POST"])
def api_save():
    body = request.get_json()
    fpath = body.get("path", "")
    data = body.get("data")
    if not os.path.isabs(fpath):
        fpath = os.path.join(BASE_DIR, fpath)
    try:
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        ordered = _order_settings(data)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False)
            f.write("\n")
        return jsonify({"ok": True, "path": fpath})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/validate", methods=["POST"])
def api_validate():
    body = request.get_json()
    data = body.get("data", {})
    issues = validate_settings(data)
    return jsonify({"issues": issues})


@app.route("/api/browse")
def api_browse():
    dir_path = request.args.get("dir", BASE_DIR)
    if not os.path.isabs(dir_path):
        dir_path = os.path.join(BASE_DIR, dir_path)
    try:
        entries = []
        for name in sorted(os.listdir(dir_path)):
            full = os.path.join(dir_path, name)
            if name.startswith("."):
                continue
            if os.path.isdir(full):
                entries.append({"name": name, "type": "dir"})
            elif name.endswith(".json"):
                entries.append({"name": name, "type": "file"})
        return jsonify({"ok": True, "dir": dir_path, "entries": entries})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    port = 5555
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])

    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    print(f"Starting GELATO Settings Editor on http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
