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

from flask import Flask, jsonify, request

app = Flask(__name__)

BASE_DIR = str(Path(__file__).resolve().parent.parent)

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
  "vertical", "kick-turn", "pitch", "pitch-yaw", "same-rate",
    "zero-lift-turn", "hold", "free",
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
        "name", "OptimizationMode", "RocketStage", "mass_payload",
        "LaunchCondition", "TerminalCondition", "solver", "sections",
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
    return HTML_PAGE


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

    # Top-level key order matching GUI tabs:
    # General -> Stages & Mass -> Launch -> Terminal -> Solver -> Sections
    TOP_KEY_ORDER = [
        "name", "Wind file", "CA file", "Initial trajectory file",
        "OptimizationMode",
        "RocketStage", "DropMass", "mass_payload",
        "LaunchCondition",
        "TerminalCondition",
        "solver", "IPOPT_options", "SNOPT_options",
        "sections",
    ]

    SECTION_KEY_ORDER = [
        "name", "num_nodes", "rocket_stage", "engine_mode",
        "thrust_vac", "Isp_vac", "throttle", "nozzle_area",
        "attitude constraint",
        "initial guess",
        "time constraint",
        "AOA constraint", "dynamic pressure constraint", "Q-alpha constraint",
        "waypoint constraint",
        "antenna constraint",
    ]

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

    if "sections" in result and isinstance(result["sections"], list):
        result["sections"] = [
            _ordered(sec, SECTION_KEY_ORDER)
            if isinstance(sec, dict) else sec
            for sec in result["sections"]
        ]

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
            json.dump(ordered, f, indent=4, ensure_ascii=False)
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


# ── HTML Page ────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GELATO Settings Editor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;font-size:14px;background:#f0f2f5;color:#1a1a1a}
#toolbar{background:#1e293b;color:#fff;padding:8px 16px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
#toolbar h1{font-size:16px;font-weight:600;margin-right:16px}
#toolbar button{background:#334155;color:#fff;border:1px solid #475569;padding:5px 12px;border-radius:4px;cursor:pointer;font-size:13px}
#toolbar button:hover{background:#475569}
#toolbar .filepath{margin-left:auto;font-size:12px;color:#94a3b8;max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tab-bar{display:flex;background:#e2e8f0;border-bottom:2px solid #cbd5e1}
.tab-bar button{padding:8px 16px;border:none;background:transparent;cursor:pointer;font-size:13px;font-weight:500;color:#475569;border-bottom:2px solid transparent;margin-bottom:-2px}
.tab-bar button.active{color:#1e40af;border-bottom-color:#1e40af;background:#fff}
.tab-bar button:hover{background:#f1f5f9}
#content{padding:16px;max-width:1200px;margin:0 auto}
.card{background:#fff;border-radius:8px;padding:16px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.card h3{font-size:14px;color:#1e40af;margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid #e2e8f0}
.field{display:flex;align-items:center;margin-bottom:8px;gap:8px}
.field label{min-width:180px;font-size:13px;color:#475569;text-align:right}
.field input,.field select{padding:5px 8px;border:1px solid #d1d5db;border-radius:4px;font-size:13px;font-family:inherit}
.field input[type="number"]{font-family:"SF Mono",Consolas,monospace;width:140px}
.field input[type="text"]{width:280px}
.field .unit{font-size:12px;color:#94a3b8;min-width:40px}
.field input:focus,.field select:focus{outline:none;border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,.2)}
.field input.invalid{border-color:#ef4444}
.sections-layout{display:flex;gap:16px;min-height:500px}
.section-list{width:220px;flex-shrink:0}
.section-list .list{border:1px solid #d1d5db;border-radius:6px;background:#fff;overflow-y:auto;max-height:calc(100vh - 220px)}
.section-list .list-item{padding:6px 8px;cursor:pointer;display:flex;align-items:center;gap:6px;border-bottom:1px solid #f1f5f9;font-size:13px}
.section-list .list-item:hover{background:#f1f5f9}
.section-list .list-item.active{background:#dbeafe;color:#1e40af;font-weight:600}
.section-list .list-item .grip{cursor:grab;color:#94a3b8;font-size:16px;user-select:none}
.section-list .list-item .name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.section-list .list-item .del-btn{color:#94a3b8;cursor:pointer;font-size:16px;padding:0 2px;border:none;background:none}
.section-list .list-item .del-btn:hover{color:#ef4444}
.section-list .btns{display:flex;gap:4px;margin-top:6px}
.section-list .btns button{flex:1;padding:5px;font-size:12px;border:1px solid #d1d5db;border-radius:4px;background:#fff;cursor:pointer}
.section-list .btns button:hover{background:#f1f5f9}
.section-detail{flex:1;min-width:0}
.constraint-toggle{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.constraint-toggle input[type="checkbox"]{width:16px;height:16px}
.constraint-block{border:1px solid #e2e8f0;border-radius:6px;padding:12px;margin-bottom:8px;background:#fafafa}
.constraint-block.disabled{opacity:.4;pointer-events:none}
.constraint-block h4{font-size:13px;color:#475569;margin-bottom:8px}
.wp-row,.ant-row{display:flex;align-items:center;gap:6px;margin-bottom:6px;flex-wrap:wrap}
.wp-row select,.wp-row input,.ant-row input,.ant-row select{padding:4px 6px;border:1px solid #d1d5db;border-radius:4px;font-size:12px}
.wp-row button,.ant-row button{padding:3px 8px;font-size:12px;border:1px solid #d1d5db;border-radius:4px;background:#fff;cursor:pointer}
.wp-row button:hover,.ant-row button:hover{background:#fee2e2;border-color:#fca5a5}
.add-btn{padding:4px 10px;font-size:12px;border:1px solid #d1d5db;border-radius:4px;background:#fff;cursor:pointer;color:#1e40af}
.add-btn:hover{background:#dbeafe}
#statusbar{background:#1e293b;color:#94a3b8;padding:6px 16px;font-size:12px;max-height:120px;overflow-y:auto}
#statusbar .issue{padding:2px 0;cursor:pointer}
#statusbar .issue:hover{color:#fff}
#statusbar .issue.error{color:#fca5a5}
#statusbar .issue.warning{color:#fcd34d}
#statusbar .ok{color:#86efac}
.modal-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.5);display:flex;align-items:center;justify-content:center;z-index:100}
.modal{background:#fff;border-radius:8px;padding:20px;min-width:500px;max-width:700px;max-height:80vh;overflow-y:auto;box-shadow:0 10px 40px rgba(0,0,0,.3)}
.modal h2{font-size:16px;margin-bottom:12px}
.modal .browse-path{font-size:12px;color:#64748b;margin-bottom:8px;word-break:break-all}
.modal .browse-list{border:1px solid #d1d5db;border-radius:4px;max-height:300px;overflow-y:auto}
.modal .browse-item{padding:6px 10px;cursor:pointer;display:flex;align-items:center;gap:6px;font-size:13px;border-bottom:1px solid #f1f5f9}
.modal .browse-item:hover{background:#f1f5f9}
.modal .browse-item .icon{width:20px;text-align:center}
.modal .modal-btns{display:flex;gap:8px;margin-top:12px;justify-content:flex-end}
.modal .modal-btns button{padding:6px 16px;border-radius:4px;cursor:pointer;font-size:13px}
.modal .modal-btns .primary{background:#1e40af;color:#fff;border:none}
.modal .modal-btns .primary:hover{background:#1e3a8a}
.modal .modal-btns .secondary{background:#fff;border:1px solid #d1d5db}
.modal input[type="text"]{width:100%;padding:6px 8px;border:1px solid #d1d5db;border-radius:4px;font-size:13px;margin-bottom:8px}
.stage-table,.dm-table{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:8px}
.stage-table th,.dm-table th{text-align:left;padding:4px 8px;background:#f1f5f9;border:1px solid #e2e8f0;font-weight:500}
.stage-table td,.dm-table td{padding:4px 8px;border:1px solid #e2e8f0}
.stage-table input,.dm-table input,.stage-table select,.dm-table select{width:100%;padding:3px 4px;border:1px solid #d1d5db;border-radius:3px;font-size:12px}
.dragging{opacity:0.4}
.drag-over{border-top:2px solid #3b82f6}
</style>
</head>
<body>
<div id="toolbar">
  <h1>GELATO Settings Editor</h1>
  <button onclick="newFile()">New</button>
  <button onclick="openFile()">Open</button>
  <button onclick="saveFile()">Save</button>
  <button onclick="saveFileAs()">Save As</button>
  <button onclick="validateFile()">Validate</button>
  <span class="filepath" id="filepath-display">No file loaded</span>
</div>
<div class="tab-bar" id="tab-bar"></div>
<div id="content"></div>
<div id="statusbar"></div>
<div id="modal-root"></div>

<script>
// ─── State ─────────────────────────────────────────────────────────────
const TABS = [
  {id:'general', label:'General'},
  {id:'stages', label:'Stages & Mass'},
  {id:'launch', label:'Launch'},
  {id:'terminal', label:'Terminal'},
  {id:'solver', label:'Solver'},
  {id:'sections', label:'Sections'},
];

const ATTITUDE_OPTIONS = ['vertical','kick-turn','pitch-yaw','same-rate','hold','free'];
const ENGINE_MODE_OPTIONS = ['full','off'];
const TIME_MODE_OPTIONS = ['fixed','free','relative'];
const WP_KEYS = ['lat','lon','altitude','lat_IIP','lon_IIP'];
const LEGACY_WP_KEYS = ['downrange'];
const WP_MODES = ['exact','min','max'];

let state = {
  data: null,
  filePath: null,
  dirty: false,
  activeTab: 'general',
  selectedSection: 0,
};

// ─── Helpers ───────────────────────────────────────────────────────────
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) Object.entries(attrs).forEach(([k,v]) => {
    if (v == null || v === undefined) return;
    if (k === 'className') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k === 'disabled') { if (v) el.disabled = true; }
    else el.setAttribute(k, v);
  });
  children.flat(Infinity).forEach(c => {
    if (c == null) return;
    el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  });
  return el;
}

function getSectionNames() {
  if (!state.data || !state.data.sections) return [];
  return state.data.sections.map(s => s.name || '');
}

function getStageKeys() {
  if (!state.data || !state.data.RocketStage) return [];
  return Object.keys(state.data.RocketStage);
}

function markDirty() { state.dirty = true; }

function numInput(value, onChange, opts={}) {
  if (opts.disabled) opts.disabled = true; else delete opts.disabled;
  const inp = h('input', {type:'number', value: value == null ? '' : value, ...opts});
  inp.addEventListener('change', e => {
    const v = e.target.value === '' ? null : parseFloat(e.target.value);
    onChange(v);
    markDirty();
  });
  return inp;
}

function textInput(value, onChange, opts={}) {
  const inp = h('input', {type:'text', value: value || '', ...opts});
  inp.addEventListener('change', e => { onChange(e.target.value); markDirty(); });
  return inp;
}

function selectInput(value, options, onChange) {
  const sel = h('select', {}, options.map(o =>
    h('option', {value:o, ...(o===value?{selected:'true'}:{})}, o)
  ));
  sel.addEventListener('change', e => { onChange(e.target.value); markDirty(); });
  return sel;
}

function field(label, ...inputs) {
  return h('div', {className:'field'},
    h('label', {}, label),
    ...inputs
  );
}

// ─── Tab Rendering ─────────────────────────────────────────────────────
function renderTabBar() {
  const bar = document.getElementById('tab-bar');
  bar.innerHTML = '';
  TABS.forEach(tab => {
    const btn = h('button', {
      className: tab.id === state.activeTab ? 'active' : '',
      onClick: () => { state.activeTab = tab.id; render(); }
    }, tab.label);
    bar.appendChild(btn);
  });
}

function render() {
  if (!state.data) return;
  renderTabBar();
  const content = document.getElementById('content');
  content.innerHTML = '';
  const renderers = {
    general: renderGeneral,
    stages: renderStages,
    launch: renderLaunch,
    terminal: renderTerminal,
    solver: renderSolver,
    sections: renderSections,
  };
  const fn = renderers[state.activeTab];
  if (fn) content.appendChild(fn());

  document.getElementById('filepath-display').textContent =
    state.filePath ? (state.dirty ? '* ' : '') + state.filePath : 'No file loaded';
}

// ─── General Tab ───────────────────────────────────────────────────────
function renderGeneral() {
  const d = state.data;
  return h('div', {},
    h('div', {className:'card'},
      h('h3', {}, 'General Settings'),
      field('Name', textInput(d.name, v => d.name = v)),
      field('Wind file', textInput(d['Wind file'], v => d['Wind file'] = v)),
      field('CA file', textInput(d['CA file'], v => d['CA file'] = v)),
      field('Initial trajectory file', textInput(d['Initial trajectory file'], v => d['Initial trajectory file'] = v)),
      field('Optimization Mode',
        selectInput(d.OptimizationMode, ['Payload','MinFuel'], v => d.OptimizationMode = v)),
    )
  );
}

// ─── Stages Tab ────────────────────────────────────────────────────────
function renderStages() {
  const d = state.data;
  const stages = d.RocketStage || {};
  const snames = getSectionNames();

  const stageCards = Object.entries(stages).map(([key, stage]) => {
    function sf(field) { return (v) => { stage[field] = v; markDirty(); }; }
    function sfSelect(field) { return (v) => { stage[field] = v; markDirty(); }; }
    return h('div', {className:'card'},
      h('h3', {}, `Stage ${key}`,
        h('button', {style:{marginLeft:'12px',fontSize:'12px',color:'#ef4444',background:'none',border:'1px solid #fca5a5',borderRadius:'4px',padding:'2px 8px',cursor:'pointer'},
          onClick:() => { delete d.RocketStage[key]; markDirty(); render(); }
        }, 'Remove')
      ),
      field('mass_dry', numInput(stage.mass_dry, sf('mass_dry')), h('span',{className:'unit'},'kg')),
      field('mass_propellant', numInput(stage.mass_propellant, sf('mass_propellant')), h('span',{className:'unit'},'kg')),
      field('reference_area', numInput(stage.reference_area, sf('reference_area'),{step:'0.01'}), h('span',{className:'unit'},'m\u00B2')),
      field('ignition_at', selectInput(stage.ignition_at, [''].concat(snames), sfSelect('ignition_at'))),
      field('cutoff_at', selectInput(stage.cutoff_at, [''].concat(snames), sfSelect('cutoff_at'))),
      field('separation_at', selectInput(stage.separation_at, [''].concat(snames), sfSelect('separation_at'))),
    );
  });

  const addStageBtn = h('button', {className:'add-btn', onClick:() => {
    const keys = Object.keys(d.RocketStage).map(Number).filter(n=>!isNaN(n));
    const next = keys.length ? Math.max(...keys) + 1 : 1;
    d.RocketStage[String(next)] = {
      mass_dry:0, mass_propellant:0, reference_area:0,
      ignition_at:'', cutoff_at:'', separation_at:''
    };
    markDirty(); render();
  }}, '+ Add Stage');

  // DropMass
  const dm = d.DropMass || {};
  const dmRows = Object.entries(dm).map(([name, val]) =>
    h('tr', {},
      h('td', {}, textInput(name, newName => {
        const data = d.DropMass[name];
        delete d.DropMass[name];
        d.DropMass[newName] = data;
        markDirty();
      }, {style:'width:100%'})),
      h('td', {}, numInput(val.mass, v => { val.mass = v; markDirty(); })),
      h('td', {}, selectInput(val.separation_at, [''].concat(snames), v => { val.separation_at = v; markDirty(); })),
      h('td', {}, h('button', {onClick:()=>{ delete d.DropMass[name]; markDirty(); render(); }, style:{color:'#ef4444',background:'none',border:'none',cursor:'pointer',fontSize:'14px'}}, '\u00D7'))
    )
  );

  const addDmBtn = h('button', {className:'add-btn', onClick:() => {
    if (!d.DropMass) d.DropMass = {};
    let name = 'new_mass';
    let i = 1;
    while (d.DropMass[name]) name = `new_mass_${i++}`;
    d.DropMass[name] = {mass:0, separation_at:''};
    markDirty(); render();
  }}, '+ Add DropMass');

  return h('div', {},
    ...stageCards,
    addStageBtn,
    h('div', {className:'card', style:{marginTop:'12px'}},
      h('h3', {}, 'Drop Mass'),
      h('table', {className:'dm-table'},
        h('thead', {}, h('tr', {},
          h('th', {}, 'Name'), h('th', {}, 'Mass (kg)'), h('th', {}, 'Separation At'), h('th', {}, '')
        )),
        h('tbody', {}, ...dmRows)
      ),
      addDmBtn,
    ),
    h('div', {className:'card', style:{marginTop:'12px'}},
      h('h3', {}, 'Payload'),
      field('mass_payload', numInput(d.mass_payload, v => { d.mass_payload = v; markDirty(); }), h('span',{className:'unit'},'kg'))
    ),
  );
}

// ─── Launch Tab ────────────────────────────────────────────────────────
function renderLaunch() {
  const lc = state.data.LaunchCondition || {};
  function sf(f) { return v => { lc[f] = v; markDirty(); }; }
  return h('div', {},
    h('div', {className:'card'},
      h('h3', {}, 'Launch Condition'),
      field('Longitude', numInput(lc.lon, sf('lon'),{step:'0.0001'}), h('span',{className:'unit'},'deg')),
      field('Latitude', numInput(lc.lat, sf('lat'),{step:'0.0001'}), h('span',{className:'unit'},'deg')),
      field('Altitude', numInput(lc.altitude, sf('altitude')), h('span',{className:'unit'},'m')),
      field('Flight Azimuth', numInput(lc.flight_azimuth_init, sf('flight_azimuth_init'),{step:'0.1'}), h('span',{className:'unit'},'deg')),
    )
  );
}

// ─── Terminal Tab ──────────────────────────────────────────────────────
function renderTerminal() {
  const tc = state.data.TerminalCondition || {};
  const fields = [
    ['altitude_perigee', 'm'],
    ['altitude_apogee', 'm'],
    ['inclination', 'deg'],
    ['radius', 'm'],
    ['vel_tangential_geocentric', 'm/s'],
    ['vel_radial_geocentric', 'm/s'],
    ['flightpath_vel_inertial_geocentric', 'm/s'],
  ];

  const rows = fields.map(([key, unit]) => {
    const isNull = tc[key] == null;
    const cb = h('input', {type:'checkbox', ...(isNull ? {} : {checked:'true'})});
    cb.addEventListener('change', e => {
      tc[key] = e.target.checked ? 0 : null;
      markDirty(); render();
    });
    return field(key,
      cb,
      numInput(tc[key], v => { tc[key] = v; markDirty(); }, {disabled: isNull, step:'0.1'}),
      h('span', {className:'unit'}, unit)
    );
  });

  return h('div', {},
    h('div', {className:'card'},
      h('h3', {}, 'Terminal Condition'),
      ...rows
    )
  );
}

// ─── Solver Tab ────────────────────────────────────────────────────────
function renderSolver() {
  const d = state.data;
  const solver = d.solver || 'IPOPT';

  const ipopt = d.IPOPT_options || {};
  const snopt = d.SNOPT_options || {};

  function ipf(f) { return v => { if(!d.IPOPT_options) d.IPOPT_options={}; d.IPOPT_options[f]=v; markDirty(); }; }
  function snf(f) { return v => { if(!d.SNOPT_options) d.SNOPT_options={}; d.SNOPT_options[f]=v; markDirty(); }; }

  return h('div', {},
    h('div', {className:'card'},
      h('h3', {}, 'Solver Selection'),
      field('Solver', selectInput(solver, ['IPOPT','SNOPT'], v => { d.solver = v; markDirty(); render(); })),
    ),
    h('div', {className:'card', style:{opacity: solver==='IPOPT'?'1':'.4'}},
      h('h3', {}, 'IPOPT Options'),
      field('linear_solver', selectInput(ipopt.linear_solver||'mumps', ['mumps','pardisomkl','ma27','ma57','ma86','ma97'], ipf('linear_solver'))),
      field('tol', numInput(ipopt.tol, ipf('tol'), {step:'1e-7'})),
      field('acceptable_tol', numInput(ipopt.acceptable_tol, ipf('acceptable_tol'), {step:'1e-6'})),
      field('max_iter', numInput(ipopt.max_iter, ipf('max_iter'), {step:'100'})),
    ),
    h('div', {className:'card', style:{opacity: solver==='SNOPT'?'1':'.4'}},
      h('h3', {}, 'SNOPT Options'),
      field('Time limit', numInput(snopt['Time limit'], snf('Time limit')), h('span',{className:'unit'},'s')),
      field('Major feasibility tol', numInput(snopt['Major feasibility tolerance'], snf('Major feasibility tolerance'), {step:'1e-7'})),
      field('Minor feasibility tol', numInput(snopt['Minor feasibility tolerance'], snf('Minor feasibility tolerance'), {step:'1e-7'})),
      field('Major optimality tol', numInput(snopt['Major optimality tolerance'], snf('Major optimality tolerance'), {step:'1e-7'})),
    ),
  );
}

// ─── Sections Tab ──────────────────────────────────────────────────────
function renderSections() {
  const sections = state.data.sections || [];
  if (state.selectedSection >= sections.length) state.selectedSection = Math.max(0, sections.length - 1);

  return h('div', {className:'sections-layout'},
    renderSectionList(sections),
    sections.length > 0 ? renderSectionDetail(sections[state.selectedSection], state.selectedSection) : h('div',{className:'section-detail card'},'No sections'),
  );
}

function renderSectionList(sections) {
  const list = h('div', {className:'list'});

  sections.forEach((sec, i) => {
    const item = h('div', {
      className: 'list-item' + (i === state.selectedSection ? ' active' : ''),
      draggable: 'true',
    },
      h('span', {className:'grip'}, '\u2261'),
      h('span', {className:'name', onClick:() => { state.selectedSection = i; render(); }}, sec.name || '(unnamed)'),
      h('button', {className:'del-btn', title:'Delete', onClick:(e) => {
        e.stopPropagation();
        if (sections.length <= 2) { alert('At least 2 sections required'); return; }
        sections.splice(i, 1);
        if (state.selectedSection >= sections.length) state.selectedSection = sections.length - 1;
        markDirty(); render();
      }}, '\u00D7'),
    );

    item.addEventListener('dragstart', e => {
      e.dataTransfer.setData('text/plain', i);
      item.classList.add('dragging');
    });
    item.addEventListener('dragend', () => item.classList.remove('dragging'));
    item.addEventListener('dragover', e => { e.preventDefault(); item.classList.add('drag-over'); });
    item.addEventListener('dragleave', () => item.classList.remove('drag-over'));
    item.addEventListener('drop', e => {
      e.preventDefault();
      item.classList.remove('drag-over');
      const from = parseInt(e.dataTransfer.getData('text/plain'));
      if (from !== i) {
        const [moved] = sections.splice(from, 1);
        sections.splice(i, 0, moved);
        state.selectedSection = i;
        markDirty(); render();
      }
    });

    list.appendChild(item);
  });

  return h('div', {className:'section-list'},
    list,
    h('div', {className:'btns'},
      h('button', {onClick:() => {
        const tmpl = JSON.parse(JSON.stringify(SECTION_TEMPLATE));
        let name = 'NEW_SECTION';
        const names = getSectionNames();
        let c = 1;
        while (names.includes(name)) name = `NEW_SECTION_${c++}`;
        tmpl.name = name;
        sections.splice(state.selectedSection + 1, 0, tmpl);
        state.selectedSection++;
        markDirty(); render();
      }}, '+ Add'),
      h('button', {onClick:() => {
        if (sections.length === 0) return;
        const copy = JSON.parse(JSON.stringify(sections[state.selectedSection]));
        copy.name = copy.name + '_copy';
        sections.splice(state.selectedSection + 1, 0, copy);
        state.selectedSection++;
        markDirty(); render();
      }}, 'Duplicate'),
    ),
  );
}

let SECTION_TEMPLATE = null;
fetch('/api/section_template').then(r=>r.json()).then(t => { SECTION_TEMPLATE = t; });

function renderSectionDetail(sec, idx) {
  const snames = getSectionNames();
  const stageKeys = getStageKeys();
  function sf(f) { return v => { sec[f] = v; markDirty(); }; }

  // Basic
  const basic = h('div', {className:'card'},
    h('h3', {}, 'Basic'),
    field('Name', textInput(sec.name, v => { sec.name = v; markDirty(); render(); })),
    field('num_nodes', numInput(sec.num_nodes, v => { sec.num_nodes = v; markDirty(); }, {min:'1',step:'1'})),
    field('rocket_stage', selectInput(sec.rocket_stage, stageKeys.length?stageKeys:['1'], sf('rocket_stage'))),
    field('engine_mode', selectInput(sec.engine_mode, ENGINE_MODE_OPTIONS, sf('engine_mode'))),
  );

  // Propulsion
  const propulsion = h('div', {className:'card'},
    h('h3', {}, 'Propulsion'),
    field('thrust_vac', numInput(sec.thrust_vac, sf('thrust_vac')), h('span',{className:'unit'},'N')),
    field('Isp_vac', numInput(sec.Isp_vac, sf('Isp_vac'),{step:'0.01'}), h('span',{className:'unit'},'s')),
    h('div', {className:'issue warning'}, 'Throttle values are currently ignored by the CasADi solver.'),
    field('throttle min (unused)', numInput(sec.throttle?sec.throttle[0]:1, v => {
      if(!sec.throttle) sec.throttle=[1,1];
      sec.throttle[0]=v; markDirty();
    }, {min:'0',max:'1',step:'0.01'})),
    field('throttle max (unused)', numInput(sec.throttle?sec.throttle[1]:1, v => {
      if(!sec.throttle) sec.throttle=[1,1];
      sec.throttle[1]=v; markDirty();
    }, {min:'0',max:'1',step:'0.01'})),
    field('nozzle_area', numInput(sec.nozzle_area, sf('nozzle_area'),{step:'0.001'}), h('span',{className:'unit'},'m\u00B2')),
  );

  // Attitude
  const attitude = h('div', {className:'card'},
    h('h3', {}, 'Attitude'),
    field('attitude constraint', selectInput(sec['attitude constraint'], ATTITUDE_OPTIONS, v => { sec['attitude constraint']=v; markDirty(); })),
  );

  // Initial Guess
  const ig = sec['initial guess'] || {};
  const guess = h('div', {className:'card'},
    h('h3', {}, 'Initial Guess'),
    field('time', numInput(ig.time, v => { ig.time=v; markDirty(); },{}), h('span',{className:'unit'},'s')),
    field('pitch_rate', numInput(ig.pitch_rate, v => { ig.pitch_rate=v; markDirty(); },{step:'0.01'}), h('span',{className:'unit'},'deg/s')),
    field('yaw_rate', numInput(ig.yaw_rate, v => { ig.yaw_rate=v; markDirty(); },{step:'0.01'}), h('span',{className:'unit'},'deg/s')),
  );

  // Time Constraint
  const tc = sec['time constraint'] || {};
  const tcMode = tc.mode || 'free';
  const isFree = tcMode === 'free';
  const timeConstraint = h('div', {className:'card'},
    h('h3', {}, 'Time Constraint'),
    field('mode', selectInput(tcMode, TIME_MODE_OPTIONS, v => {
      sec['time constraint'] = sec['time constraint'] || {};
      sec['time constraint'].mode = v;
      if (v === 'free') {
        sec['time constraint'].value = null;
        sec['time constraint']['reference point'] = null;
      }
      markDirty(); render();
    })),
    field('value', numInput(tc.value, v => { tc.value=v; markDirty(); }, {disabled:isFree}), h('span',{className:'unit'},'s')),
    field('reference point', (() => {
      const opts = ['',...snames.filter(n => n !== sec.name)];
      const sel = selectInput(tc['reference point']||'', opts, v => {
        tc['reference point'] = v || null; markDirty();
      });
      if (isFree) sel.disabled = true;
      return sel;
    })()),
  );

  // AOA / Q / Q-alpha constraints
  function renderSimpleConstraint(label, key) {
    const con = sec[key] || {};
    const enabled = con.mode != null;
    const toggle = h('input', {type:'checkbox', ...(enabled?{checked:'true'}:{})});
    toggle.addEventListener('change', e => {
      if (e.target.checked) {
        sec[key] = {mode:'max', value:0, range:'all'};
      } else {
        sec[key] = {};
      }
      markDirty(); render();
    });
    const block = h('div', {className:'constraint-block'+(enabled?'':' disabled')},
      field('value', numInput(con.value, v => { con.value=v; markDirty(); })),
      field('range', selectInput(con.range||'all', ['all','initial'], v => { con.range=v; markDirty(); })),
    );
    return h('div', {},
      h('div', {className:'constraint-toggle'}, toggle, h('span',{},label)),
      enabled ? block : null,
    );
  }

  const constraintCards = h('div', {className:'card'},
    h('h3', {}, 'Flight Constraints'),
    renderSimpleConstraint('AOA constraint', 'AOA constraint'),
    renderSimpleConstraint('Dynamic pressure constraint', 'dynamic pressure constraint'),
    renderSimpleConstraint('Q-alpha constraint', 'Q-alpha constraint'),
  );

  // Waypoint constraint
  const wp = sec['waypoint constraint'] || {};
  const wpEntries = Object.entries(wp);
  const wpCard = h('div', {className:'card'},
    h('h3', {}, 'Waypoint Constraint'),
    Object.keys(wp).some(k => LEGACY_WP_KEYS.includes(k))
      ? h('div', {className:'issue warning'}, 'Legacy downrange waypoint constraints are preserved, but not supported by the current CasADi solver.')
      : null,
    ...wpEntries.map(([wkey, wval]) =>
      h('div', {className:'wp-row'},
        selectInput(wkey, Array.from(new Set(WP_KEYS.concat(LEGACY_WP_KEYS.filter(k => k === wkey)))), newKey => {
          const data = wp[wkey];
          delete wp[wkey];
          wp[newKey] = data;
          markDirty(); render();
        }),
        selectInput(wval.mode||'exact', WP_MODES, v => { wval.mode=v; markDirty(); }),
        numInput(wval.value, v => { wval.value=v; markDirty(); }, {step:'0.001'}),
        h('button', {onClick:() => { delete wp[wkey]; markDirty(); render(); }}, '\u00D7'),
      )
    ),
    h('button', {className:'add-btn', onClick:() => {
      const used = Object.keys(wp);
      const avail = WP_KEYS.filter(k=>!used.includes(k));
      const key = avail.length ? avail[0] : WP_KEYS[0];
      wp[key] = {mode:'exact', value:0};
      markDirty(); render();
    }}, '+ Add Waypoint'),
  );

  // Antenna constraint
  const ant = sec['antenna constraint'] || {};
  const antEntries = Object.entries(ant);
  const antCard = h('div', {className:'card'},
    h('h3', {}, 'Antenna Constraint'),
    ...antEntries.map(([aname, aval]) =>
      h('div', {className:'ant-row'},
        h('span', {style:{fontWeight:'600',fontSize:'12px'}}, 'Station:'),
        textInput(aname, newName => {
          const data = ant[aname];
          delete ant[aname];
          ant[newName] = data;
          markDirty();
        }, {style:'width:100px'}),
        h('span',{style:{fontSize:'12px'}},'lat:'), numInput(aval.lat, v=>{aval.lat=v;markDirty();},{step:'0.0001',style:'width:100px'}),
        h('span',{style:{fontSize:'12px'}},'lon:'), numInput(aval.lon, v=>{aval.lon=v;markDirty();},{step:'0.0001',style:'width:100px'}),
        h('span',{style:{fontSize:'12px'}},'alt:'), numInput(aval.altitude, v=>{aval.altitude=v;markDirty();},{style:'width:80px'}),
        h('span',{style:{fontSize:'12px'}},'elev_min:'), numInput(aval.elevation_min, v=>{aval.elevation_min=v;markDirty();},{step:'0.1',style:'width:70px'}),
        h('button', {onClick:() => { delete ant[aname]; markDirty(); render(); }}, '\u00D7'),
      )
    ),
    h('button', {className:'add-btn', onClick:() => {
      let name = 'STATION1';
      let c = 1;
      while (ant[name]) name = `STATION${c++}`;
      ant[name] = {lat:0, lon:0, altitude:0, elevation_min:0};
      markDirty(); render();
    }}, '+ Add Station'),
  );

  return h('div', {className:'section-detail'},
    basic, propulsion, attitude, guess, timeConstraint,
    constraintCards, wpCard, antCard,
  );
}

// ─── File Operations ───────────────────────────────────────────────────
async function newFile() {
  if (state.dirty && !confirm('Discard unsaved changes?')) return;
  const resp = await fetch('/api/template');
  state.data = await resp.json();
  state.filePath = null;
  state.dirty = false;
  state.selectedSection = 0;
  state.activeTab = 'general';
  render();
  setStatus([{level:'ok', message:'New file created from template'}]);
}

async function openFile() {
  if (state.dirty && !confirm('Discard unsaved changes?')) return;
  showBrowseModal('Open Settings File', async (path) => {
    try {
      const resp = await fetch('/api/load', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({path})
      });
      const result = await resp.json();
      if (!result.ok) { alert('Error: ' + result.error); return; }
      state.data = result.data;
      state.filePath = result.path;
      state.dirty = false;
      state.selectedSection = 0;
      render();
      setStatus([{level:'ok', message:'Loaded: ' + result.path}]);
    } catch(e) { alert('Failed to load: ' + e.message); }
  });
}

async function saveFile() {
  if (!state.data) return;
  if (!state.filePath) { saveFileAs(); return; }
  await doSave(state.filePath);
}

async function saveFileAs() {
  if (!state.data) return;
  showSaveModal(async (path) => { await doSave(path); });
}

async function doSave(path) {
  try {
    const resp = await fetch('/api/save', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({path, data: state.data})
    });
    const result = await resp.json();
    if (!result.ok) { alert('Error: ' + result.error); return; }
    state.filePath = result.path;
    state.dirty = false;
    render();
    setStatus([{level:'ok', message:'Saved: ' + result.path}]);
  } catch(e) { alert('Failed to save: ' + e.message); }
}

async function validateFile() {
  if (!state.data) return;
  try {
    const resp = await fetch('/api/validate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({data: state.data})
    });
    const result = await resp.json();
    if (result.issues.length === 0) {
      setStatus([{level:'ok', message:'Validation passed - no issues found'}]);
    } else {
      setStatus(result.issues);
    }
  } catch(e) { alert('Validation failed: ' + e.message); }
}

function setStatus(issues) {
  const bar = document.getElementById('statusbar');
  bar.innerHTML = '';
  issues.forEach(iss => {
    const cls = iss.level === 'ok' ? 'ok' : iss.level;
    const prefix = iss.level === 'error' ? '\u2716 ' : iss.level === 'warning' ? '\u26A0 ' : '\u2714 ';
    const pathStr = iss.path ? `[${iss.path}] ` : '';
    const div = h('div', {className:'issue '+cls, onClick:() => {
      if (iss.path && iss.path.startsWith('sections[')) {
        const m = iss.path.match(/sections\[(\d+)\]/);
        if (m) {
          state.activeTab = 'sections';
          state.selectedSection = parseInt(m[1]);
          render();
        }
      }
    }}, prefix + pathStr + iss.message);
    bar.appendChild(div);
  });
}

// ─── Modals ────────────────────────────────────────────────────────────
function showModal(content) {
  const root = document.getElementById('modal-root');
  root.innerHTML = '';
  const overlay = h('div', {className:'modal-overlay', onClick:e => {
    if (e.target === overlay) root.innerHTML = '';
  }}, h('div', {className:'modal'}, content));
  root.appendChild(overlay);
}

function hideModal() { document.getElementById('modal-root').innerHTML = ''; }

async function showBrowseModal(title, onSelect) {
  let currentDir = state.filePath ? state.filePath.substring(0, state.filePath.lastIndexOf('/')) : null;

  async function loadDir(dir) {
    const url = dir ? `/api/browse?dir=${encodeURIComponent(dir)}` : '/api/browse';
    const resp = await fetch(url);
    const result = await resp.json();
    if (!result.ok) { alert(result.error); return; }
    currentDir = result.dir;

    const content = h('div', {},
      h('h2', {}, title),
      h('div', {className:'browse-path'}, result.dir),
      h('div', {className:'browse-list'},
        h('div', {className:'browse-item', onClick:() => loadDir(result.dir + '/..')},
          h('span', {className:'icon'}, '\uD83D\uDCC1'), h('span', {}, '..')
        ),
        ...result.entries.map(e =>
          h('div', {className:'browse-item', onClick:() => {
            const full = result.dir + '/' + e.name;
            if (e.type === 'dir') loadDir(full);
            else { hideModal(); onSelect(full); }
          }},
            h('span', {className:'icon'}, e.type === 'dir' ? '\uD83D\uDCC1' : '\uD83D\uDCC4'),
            h('span', {}, e.name)
          )
        )
      ),
      h('div', {className:'modal-btns'},
        h('button', {className:'secondary', onClick:hideModal}, 'Cancel')
      )
    );
    showModal(content);
  }

  await loadDir(currentDir);
}

function showSaveModal(onSave) {
  const pathInput = h('input', {type:'text', value: state.filePath || '', placeholder:'Enter file path...'});
  const content = h('div', {},
    h('h2', {}, 'Save As'),
    pathInput,
    h('div', {className:'modal-btns'},
      h('button', {className:'secondary', onClick:hideModal}, 'Cancel'),
      h('button', {className:'primary', onClick:() => {
        const path = pathInput.value.trim();
        if (!path) { alert('Please enter a path'); return; }
        hideModal();
        onSave(path);
      }}, 'Save'),
    )
  );
  showModal(content);
}

// ─── Init ──────────────────────────────────────────────────────────────
async function init() {
  const resp = await fetch('/api/template');
  state.data = await resp.json();
  render();
}

init();
</script>
</body>
</html>"""


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
