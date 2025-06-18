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

import sys
import json
import pickle

import numpy as np
import pandas as pd
from pyoptsparse import IPOPT, SNOPT, Optimization

from lib.coordinate_c import *
from initialize import initialize_xdict_6DoF_from_file, initialize_xdict_6DoF_2
from output_result import output_result
import lib.con_init_terminal_knot as con_a
import lib.con_trajectory as con_traj
import lib.con_aero as con_aero
import lib.con_dynamics as con_dynamics
import lib.con_waypoint as con_wp
from user_constraints import equality_user, inequality_user
import lib.con_user as con_user
from lib.cost_gradient import cost_6DoF, cost_jac
from lib.SectionParameters import PSparams

version = "0.8.1"

mission_name = sys.argv[1]

fin = open(mission_name, "r")
settings = json.load(fin)
fin.close()

wind = pd.read_csv(settings["Wind file"])
wind["wind_n"] = wind["wind_speed[m/s]"] * -np.cos(np.radians(wind["direction[deg]"]))
wind["wind_e"] = wind["wind_speed[m/s]"] * -np.sin(np.radians(wind["direction[deg]"]))

wind_table = wind[["altitude[m]", "wind_n", "wind_e"]].to_numpy()

ca = pd.read_csv(settings["CA file"])
ca_table = ca.to_numpy()

stages = settings["RocketStage"]
launch_conditions = settings["LaunchCondition"]
terminal_conditions = settings["TerminalCondition"]

t_init = 0.0
launchsite_ecef = np.array(
    geodetic2ecef(
        launch_conditions["lat"],
        launch_conditions["lon"],
        launch_conditions["altitude"],
    )
)
launchsite_eci = ecef2eci(launchsite_ecef, t_init)

events = pd.read_csv(settings["Event setting file"], index_col=0)

num_sections = len(events) - 1

events["timeduration"] = -events["time"].diff(-1)
events["timeduration"].iat[-1] = 9000.0
events["timeFinishAt"] = events["time"] + events["timeduration"]
events["mass_jettison"] = 0.0

for key, stage in stages.items():
    if stage["separation_at"] in events.index:
        events.at[stage["separation_at"], "mass_jettison"] = stage["mass_dry"]
    elif stage["separation_at"] is not None:
        print("WARNING: separation time is invalid : stage {}".format(key))

    if stage["dropMass"] is not None:
        for key, item in stage["dropMass"].items():
            if item["separation_at"] in events.index:
                events.at[item["separation_at"], "mass_jettison"] = item["mass"]
            else:
                print("WARNING: separation time is invalid : {}".format(key))

events["massflow"] = 0.0
events["reference_area"] = 0.0
events["hold_pitch"] = False
events["hold_yaw"] = False

for i in events.index:

    stage = stages[str(events.at[i, "rocketStage"])]

    events.at[i, "reference_area"] = stage["reference_area"]

    if events.at[i, "engineOn"]:
        events.at[i, "massflow"] = events.at[i, "thrust"] / stage["Isp_vac"] / 9.80665

    att = events.at[i, "attitude"]

pdict = settings
pdict["params"] = events.to_dict("records")
nodes = events["num_nodes"][:-1]
for i, event_name in enumerate(events.index):
    pdict["params"][i]["name"] = event_name
pdict["event_index"] = {val["name"]: i for i, val in enumerate(pdict["params"])}
assert len(nodes) == len(pdict["params"]) - 1

N = sum(nodes)

index = [0]
k = 0
for n in nodes:
    k += n
    index.append(k)

pdict["ps_params"] = PSparams(nodes)

pdict["wind_table"] = wind_table
pdict["ca_table"] = ca_table
pdict["N"] = N
pdict["M"] = N + num_sections
pdict["num_sections"] = num_sections

r_init = launchsite_eci
v_init = vel_ecef2eci(np.zeros(3), launchsite_ecef, t_init)
quat_init = quatmult(
    quat_eci2nedg(r_init, t_init),
    quat_from_euler(launch_conditions["flight_azimuth_init"], 90.0, 0.0),
)
m_init = sum([s["mass_dry"] + s["mass_propellant"] for s in stages.values()])
if settings["OptimizationMode"] != "Payload":
    m_init += settings["mass_payload"]
x_init = np.hstack((m_init, r_init, v_init, quat_init))

u_init = np.zeros(3)

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

pdict["dx"] = 1.0e-8

condition = {**settings["TerminalCondition"], **settings["FlightConstraint"]}
condition["init"] = {}
condition["init"]["mass"] = m_init
condition["init"]["position"] = r_init
condition["init"]["velocity"] = v_init
condition["init"]["quaternion"] = quat_init
condition["init"]["u"] = u_init
condition["flight_azimuth_init"] = launch_conditions["flight_azimuth_init"]
condition["OptimizationMode"] = settings["OptimizationMode"]

if (
    "Initial trajectory file" in settings.keys()
    and settings["Initial trajectory file"] is not None
):
    x_ref = pd.read_csv(settings["Initial trajectory file"])
    xdict_init = initialize_xdict_6DoF_from_file(
        x_ref, pdict, condition, unitdict, "LGR", False
    )

else:
    xdict_init = initialize_xdict_6DoF_2(
        x_init, pdict, condition, unitdict, "LGR", 0.1, False
    )


def objfunc(xdict):
    funcs = {}
    funcs["obj"] = cost_6DoF(xdict, condition)
    funcs["eqcon_init"] = con_a.equality_init(xdict, pdict, unitdict, condition)
    funcs["eqcon_time"] = con_a.equality_time(xdict, pdict, unitdict, condition)
    funcs["eqcon_dyn_mass"] = con_dynamics.equality_dynamics_mass(
        xdict, pdict, unitdict, condition
    )
    funcs["eqcon_dyn_pos"] = con_dynamics.equality_dynamics_position(
        xdict, pdict, unitdict, condition
    )
    funcs["eqcon_dyn_vel"] = con_dynamics.equality_dynamics_velocity(
        xdict, pdict, unitdict, condition
    )
    funcs["eqcon_dyn_quat"] = con_dynamics.equality_dynamics_quaternion(
        xdict, pdict, unitdict, condition
    )

    funcs["eqcon_knot"] = con_a.equality_knot_LGR(xdict, pdict, unitdict, condition)
    funcs["eqcon_terminal"] = con_a.equality_6DoF_LGR_terminal(
        xdict, pdict, unitdict, condition
    )
    funcs["eqcon_rate"] = con_traj.equality_6DoF_rate(xdict, pdict, unitdict, condition)
    funcs["eqcon_pos"] = con_wp.equality_posLLH(xdict, pdict, unitdict, condition)
    funcs["eqcon_iip"] = con_wp.equality_IIP(xdict, pdict, unitdict, condition)
    funcs["eqcon_user"] = con_user.equality_user(xdict, pdict, unitdict, condition)

    funcs["ineqcon_alpha"] = con_aero.inequality_max_alpha(
        xdict, pdict, unitdict, condition
    )
    funcs["ineqcon_q"] = con_aero.inequality_max_q(xdict, pdict, unitdict, condition)
    funcs["ineqcon_qalpha"] = con_aero.inequality_max_qalpha(
        xdict, pdict, unitdict, condition
    )
    funcs["ineqcon_mass"] = con_traj.inequality_mass(xdict, pdict, unitdict, condition)
    funcs["ineqcon_kick"] = con_traj.inequality_kickturn(
        xdict, pdict, unitdict, condition
    )
    funcs["ineqcon_time"] = con_a.inequality_time(xdict, pdict, unitdict, condition)
    funcs["ineqcon_pos"] = con_wp.inequality_posLLH(xdict, pdict, unitdict, condition)
    funcs["ineqcon_iip"] = con_wp.inequality_IIP(xdict, pdict, unitdict, condition)
    funcs["ineqcon_antenna"] = con_wp.inequality_antenna(
        xdict, pdict, unitdict, condition
    )
    funcs["ineqcon_user"] = con_user.inequality_user(xdict, pdict, unitdict, condition)

    fail = False

    return funcs, fail


def sens(xdict, funcs):
    funcsSens = {}
    funcsSens["obj"] = cost_jac(xdict, condition)
    funcsSens["eqcon_init"] = con_a.equality_jac_init(xdict, pdict, unitdict, condition)
    funcsSens["eqcon_time"] = con_a.equality_jac_time(xdict, pdict, unitdict, condition)
    funcsSens["eqcon_dyn_mass"] = con_dynamics.equality_jac_dynamics_mass(
        xdict, pdict, unitdict, condition
    )
    funcsSens["eqcon_dyn_pos"] = con_dynamics.equality_jac_dynamics_position(
        xdict, pdict, unitdict, condition
    )
    funcsSens["eqcon_dyn_vel"] = con_dynamics.equality_jac_dynamics_velocity(
        xdict, pdict, unitdict, condition
    )
    funcsSens["eqcon_dyn_quat"] = con_dynamics.equality_jac_dynamics_quaternion(
        xdict, pdict, unitdict, condition
    )

    funcsSens["eqcon_knot"] = con_a.equality_jac_knot_LGR(
        xdict, pdict, unitdict, condition
    )
    funcsSens["eqcon_terminal"] = con_a.equality_jac_6DoF_LGR_terminal(
        xdict, pdict, unitdict, condition
    )
    funcsSens["eqcon_rate"] = con_traj.equality_jac_6DoF_rate(
        xdict, pdict, unitdict, condition
    )
    funcsSens["eqcon_pos"] = con_wp.equality_jac_posLLH(
        xdict, pdict, unitdict, condition
    )
    funcsSens["eqcon_iip"] = con_wp.equality_jac_IIP(xdict, pdict, unitdict, condition)
    funcsSens["eqcon_user"] = con_user.equality_jac_user(
        xdict, pdict, unitdict, condition
    )

    funcsSens["ineqcon_alpha"] = con_aero.inequality_jac_max_alpha(
        xdict, pdict, unitdict, condition
    )
    funcsSens["ineqcon_q"] = con_aero.inequality_jac_max_q(
        xdict, pdict, unitdict, condition
    )
    funcsSens["ineqcon_qalpha"] = con_aero.inequality_jac_max_qalpha(
        xdict, pdict, unitdict, condition
    )
    funcsSens["ineqcon_mass"] = con_traj.inequality_jac_mass(
        xdict, pdict, unitdict, condition
    )
    funcsSens["ineqcon_kick"] = con_traj.inequality_jac_kickturn(
        xdict, pdict, unitdict, condition
    )
    funcsSens["ineqcon_time"] = con_a.inequality_jac_time(
        xdict, pdict, unitdict, condition
    )
    funcsSens["ineqcon_pos"] = con_wp.inequality_jac_posLLH(
        xdict, pdict, unitdict, condition
    )
    funcsSens["ineqcon_iip"] = con_wp.inequality_jac_IIP(
        xdict, pdict, unitdict, condition
    )
    funcsSens["ineqcon_antenna"] = con_wp.inequality_jac_antenna(
        xdict, pdict, unitdict, condition
    )
    funcsSens["ineqcon_user"] = con_user.inequality_jac_user(
        xdict, pdict, unitdict, condition
    )

    fail = False
    return funcsSens, fail


optProb = Optimization("Rocket trajectory optimization", objfunc)


optProb.addVarGroup(
    "mass",
    len(xdict_init["mass"]),
    value=xdict_init["mass"],
    lower=1.0e-9,
    upper=2.0,
)
optProb.addVarGroup(
    "position",
    len(xdict_init["position"]),
    value=xdict_init["position"],
    lower=-10.0,
    upper=10.0,
)
optProb.addVarGroup(
    "velocity",
    len(xdict_init["velocity"]),
    value=xdict_init["velocity"],
    lower=-20.0,
    upper=20.0,
)
optProb.addVarGroup(
    "quaternion",
    len(xdict_init["quaternion"]),
    value=xdict_init["quaternion"],
    lower=-1.0,
    upper=1.0,
)

optProb.addVarGroup(
    "u", len(xdict_init["u"]), value=xdict_init["u"], lower=-9.0, upper=9.0
)
optProb.addVarGroup(
    "t", len(xdict_init["t"]), value=xdict_init["t"], lower=0.0, upper=1.5
)

f_init = objfunc(xdict_init)[0]
jac_init = sens(xdict_init, f_init)[0]


wrt = {
    "eqcon_init": ["mass", "position", "velocity", "quaternion"],
    "eqcon_time": ["t"],
    "eqcon_dyn_mass": ["mass", "t"],
    "eqcon_dyn_pos": ["position", "velocity", "t"],
    "eqcon_dyn_vel": ["mass", "position", "velocity", "quaternion", "t"],
    "eqcon_dyn_quat": ["quaternion", "u", "t"],
    "eqcon_knot": ["mass", "position", "velocity", "quaternion"],
    "eqcon_terminal": ["position", "velocity"],
    "eqcon_rate": ["position", "quaternion", "u"],
    "eqcon_pos": ["position", "t"],
    "eqcon_iip": ["position", "velocity", "t"],
    "eqcon_user": ["mass", "position", "velocity", "quaternion", "u", "t"],
    "ineqcon_alpha": ["position", "velocity", "quaternion", "t"],
    "ineqcon_q": ["position", "velocity", "quaternion", "t"],
    "ineqcon_qalpha": ["position", "velocity", "quaternion", "t"],
    "ineqcon_mass": ["mass"],
    "ineqcon_kick": ["u"],
    "ineqcon_time": ["t"],
    "ineqcon_pos": ["position", "t"],
    "ineqcon_iip": ["position", "velocity", "t"],
    "ineqcon_antenna": ["position", "t"],
    "ineqcon_user": ["mass", "position", "velocity", "quaternion", "u", "t"],
}

if condition["OptimizationMode"] == "Payload":
    wrt["eqcon_init"] = ["position", "velocity", "quaternion"]

for key, val in f_init.items():
    if key == "obj":
        optProb.addObj("obj")
    else:
        if val is None:
            pass
        else:
            lower_bound = 0.0
            if "ineqcon" in key:
                upper_bound = None
            else:
                upper_bound = 0.0

            if hasattr(val, "__len__"):
                optProb.addConGroup(
                    key,
                    len(val),
                    lower=lower_bound,
                    upper=upper_bound,
                    wrt=wrt[key],
                    jac=jac_init[key],
                )
            else:
                optProb.addConGroup(
                    key,
                    1,
                    lower=lower_bound,
                    upper=upper_bound,
                    wrt=wrt[key],
                    jac=jac_init[key],
                )


if "SNOPT" in settings.keys():
    options_SNOPT = settings["SNOPT"]
    options_SNOPT["Print file"] = "output/{}-SNOPT-print.out".format(settings["name"])
    options_SNOPT["Summary file"] = "output/{}-SNOPT-summary.out".format(
        settings["name"]
    )
    if "Return work arrays" not in options_SNOPT.keys():
        options_SNOPT["Return work arrays"] = True

    rdict = None
    if (
        "SNOPT work array file" in settings.keys()
        and settings["SNOPT work array file"] is not None
    ):
        with open(settings["SNOPT work array file"], "rb") as f:
            rdict = pickle.load(f)
        # raw data size check
        if len(rdict["xs"]) != sum([len(v) for v in optProb.variables.values()]) + sum(
            [v.ncon for v in optProb.constraints.values()]
        ):
            print(
                "WARNING : The dimension of raw data does not match. Switched to cold start mode."
            )
            rdict = None
            options_SNOPT["Start"] = "Cold"

    opt = SNOPT(options=options_SNOPT)

    if options_SNOPT["Return work arrays"]:
        sol, raw = opt(optProb, sens=sens, restartDict=rdict)
        with open("output/{}-SNOPT-raw.bin".format(settings["name"]), "wb") as f:
            pickle.dump(raw, f)
    else:
        sol = opt(optProb, sens=sens, restartDict=rdict)

elif "IPOPT" in settings.keys():
    options_IPOPT = settings["IPOPT"]
    options_IPOPT["output_file"] = "output/{}-IPOPT.out".format(settings["name"])
    opt = IPOPT(options=options_IPOPT)
    sol = opt(optProb, sens=sens)

else:
    print("ERROR : UNRECOGNIZED OPTIMIZER. USE IPOPT OR SNOPT.")
    sys.exit()


# Post processing

for i, p in enumerate(pdict["params"]):
    p["time"] = sol.xStar["t"][i] * unitdict["t"]

flag_savefig = False
# ========================
# Post Process
# ------------------------
# Convert parameter vector to variable

tu_res = np.array([])
tx_res = np.array([])

for i in range(num_sections):
    to = sol.xStar["t"][i]
    tf = sol.xStar["t"][i + 1]
    tau_x = np.hstack((-1.0, pdict["ps_params"].tau(i)))
    tu_res = np.hstack(
        (
            tu_res,
            (pdict["ps_params"].tau(i) * (tf - to) / 2 + (tf + to) / 2) * unitdict["t"],
        )
    )
    tx_res = np.hstack(
        (tx_res, (tau_x * (tf - to) / 2 + (tf + to) / 2) * unitdict["t"])
    )

# output

m_res = sol.xStar["mass"] * unitdict["mass"]

res_info = []
res_info.append("GELATO: GENERIC LAUNCH TRAJECTORY OPTIMIZER v{}\n\n".format(version))
res_info.append("Input file name : {}\n\n".format(mission_name))
res_info.append("initial mass    : {:10.3f} kg\n".format(m_res[0]))
res_info.append("final mass      : {:10.3f} kg\n".format(m_res[-1]))

mass_drop = 0.0
for i, stage in settings["RocketStage"].items():
    if stage["dropMass"] is not None:
        mass_drop += sum([item["mass"] for item in stage["dropMass"].values()])
res_info.append(
    "payload         : {:10.3f} kg\n\n".format(m_res[0] - m_init - mass_drop)
)

res_info.append("optTime         : {:11.6f}\n".format(sol.optTime))
res_info.append("userObjTime     : {:11.6f}\n".format(sol.userObjTime))
res_info.append("userSensTime    : {:11.6f}\n".format(sol.userSensTime))
res_info.append("interfaceTime   : {:11.6f}\n".format(sol.interfaceTime))
res_info.append("optCodeTime     : {:11.6f}\n".format(sol.optCodeTime))
res_info.append("userObjCalls    : {:4d}\n".format(sol.userObjCalls))
res_info.append("userSensCalls   : {:4d}\n\n".format(sol.userSensCalls))

res_info.append(
    "{} (code {})\n".format(sol.optInform["text"], str(sol.optInform["value"]))
)


print("".join(res_info[1:]))
with open("output/{}-optResult.txt".format(settings["name"]), mode="w") as fout:
    fout.write("".join(res_info))

out = output_result(sol.xStar, unitdict, tx_res, tu_res, pdict)

out.to_csv("output/{}-trajectoryResult.csv".format(settings["name"]), index=False)
