import sys 

import numpy as np
import pandas as pd
import json

from utils import *
from coordinate import *
from optimization6DoFquat import *
from user_constraints import *
from PSfunctions import *
from USStandardAtmosphere import *
from pyoptsparse import IPOPT, SNOPT, Optimization

version = "0.4.0"

mission_name = sys.argv[1]

fin = open(mission_name, 'r')
settings = json.load(fin)
fin.close()

wind = pd.read_csv(settings["Wind file"])
wind["wind_n"] = wind["wind_speed[m/s]"] * -np.cos(np.radians(wind["direction[deg]"]))
wind["wind_e"] = wind["wind_speed[m/s]"] * -np.sin(np.radians(wind["direction[deg]"]))

wind_table = wind[["altitude[m]","wind_n","wind_e"]].to_numpy()

ca = pd.read_csv(settings["CA file"])
ca_table = ca.to_numpy()

stages = settings["RocketStage"]
dropmass = settings["dropMass"]
launch_conditions = settings["LaunchCondition"]
terminal_conditions = settings["TerminalCondition"]

t_init = 0.0
launchsite_ecef = np.array(geodetic2ecef(launch_conditions["latitude_deg"], launch_conditions["longitude_deg"], launch_conditions["height_m"]))
launchsite_eci = ecef2eci(launchsite_ecef, t_init)

events = pd.read_csv(settings["Event setting file"], index_col=0)

num_sections = len(events) - 1

events["timeduration_sec"] = -events["timeAt_sec"].diff(-1)
events["timeduration_sec"].iat[-1] = 9000.0
events["timeFinishAt_sec"] = events["timeAt_sec"] + events["timeduration_sec"]
events["mass_jettison_kg"] = 0.0

for key, stage in stages.items():
    if stage["separation_at"] in events.index:
        events.at[stage["separation_at"], "mass_jettison_kg"] = stage["dryMass_kg"]
    elif stage["separation_at"] is not None:
        print("WARNING: separation time is invalid : stage {}".format(key))

for key, item in settings["dropMass"].items():
    if item["separation_at"] in events.index:
        events.at[item["separation_at"], "mass_jettison_kg"] = item["mass_kg"]
    else:
        print("WARNING: separation time is invalid : {}".format(key))

events["massflow_kgps"] = 0.0
events["airArea_m2"] = 0.0
events["hold_pitch"] = False
events["hold_yaw"] = False

for i in events.index:
    
    stage = stages[str(events.at[i, "rocketStage"])]
    
    events.at[i, "airArea_m2"] = stage["airArea_m2"]
    

    if events.at[i, "engineOn"]:
        events.at[i, "massflow_kgps"] = events.at[i, "thrust_n"] / stage["Isp_vac"] / 9.80665

    att = events.at[i, "attitude"]
    
    if att == "zero-lift-turn":
        events.at[i, "do_zeroliftturn"] = True
    else:
        events.at[i, "do_zeroliftturn"] = False
        
        if att == "kick-turn":
            events.at[i, "hold_yaw"] = True
        elif att == "hold" or att == "vertical":
            events.at[i, "hold_yaw"] = True
            events.at[i, "hold_pitch"] = True
        elif att == "pitch":
            events.at[i, "hold_yaw"] = True
            

pdict = {"params": events.to_dict('records')}
nodes = events["num_nodes"][:-1]
for i,event_name in enumerate(events.index):
    pdict["params"][i]["name"] = event_name
pdict["event_index"] = {}
assert(len(nodes) == len(pdict["params"])-1)

N = sum(nodes)

D = [differentiation_matrix_LGR(n) for n in nodes]
tau = [nodes_LGR(n) for n in nodes]
index = [0]
k = 0
for n in nodes:
    k += n
    index.append(k)
    
pdict["ps_params"]= [{"index_start": index[i],"nodes": nodes[i], "D" : D[i], "tau": tau[i]} for i in range(num_sections)]
pdict["wind_table"] = wind_table
pdict["ca_table"] = ca_table
pdict["N"] = N
pdict["total_points"] = N + num_sections
pdict["num_sections"] = num_sections

r_init = launchsite_eci
v_init = vel_ecef2eci(np.zeros(3),launchsite_ecef, t_init)
quat_init = quatmult(quat_eci2nedg(r_init, t_init), quat_from_euler(launch_conditions["init_azimuth_deg"], 90.0, 0.0))
m_init = sum([s["dryMass_kg"]+s["propellantMass_kg"] for s in stages.values()])
if settings["OptimizationMode"] != "Payload":
    m_init += settings["PayloadMass"]
x_init = np.hstack((m_init, r_init, v_init, quat_init))

u_init = np.zeros(3)

unit_R = 6378137
unit_V = 1000.0
unit_m = m_init
unit_u = 1.0
unit_t = pdict["params"][-1]["timeAt_sec"]

unitdict = {"mass"     : unit_m,
            "position" : unit_R,
            "velocity" : unit_V,
            "u"        : unit_u,
            "t"        : unit_t
}


condition = {**settings["TerminalCondition"], **settings["FlightConstraint"]}
condition["init"] = {}
condition["init"]["mass"] = m_init
condition["init"]["position"] = r_init
condition["init"]["velocity"] = v_init
condition["init"]["quaternion"] = quat_init
condition["init"]["u"] = u_init
condition["init_azimuth_deg"] = launch_conditions["init_azimuth_deg"]
condition["OptimizationMode"] = settings["OptimizationMode"]

if "Initial trajectory file" in settings.keys() and settings["Initial trajectory file"] is not None:
    x_ref = pd.read_csv(settings["Initial trajectory file"])
    xdict_init = initialize_xdict_6DoF_from_file(x_ref, pdict, condition, unitdict, 'LGR', False)

else:
    xdict_init = initialize_xdict_6DoF_2(x_init, pdict, condition, unitdict, 'LGR', 0.1, False)


def objfunc(xdict):
    funcs = {}
    funcs["obj"] = cost_6DoF(xdict, condition)
    funcs["eqcon_init"] = equality_init(xdict, pdict, unitdict, condition)
    funcs["eqcon_time"] = equality_time(xdict, pdict, unitdict, condition)
    funcs["eqcon_dyn_mass"] = equality_dynamics_mass(xdict, pdict, unitdict, condition)
    funcs["eqcon_dyn_pos"]  = equality_dynamics_position(xdict, pdict, unitdict, condition)
    funcs["eqcon_dyn_vel"]  = equality_dynamics_velocity(xdict, pdict, unitdict, condition)
    funcs["eqcon_dyn_quat"] = equality_dynamics_quaternion(xdict, pdict, unitdict, condition)

    funcs["eqcon_knot"] = equality_knot_LGR(xdict, pdict, unitdict, condition)
    funcs["eqcon_terminal"] = equality_6DoF_LGR_terminal(xdict, pdict, unitdict, condition)
    funcs["eqcon_rate"] = equality_6DoF_rate(xdict, pdict, unitdict, condition)
    funcs["eqcon_user"] = equality_user(xdict, pdict, unitdict, condition)

    funcs["ineqcon"] = inequality_6DoF(xdict, pdict, unitdict, condition)
    funcs["ineqcon_time"] = inequality_time(xdict, pdict, unitdict, condition)
    funcs["ineqcon_user"] = inequality_user(xdict, pdict, unitdict, condition)

    fail = False
    
    return funcs, fail

def sens(xdict, funcs):
    funcsSens = {}
    for key_f in funcs.keys():
        if key_f == "obj":
            funcsSens[key_f] = cost_jac(xdict, condition)
        else:
            funcsSens[key_f] = jac_fd((lambda xd,a,b,c:objfunc(xd)[0][key_f]), xdict, pdict, unitdict, condition)

    fail = False
    return funcsSens, fail

optProb = Optimization("Rocket trajectory optimization", objfunc)

optProb.addVarGroup("mass", len(xdict_init["mass"]), value=xdict_init["mass"], lower=0.0, upper=2.0)
optProb.addVarGroup("position", len(xdict_init["position"]), value=xdict_init["position"], lower=-10.0, upper=10.0)
optProb.addVarGroup("velocity", len(xdict_init["velocity"]), value=xdict_init["velocity"], lower=-20.0, upper=20.0)
optProb.addVarGroup("quaternion", len(xdict_init["quaternion"]), value=xdict_init["quaternion"], lower=-1.0, upper=1.0)

optProb.addVarGroup("u", len(xdict_init["u"]), value=xdict_init["u"], lower=-9.0, upper=9.0)
optProb.addVarGroup("t", len(xdict_init["t"]), value=xdict_init["t"], lower=0.0,  upper=1.5)


e_init     = equality_init(xdict_init, pdict, unitdict, condition)
e_time     = equality_time(xdict_init, pdict, unitdict, condition)
e_dyn_mass = equality_dynamics_mass(xdict_init, pdict, unitdict, condition)
e_dyn_pos  = equality_dynamics_position(xdict_init, pdict, unitdict, condition)
e_dyn_vel  = equality_dynamics_velocity(xdict_init, pdict, unitdict, condition)
e_dyn_quat = equality_dynamics_quaternion(xdict_init, pdict, unitdict, condition)

e_knot = equality_knot_LGR(xdict_init, pdict, unitdict, condition)
e_terminal = equality_6DoF_LGR_terminal(xdict_init, pdict, unitdict, condition)
e_rate = equality_6DoF_rate(xdict_init, pdict, unitdict, condition)
e_user = equality_user(xdict_init, pdict, unitdict, condition)

ie = inequality_6DoF(xdict_init, pdict, unitdict, condition)
ie_time = inequality_time(xdict_init, pdict, unitdict, condition)
ie_user = inequality_user(xdict_init, pdict, unitdict, condition)

optProb.addConGroup("eqcon_init",     len(e_init),     lower=0.0, upper=0.0, wrt=["mass","position","velocity","quaternion"], linear=True, jac=jac_fd(equality_init, xdict_init, pdict, unitdict, condition))
optProb.addConGroup("eqcon_time",     len(e_time),     lower=0.0, upper=0.0, wrt=["t"], linear=True, jac=jac_fd(equality_time, xdict_init, pdict, unitdict, condition))
optProb.addConGroup("eqcon_dyn_mass", len(e_dyn_mass), lower=0.0, upper=0.0, wrt=["mass","t"], jac=jac_fd(equality_dynamics_mass, xdict_init, pdict, unitdict, condition))
optProb.addConGroup("eqcon_dyn_pos",  len(e_dyn_pos),  lower=0.0, upper=0.0, wrt=["position","velocity","t"], jac=jac_fd(equality_dynamics_position, xdict_init, pdict, unitdict, condition))
optProb.addConGroup("eqcon_dyn_vel",  len(e_dyn_vel),  lower=0.0, upper=0.0, wrt=["mass","position","velocity","quaternion","t"], jac=jac_fd(equality_dynamics_velocity, xdict_init, pdict, unitdict, condition))
optProb.addConGroup("eqcon_dyn_quat", len(e_dyn_quat), lower=0.0, upper=0.0, wrt=["quaternion","u","t"], jac=jac_fd(equality_dynamics_quaternion, xdict_init, pdict, unitdict, condition))
optProb.addConGroup("eqcon_knot",     len(e_knot),     lower=0.0, upper=0.0, linear=True, jac=jac_fd(equality_knot_LGR, xdict_init, pdict, unitdict, condition))
optProb.addConGroup("eqcon_terminal", len(e_terminal), lower=0.0, upper=0.0, jac=jac_fd(equality_6DoF_LGR_terminal, xdict_init, pdict, unitdict, condition))
optProb.addConGroup("eqcon_rate",     len(e_rate),     lower=0.0, upper=0.0, wrt=["position","quaternion","u"], jac=jac_fd(equality_6DoF_rate, xdict_init, pdict, unitdict, condition))
optProb.addConGroup("ineqcon",        len(ie),         lower=0.0, upper=None, jac=jac_fd(inequality_6DoF, xdict_init, pdict, unitdict, condition))
optProb.addConGroup("ineqcon_time",   len(ie_time),    lower=0.0, upper=None, linear=True, jac=jac_fd(inequality_time, xdict_init, pdict, unitdict, condition))

if e_user is not None:
    if hasattr(e_user,"__len__"):
        optProb.addConGroup("eqcon_user", len(e_user), lower=0.0, upper=0.0, jac=jac_fd(equality_user, xdict_init, pdict, unitdict, condition))
    else:
        optProb.addConGroup("eqcon_user", 1, lower=0.0, upper=0.0, jac=jac_fd(equality_user, xdict_init, pdict, unitdict, condition))

if ie_user is not None:
    if hasattr(ie_user,"__len__"):
        optProb.addConGroup("ineqcon_user", len(ie_user), lower=0.0, upper=None, jac=jac_fd(inequality_user, xdict_init, pdict, unitdict, condition))
    else:
        optProb.addConGroup("ineqcon_user", 1, lower=0.0, upper=None, jac=jac_fd(inequality_user, xdict_init, pdict, unitdict, condition))


optProb.addObj("obj")


if "IPOPT" in settings.keys():
    options_IPOPT = settings["IPOPT"]
    options_IPOPT["output_file"] = "output/{}-IPOPT.out".format(settings["name"])
    opt = IPOPT(options=options_IPOPT)
elif "SNOPT" in settings.keys():
    options_SNOPT = settings["SNOPT"]
    options_SNOPT["Print file"] = "output/{}-SNOPT-print.out".format(settings["name"])
    options_SNOPT["Summary file"] = "output/{}-SNOPT-summary.out".format(settings["name"])
    opt = SNOPT(options=options_SNOPT)
else:
    print("ERROR : UNRECOGNIZED OPTIMIZER. USE IPOPT OR SNOPT.")
    sys.exit()

sol = opt(optProb, sens=sens)

# Post processing

for i,p in enumerate(pdict["params"]):
    if not p["timeFixed"]:
        p["timeAt_sec"] = sol.xStar["t"][i] * unitdict["t"]
        
flag_savefig = False
# ========================
# Post Process
# ------------------------
# Convert parameter vector to variable

tu_res = np.array([])
tx_res = np.array([])

for i in range(num_sections):
    to = sol.xStar["t"][i]
    tf = sol.xStar["t"][i+1]
    tau_x = np.hstack((-1.0, pdict["ps_params"][i]["tau"]))
    tu_res = np.hstack((tu_res, (pdict["ps_params"][i]["tau"]*(tf-to)/2 + (tf+to)/2) * unitdict["t"]))
    tx_res = np.hstack((tx_res, (tau_x*(tf-to)/2 + (tf+to)/2) * unitdict["t"]))

# output

m_res = sol.xStar["mass"] * unitdict["mass"]

res_version = "IST Trajectory Optimizer version : {}\n".format(version)
res_initial = "initial mass : {:.3f} kg\n".format(m_res[0])
res_final   = "final mass : {:.3f} kg\n".format(m_res[-1])
res_payload = "payload : {:.3f} kg\n".format(m_res[0] - m_init - sum([item["mass_kg"] for item in dropmass.values()]))

print("".join([res_initial, res_final, res_payload]))
with open("output/{}-optResult.txt".format(settings["name"]), mode="w") as fout:
    fout.write("".join([res_version, res_initial, res_final, res_payload]))

out = output_6DoF(sol.xStar, unitdict, tx_res, tu_res, pdict)

out.to_csv("output/{}-trajectoryResult.csv".format(settings["name"]))