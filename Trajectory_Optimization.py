import sys 

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime

from utils import *
from coordinate import *
from optimization6DoFquat import *
from PSfunctions import *
from USStandardAtmosphere import *
from pyoptsparse import IPOPT, Optimization

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
launch_conditions = settings["LaunchCondition"]
terminal_conditions = settings["TerminalCondition"]

t_init = 0.0
launchsite_ecef = np.array(pm.geodetic2ecef(launch_conditions["latitude_deg"], launch_conditions["longitude_deg"], launch_conditions["height_m"]))
launchsite_eci = ecef2eci(launchsite_ecef, t_init)

events = pd.read_csv(settings["Event setting file"])

num_sections = len(events) - 1

events["timeduration_sec"] = -events["timeAt_sec"].diff(-1)
events["timeduration_sec"].iat[-1] = 9000.0
events["timeFinishAt_sec"] = events["timeAt_sec"] + events["timeduration_sec"]
for i in range(num_sections):
    if events.at[i,"rocketStage"] < events.at[i+1,"rocketStage"]:
        events.at[i+1, "mass_jettison_kg"] = stages[str(events.at[i,"rocketStage"])]["dryMass_kg"]
        
events["massflow_kgps"] = 0.0
events["airArea_m2"] = 0.0
events["hold_pitch"] = False
events["hold_yaw"] = False
events["timeFixed"] = True

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
num_states = 11
num_controls = 3

assert(len(nodes) == len(pdict["params"])-1)

N = sum(nodes)

D = [differentiation_matrix_LG(n) for n in nodes]
weight = [weight_LG(n) for n in nodes]
tau = [nodes_LG(n) for n in nodes]
index = [0]
k = 0
for n in nodes:
    k += n
    index.append(k)
    
pdict["ps_params"]= [{"index_start": index[i],"nodes": nodes[i], "D" : D[i], "tau": tau[i], "weight": weight[i]} for i in range(num_sections)]
pdict["wind_table"] = wind_table
pdict["ca_table"] = ca_table
pdict["N"] = N
pdict["num_states"] = num_states
pdict["num_controls"] = num_controls
pdict["num_sections"] = num_sections

r_init = launchsite_eci
v_init = vel_ecef2eci(np.zeros(3),launchsite_ecef, t_init)
quat_init = quatmult(quat_eci2nedg(r_init, t_init), quat_from_euler(launch_conditions["init_azimuth_deg"], 90.0, 0.0))
m_init = sum([s["dryMass_kg"]+s["propellantMass_kg"] for s in stages.values()])
if not settings["OptimizationMode"]["Maximize initial mass"]:
    m_init += settings["PayloadMass"]
x_init = np.hstack((m_init, r_init, v_init, quat_init))

u_init = np.zeros(num_controls)

unit_R = 6378137
unit_V = 1000.0
unit_m = m_init

unit_x = np.array([unit_m] + [unit_R]*3 + [unit_V]*3 + [1.0]*4)
unit_u = np.ones(num_controls)
unit_t = pdict["params"][-1]["timeAt_sec"]

unit_xut = {"x": unit_x, "u": unit_u, "t": unit_t}

condition = {**settings["TerminalCondition"], **settings["FlightConstraint"]}
condition["x_init"] = x_init
condition["u_init"] = u_init
condition["init_azimuth_deg"] = launch_conditions["init_azimuth_deg"]
condition["OptimizationMode"] = settings["OptimizationMode"]

xdict_init = initialize_xdict_6DoF_2(x_init, pdict, condition, unit_xut, 'LG', 0.1, False)

xx_lower0 = np.array(([0.0] + [-unit_R*10]*3 + [-unit_V*20]*3 + [-1.0]*4)        * (N + num_sections))
xx_upper0 = np.array(([m_init*2.0] + [ unit_R*10]*3 + [ unit_V*20]*3 + [ 1.0]*4) * (N + num_sections)) 
xx_scale  = np.array(unit_x.tolist() * (N + num_sections))
uu_lower0 = np.array([-6.0] * num_controls * N)
uu_upper0 = np.array([ 6.0] * num_controls * N)
uu_scale  = np.array(unit_u.tolist() * N)

xx_lower = xx_lower0 / xx_scale
xx_upper = xx_upper0 / xx_scale
uu_lower = uu_lower0 / uu_scale
uu_upper = uu_upper0 / uu_scale

t_lower = 0.0
t_upper = 1.0

def objfunc(xdict):
    funcs = {}
    funcs["obj"] = cost_6DoF_LG(xdict, condition)
    funcs["eqcon_init"] = equality_init(xdict, pdict, unit_xut, condition)
    funcs["eqcon_time"] = equality_time(xdict, pdict, unit_xut, condition)
    funcs["eqcon_diff"] = equality_6DoF_LG_diff(xdict, pdict, unit_xut, condition)
    funcs["eqcon_knot"] = equality_6DoF_LG_knot(xdict, pdict, unit_xut, condition)
    funcs["eqcon_terminal"] = equality_6DoF_LG_terminal(xdict, pdict, unit_xut, condition)
    funcs["eqcon_rate"] = equality_6DoF_LG_rate(xdict, pdict, unit_xut, condition)

    funcs["ineqcon"] = inequality_6DoF_LG(xdict, pdict, unit_xut, condition)
    fail = False
    
    return funcs, fail

optProb = Optimization("Rocket trajectory optimization", objfunc)

optProb.addVarGroup("xvars", len(xdict_init["xvars"]), value=xdict_init["xvars"], lower=xx_lower, upper=xx_upper)
optProb.addVarGroup("uvars", len(xdict_init["uvars"]), value=xdict_init["uvars"], lower=uu_lower, upper=uu_upper)
optProb.addVarGroup("t",     len(xdict_init["t"]),     value=xdict_init["t"],     lower=t_lower,  upper=t_upper)


e_init  = equality_init(xdict_init, pdict, unit_xut, condition)
e_time  = equality_time(xdict_init, pdict, unit_xut, condition)
e_diff  = equality_6DoF_LG_diff(xdict_init, pdict, unit_xut, condition)
e_knot  = equality_6DoF_LG_knot(xdict_init, pdict, unit_xut, condition)
e_terminal  = equality_6DoF_LG_terminal(xdict_init, pdict, unit_xut, condition)
e_rate  = equality_6DoF_LG_rate(xdict_init, pdict, unit_xut, condition)

ie = inequality_6DoF_LG(xdict_init, pdict, unit_xut, condition)

#print("number of variables             : {}".format(len(xdict_init["xvars"])+len(xdict_init["uvars"])+len(xdict_init["t"])))
#print("number of equality constraints  : {}".format(len(e)))
#print("number of inequality constraints: {}".format(len(ie)))

optProb.addConGroup("eqcon_init", len(e_init), lower=0.0, upper=0.0)
optProb.addConGroup("eqcon_time", len(e_time), lower=0.0, upper=0.0)
optProb.addConGroup("eqcon_diff", len(e_diff), lower=0.0, upper=0.0)
optProb.addConGroup("eqcon_knot", len(e_knot), lower=0.0, upper=0.0)
optProb.addConGroup("eqcon_terminal", len(e_terminal), lower=0.0, upper=0.0)
optProb.addConGroup("eqcon_rate", len(e_rate), lower=0.0, upper=0.0)
optProb.addConGroup("ineqcon", len(ie), lower=0.0, upper=None)
optProb.addObj("obj")

timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

options_IPOPT = settings["IPOPT"]
options_IPOPT["output_file"] = "{}_{}_pyIPOPT.out".format(settings["name"],timestamp)
opt = IPOPT(options=options_IPOPT)

sol = opt(optProb, sens="FD")

# Post processing

for i,p in enumerate(pdict["params"]):
    if not p["timeFixed"]:
        p["timeAt_sec"] = sol.xStar["t"][i] * unit_xut["t"]
        
flag_savefig = False
# ========================
# Post Process
# ------------------------
# Convert parameter vector to variable

xvars_res = sol.xStar["xvars"] * xx_scale
x_res = np.reshape(xvars_res, (-1,pdict["num_states"]))
uvars_res = sol.xStar["uvars"] * uu_scale
u_res = np.reshape(uvars_res, (-1,pdict["num_controls"]))

tu_res = np.array([])
tx_res = np.array([])

for i in range(num_sections):
    to = sol.xStar["t"][i]
    tf = sol.xStar["t"][i+1]
    tau_x = np.hstack((-1.0, pdict["ps_params"][i]["tau"]))
    tu_res = np.hstack((tu_res, (pdict["ps_params"][i]["tau"]*(tf-to)/2 + (tf+to)/2) * unit_xut["t"]))
    tx_res = np.hstack((tx_res, (tau_x*(tf-to)/2 + (tf+to)/2) * unit_xut["t"]))

# output raw data

#with open("{}_{}_pyIPOPT_xres.csv".format(mission_name,timestamp), "w") as f1:
#    writer = csv.writer(f1)
#    writer.writerows(np.hstack((tx_res.reshape(-1,1),x_res)))
    
#with open("{}_{}_pyIPOPT_ures.csv".format(mission_name,timestamp), "w") as f2:
#    writer = csv.writer(f2)
#    writer.writerows(np.hstack((tu_res.reshape(-1,1),u_res)))
    

plt.figure()
plt.title("Mass[kg]")
plt.plot(tx_res,x_res[:,0], '.-', lw=0.8)
plt.grid()
plt.xlim([0,None])
plt.ylim([0,None])
if flag_savefig:
    plt.savefig("figures/mass.png")

print("initial mass          : {} kg".format(x_res[0,0]))
print("payload + fairing     : {} kg".format(x_res[0,0] - m_init))

plt.figure()
plt.title("Target rate[deg/s]")
plt.plot(tu_res,u_res, '.-', lw=0.8, label=["roll", "pitch", "yaw"])
plt.grid()
plt.legend()
plt.xlim([0,None])
if flag_savefig:
    plt.savefig("figures/omega.png")

out = output_6DoF(x_res, u_res, tx_res, tu_res, pdict)

out.to_csv("{}_{}_pyIPOPT_result.csv".format(settings["name"], timestamp))