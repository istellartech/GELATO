import numpy as np
from coordinate import orbital_elements

def get_values(xdict, pdict, unitdict, section_name):
    """
    get vector of states, contols and time in the specified section.

    Args:
        xdict(dict) : variables(dimensionless)
        pdict(dict) : parameters
        unitdict(dict) : unit of the variables
        section_name(str) : section name

    Returns:
        out(dict) : dict of variables
    """
    unit_mass= unitdict["mass"]
    unit_pos = unitdict["position"]
    unit_vel = unitdict["velocity"]
    unit_u = unitdict["u"]
    unit_t = unitdict["t"]

    mass_ = xdict["mass"] * unit_mass
    pos_ = xdict["position"].reshape(-1,3) * unit_pos
    vel_ = xdict["velocity"].reshape(-1,3) * unit_vel
    quat_ = xdict["quaternion"].reshape(-1,4)
    
    u_ = xdict["u"].reshape(-1,3) * unit_u
    t = xdict["t"] * unit_t
    
    # read index number
    index = [i for i,value in enumerate(pdict["params"]) if value["name"] == section_name][0]
    
    # sample variables in specified section

    a = pdict["ps_params"][index]["index_start"]
    a2 = a + index
    n = pdict["ps_params"][index]["nodes"]

    out = {}
    out["mass"] = mass_[a2:a2+n+1] # mass array
    out["position"] = pos_[a2:a2+n+1]   # position 2d array
    out["velocity"] = vel_[a2:a2+n+1]   # velocity 2d array
    out["quaternion"] = quat_[a2:a2+n+1] # quaternion 2d array
    out["u"] = u_[a:a+n]           # control 2d array
    to = t[index]
    tf = t[index+1]
    t_nodes = np.zeros(n+1)
    t_nodes[0] = to
    t_nodes[1:] = pdict["ps_params"][index]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
    out["t"] = t_nodes # time array (includes initial point)

    return out

def equality_user(xdict, pdict, unitdict, condition):
    """
    set additional  equality constraints.

    the return values of this function will be constrained to zero.

    it is strongly recommended to normalize the return values.

    the type of return value is float64 or numpy.ndarray(float64)

    """


    # get state value vector
    pos_IIP0 = get_values(xdict, pdict, unitdict, "IIP_END")["position"][0]
    vel_IIP0 = get_values(xdict, pdict, unitdict, "IIP_END")["velocity"][0]

    elem = orbital_elements(pos_IIP0, vel_IIP0)
    ha = elem[0] * (1.0 - elem[1]) - 6378137.0 # height of apogee at the event IIPEnd

    return ha

def inequality_user(xdict, pdict, unitdict, condition):
    """
    set additional inequality constraints.

    the return values of this function will be constrained to positive or zero.

    it is strongly recommended to normalize the return values.

    the type of return value is float64 or numpy.ndarray(float64)

    """

    return None
    