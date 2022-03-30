import numpy as np
from utils import ecef2geodetic

def equality_user(xdict, pdict, unitdict, condition):
    """
    set additional  equality constraints.

    the return values of this function will be constrained to zero.

    it is strongly recommended to normalize the return values.

    the type of return value is float64 or numpy.ndarray(float64)

    """
    # get state value vector

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

    num_sections = pdict["num_sections"]
    
    # read index number
    index = [i for i,value in enumerate(pdict["params"]) if value["name"] == "FRGDR"][0]
    
    # sample variables in specified section

    a = pdict["ps_params"][index]["index_start"]
    a2 = a + index
    n = pdict["ps_params"][index]["nodes"]
    mass_i_ = mass_[a2:a2+n+1] # mass array
    pos_i_ = pos_[a2:a2+n+1]   # position 2d array
    vel_i_ = vel_[a2:a2+n+1]   # velocity 2d array
    quat_i_ = quat_[a2:a2+n+1] # quaternion 2d array
    u_i_ = u_[a:a+n]           # control 2d array
    to = t[index]
    tf = t[index+1]
    t_nodes = pdict["ps_params"][index]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
    t_i_ = np.hstack((to, t_nodes)) # time array (includes initial point)

    # user-defined section

    _,_,alt_m_FRGDR = ecef2geodetic(pos_i_[0,0], pos_i_[0,1], pos_i_[0,2])
    alt_m_FRGDR_value = 130000.0

    #return (alt_m_FRGDR / alt_m_FRGDR_value) - 1.0
    return None

def inequality_user(xdict, pdict, unitdict, condition):
    """
    set additional inequality constraints.

    the return values of this function will be constrained to positive or zero.

    it is strongly recommended to normalize the return values.

    the type of return value is float64 or numpy.ndarray(float64)

    """
    # get state value vector

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

    num_sections = pdict["num_sections"]
    
    # read index number
    index = [i for i,value in enumerate(pdict["params"]) if value["name"] == "TGTE"][0]
    
    # sample variables in specified section

    a = pdict["ps_params"][index]["index_start"]
    a2 = a + index
    n = pdict["ps_params"][index]["nodes"]
    mass_i_ = mass_[a2:a2+n+1] # mass array
    pos_i_ = pos_[a2:a2+n+1]   # position 2d array
    vel_i_ = vel_[a2:a2+n+1]   # velocity 2d array
    quat_i_ = quat_[a2:a2+n+1] # quaternion 2d array
    u_i_ = u_[a:a+n]           # control 2d array
    to = t[index]
    tf = t[index+1]
    t_nodes = pdict["ps_params"][index]["tau"] * (tf-to) / 2.0 + (tf+to) / 2.0
    t_i_ = np.hstack((to, t_nodes)) # time array (includes initial point)

    # user-defined section

    to_lower = 80.0
    to_upper = 95.0

    ret = np.array([to - to_lower, to_upper - to]) / unit_t

    return ret
