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

import numpy as np
from coordinate import orbital_elements

def get_value(xdict, pdict, unitdict, section_name, key):
    """
    get vector of states, contols and time at the specified knot.

    Args:
        xdict(dict) : variables(dimensionless)
        pdict(dict) : parameters
        unitdict(dict) : unit of the variables
        section_name(str) : section name
        key(str) : "mass", "position", "velocity", "quaternion", "u" or "t"

    Returns:
        out(float or ndarray) : array of variables
    """

    # read index number
    index = pdict["event_index"][section_name]

    if key == "t":
        out = xdict[key][index] * unitdict[key]

    else:
        # sample variables in specified section

        a = pdict["ps_params"][index]["index_start"]

        if key == "u":
            out = xdict[key][a * 3 : (a + 1) * 3] * unitdict[key]
        else:
            a2 = a + index
            if key in ["position", "velocity"]:
                out = xdict[key][a2 * 3 : (a2 + 1) * 3] * unitdict[key]
            elif key == "mass":
                out = xdict[key][a2] * unitdict[key]
            elif key == "quaternion":
                out = xdict[key][a2 * 4 : (a2 + 1) * 4]

    return out


def get_values_section(xdict, pdict, unitdict, section_name, key):
    """
    get vector of states, contols and time in the specified section.

    Args:
        xdict(dict) : variables(dimensionless)
        pdict(dict) : parameters
        unitdict(dict) : unit of the variables
        section_name(str) : section name
        key(str) : "mass", "position", "velocity", "quaternion", "u" or "t"

    Returns:
        out(ndarray) : array of variables
    """

    # read index number
    index = [i for i, value in enumerate(pdict["params"]) if value["name"] == section_name][0]
    n = pdict["ps_params"][index]["nodes"]

    if key == "t":
        t = xdict[key] * unitdict[key]
        to = t[index]
        tf = t[index + 1]
        t_nodes = np.zeros(n + 1)
        t_nodes[0] = to
        t_nodes[1:] = pdict["ps_params"][index]["tau"] * (tf - to) / 2.0 + (tf + to) / 2.0
        out = t_nodes  # time array (includes initial point)

    else:
        if key == "mass":
            val_ = xdict[key] * unitdict[key]
        elif key == "quaternion":
            val_ = xdict[key].reshape(-1, 4)
        else:  # position, velocity or u
            val_ = xdict[key].reshape(-1, 3) * unitdict[key]

        # sample variables in specified section

        a = pdict["ps_params"][index]["index_start"]
        a2 = a + index

        if key == "u":
            out = val_[a : a + n]
        else:
            out = val_[a2 : a2 + n + 1]

    return out


def equality_user(xdict, pdict, unitdict, condition):
    """
    set additional  equality constraints.

    the return values of this function will be constrained to zero.

    it is strongly recommended to normalize the return values.

    the type of return value is float64 or numpy.ndarray(float64)

    """

    # get state value vector
    pos_IIP0 = get_value(xdict, pdict, unitdict, "IIP_END", "position")
    vel_IIP0 = get_value(xdict, pdict, unitdict, "IIP_END", "velocity")

    elem = orbital_elements(pos_IIP0, vel_IIP0)
    ha = (elem[0] * (1.0 - elem[1]) / 6378137.0) - 1.0  # height of apogee at the event IIP_END

    return ha


def inequality_user(xdict, pdict, unitdict, condition):
    """
    set additional inequality constraints.

    the return values of this function will be constrained to positive or zero.

    it is strongly recommended to normalize the return values.

    the type of return value is float64 or numpy.ndarray(float64)

    """

    return None
