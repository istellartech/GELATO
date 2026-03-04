#
# The MIT License
#
# Copyright (c) 2022-2025 Interstellar Technologies Inc.
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


def get_index_event(pdict, section_name, key):
    """
    get index

    Args:
        pdict(dict) : parameters
        section_name(str) : section name
        key(str) : "mass", "position", "velocity", "quaternion", "u" or "t"

    Returns:
        out(float or ndarray) : array of variables
    """

    # read section number
    section_num = pdict["event_index"][section_name]

    if key == "t":
        index_start = section_num
        index_end = section_num + 1

    else:
        # sample variables in specified section

        ua, ub, xa, xb, _ = pdict["ps_params"].get_index(section_num)

        if key == "u":
            index_start = ua * 3
            index_end = ub * 3
        else:
            if key in ["position", "velocity"]:
                index_start = xa * 3
                index_end = xb * 3
            elif key == "mass":
                index_start = xa
                index_end = xb
            elif key == "quaternion":
                index_start = xa * 4
                index_end = xb * 4
            else:
                raise ValueError(
                    f"Unsupported key {key!r} in get_index_event; expected one of "
                    "'mass', 'position', 'velocity', 'quaternion', 'u', or 't'."
                )

    return index_start, index_end


def get_value(xdict, pdict, unitdict, section_name, key):
    """
    get vector of states, controls and time at the specified knot.

    Args:
        xdict(dict) : variables(dimensionless)
        pdict(dict) : parameters
        unitdict(dict) : unit of the variables
        section_name(str) : section name
        key(str) : "mass", "position", "velocity", "quaternion", "u" or "t"

    Returns:
        out(float or ndarray) : array of variables
    """

    # index number
    index_start, _ = get_index_event(pdict, section_name, key)

    if key == "t":
        out = xdict[key][index_start] * unitdict[key]
    elif key == "mass":
        # mass is a scalar value at the specified event
        out = xdict[key][index_start] * unitdict.get(key, 1.0)
    else:
        if key == "quaternion":
            # quaternion is treated as unitless by default
            out = xdict[key][index_start : (index_start + 4)] * unitdict.get(key, 1.0)
        else:  # position, velocity or u
            out = xdict[key][index_start : (index_start + 3)] * unitdict[key]

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
    index = [
        i for i, value in enumerate(pdict["params"]) if value["name"] == section_name
    ][0]
    n = pdict["ps_params"][index]["nodes"]

    if key == "t":
        t = xdict[key] * unitdict[key]
        to = t[index]
        tf = t[index + 1]
        t_nodes = np.zeros(n + 1)
        t_nodes[0] = to
        t_nodes[1:] = (
            pdict["ps_params"][index]["tau"] * (tf - to) / 2.0 + (tf + to) / 2.0
        )
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
