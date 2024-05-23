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


def cost_6DoF(xdict, condition):
    """Objective function of the optimization problem."""
    if condition["OptimizationMode"] == "Payload":
        return -xdict["mass"][0]  # 初期質量(無次元)を最大化
    else:
        return xdict["t"][-1]  # 到達時間を最小化(=余剰推進剤を最大化)


def cost_jac(xdict, condition):
    """Gradient of the objective function."""

    jac = {}
    if condition["OptimizationMode"] == "Payload":
        jac["mass"] = np.zeros(xdict["mass"].size)
        jac["mass"][0] = -1.0
    else:
        jac["t"] = np.zeros(xdict["t"].size)
        jac["t"][-1] = 1.0
    return jac
