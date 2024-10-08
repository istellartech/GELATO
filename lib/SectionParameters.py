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
from .PSfunctions import differentiation_matrix_LGR, nodes_LGR


class PSparams:
    def __init__(self, num_nodes: "list[int]"):
        self._num_sections = len(num_nodes)
        self._num_nodes = [n for n in num_nodes]
        self._tau = [nodes_LGR(n) for n in num_nodes]
        self._D = [differentiation_matrix_LGR(n) for n in num_nodes]
        self._index_start_u = [
            sum(self._num_nodes[:i]) for i in range(self._num_sections)
        ]
        self._N = sum(self._num_nodes)

    def tau(self, i):
        if i < 0 or i >= self._num_sections:
            raise ValueError("Index out of range")
        return self._tau[i]

    def D(self, i):
        if i < 0 or i >= self._num_sections:
            raise ValueError("Index out of range")
        return self._D[i]

    def index_start_u(self, i):
        return self._index_start_u[i]

    def index_end_u(self, i):
        return self.index_start_u(i) + self._num_nodes[i]

    def index_start_x(self, i):
        return self._index_start_u[i] + i

    def index_end_x(self, i):
        return self.index_start_x(i) + self._num_nodes[i] + 1

    def num_u(self):
        return self._N

    def num_x(self):
        return self._N + self._num_sections

    def num_sections(self):
        return self._num_sections

    def nodes(self, i):
        if i < 0 or i >= self._num_sections:
            raise ValueError("Index out of range")
        return self._num_nodes[i]

    def time_nodes(self, i, to, tf):
        t = np.zeros(self._num_nodes[i] + 1)
        t[0] = to
        t[1:] = self.tau(i) * (tf - to) / 2 + (tf + to) / 2
        return t

    def get_index(self, section):
        """get set of index for a given section.

        Args:
            section (int): section number

        Returns:
            ua (int): start index for u
            ub (int): end index for u
            xa (int): start index for x
            xb (int): end index for x
            n (int): number of nodes in the section
        """

        ua = self._index_start_u[section]
        n = self._num_nodes[section]
        ub = ua + n
        xa = ua + section
        xb = xa + n + 1

        return ua, ub, xa, xb, n

    def __getitem__(self, i):
        """get item of PSparams.(back compatibility)"""
        if i < 0 or i >= self._num_sections:
            raise ValueError("Index out of range")
        return {
            "index_start": self._index_start_u[i],
            "nodes": self._num_nodes[i],
            "D": self._D[i],
            "tau": self._tau[i],
        }
