#
# Legendre-Gauss-Radau pseudospectral method utilities.
# Adapted from GELATO2/lib/PSfunctions.py and SectionParameters.py
#

import numpy as np
from scipy import special


def nodes_LGR(n, reverse=True):
    """Legendre-Gauss-Radau (LGR) points.

    Args:
        n: number of nodes (>= 2)
        reverse: if True, includes +1 (right end); if False, includes -1 (left end)

    Returns:
        LGR node locations in [-1, 1]
    """
    roots, _ = special.j_roots(n - 1, 0, 1)
    nodes = np.hstack((-1, roots))
    if reverse:
        return np.sort(-nodes)
    else:
        return nodes


def _lagrangeD(tn, k, t):
    """Derivative of k-th Lagrange basis polynomial at t."""
    N = len(tn)
    den = 1.0
    for i in range(N):
        if i != k:
            den *= tn[k] - tn[i]
    num = 0.0
    for j in range(N):
        num_j = 1.0
        if j != k:
            for i in range(N):
                if i != k and i != j:
                    num_j *= t - tn[i]
            num += num_j
    return num / den


def differentiation_matrix_LGR(n, reverse=True):
    """LGR differentiation matrix (n x n+1).

    Maps from (n+1) state values (boundary + LGR nodes) to n derivatives at LGR nodes.
    """
    tk = nodes_LGR(n, reverse)
    if reverse:
        tk = np.hstack((-1.0, tk))
    else:
        tk = np.hstack((tk, 1.0))
    D = np.zeros((n, n + 1))
    for k in range(n):
        for i in range(n + 1):
            if reverse:
                D[k, i] = _lagrangeD(tk, i, tk[k + 1])
            else:
                D[k, i] = _lagrangeD(tk, i, tk[k])
    return D


class PSparams:
    """Pseudospectral parameter manager for multi-section trajectory."""

    def __init__(self, num_nodes: list):
        self._num_sections = len(num_nodes)
        self._num_nodes = list(num_nodes)
        self._tau = [nodes_LGR(n) for n in num_nodes]
        self._D = [differentiation_matrix_LGR(n) for n in num_nodes]
        self._index_start_u = [
            sum(self._num_nodes[:i]) for i in range(self._num_sections)
        ]
        self._N = sum(self._num_nodes)

    def tau(self, i):
        return self._tau[i]

    def D(self, i):
        return self._D[i]

    def index_start_u(self, i):
        return self._index_start_u[i]

    def index_end_u(self, i):
        return self.index_start_u(i) + self._num_nodes[i]

    def index_start_x(self, i):
        return self._index_start_u[i] + i

    def index_end_x(self, i):
        return self.index_start_x(i) + self._num_nodes[i] + 1

    def nodes(self, i):
        return self._num_nodes[i]

    def time_nodes(self, i, to, tf):
        t = np.zeros(self._num_nodes[i] + 1)
        t[0] = to
        t[1:] = self.tau(i) * (tf - to) / 2 + (tf + to) / 2
        return t

    def get_index(self, section):
        """Returns (ua, ub, xa, xb, n) for a given section."""
        ua = self._index_start_u[section]
        n = self._num_nodes[section]
        ub = ua + n
        xa = ua + section
        xb = xa + n + 1
        return ua, ub, xa, xb, n
