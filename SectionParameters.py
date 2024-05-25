import numpy as np
from PSfunctions import differentiation_matrix_LGR, nodes_LGR

class PSparams:
    def __init__(self, num_nodes: 'list[int]'):
        self._num_sections = len(num_nodes)
        self._num_nodes = num_nodes
        self._tau = [nodes_LGR(n) for n in num_nodes]
        self._D = [differentiation_matrix_LGR(n) for n in num_nodes]
        self._index_start_u = [sum(self._num_nodes[:i]) for i in range(self._num_sections)]
        self._N = sum(self._num_nodes)

    def tau(self, i):
        if i < 0 or i >= self._num_sections:
            raise ValueError('Index out of range')
        return self._tau[i]

    def D(self, i):
        if i < 0 or i >= self._num_sections:
            raise ValueError('Index out of range')
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
            raise ValueError('Index out of range')
        return self._num_nodes[i]
