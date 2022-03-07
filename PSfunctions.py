from pyoptsparse import IPOPT,SLSQP, Optimization
from scipy import special,optimize,integrate
import numpy as np
import matplotlib.pyplot as plt


def LegendreFunction(x, n):
        Legendre, Derivative = special.lpn(n, x)
        return Legendre[-1]
    
def lagrange(tn,k,t):
    l = 1.0
    N = len(tn)
    for i in range(N):
        if i != k:
            l = l * (t - tn[i]) / (tn[k] - tn[i])
    return l

def lagrangeD(tn,k,t):
    dt = 0.001
    lp = lagrange(tn,k,t+dt*0.5)
    lm = lagrange(tn,k,t-dt*0.5)
    return (lp-lm)/dt

def nodes_LGL(n):
    """ Legendre-Gauss-Lobatto(LGL) points"""
    roots, weight = special.j_roots(n-2, 1, 1)
    nodes = np.hstack((-1, roots, 1))
    return nodes

def weight_LGL(n):
    """ Legendre-Gauss-Lobatto(LGL) weights."""
    nodes = nodes_LGL(n)
    w = np.zeros(0)
    for i in range(n):
        w = np.append(w, 2/(n*(n-1)*LegendreFunction(nodes[i], n-1)**2))
    return w

def differentiation_matrix_LGL(n):
    """ Legendre-Gauss-Lobatto(LGL) differentiation matrix."""
    tau = nodes_LGL(n)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = LegendreFunction(tau[i], n-1) \
                          / LegendreFunction(tau[j], n-1) \
                          / (tau[i] - tau[j])
            elif i == j and i == 0:
                D[i, j] = -n*(n-1)*0.25
            elif i == j and i == n-1:
                D[i, j] = n*(n-1)*0.25
            else:
                D[i, j] = 0.0
    return D

def nodes_LG(n):
    return special.roots_legendre(n)[0]

def weight_LG(n):
    return special.roots_legendre(n)[1]

def differentiation_matrix_LG(n):
    tk_lg,_=special.roots_legendre(n)
    tk_lg = np.hstack((-1.0,tk_lg))
    D = np.zeros((n,n+1))
    for k in range(1,n+1):
        for i in range(n+1):
            D[k-1,i] = lagrangeD(tk_lg, i, tk_lg[k])
    return D

def nodes_LGR(n, reverse=True):
    roots, weight = special.j_roots(n-1, 0, 1)
    nodes = np.hstack((-1, roots))
    if reverse:
        return np.sort(-nodes)
    else:
        return nodes
    
def weight_LGR(n):
    nodes = nodes_LGR(n)
    w = np.zeros(0)
    for i in range(n):
        w = np.append(w, (1-nodes[i])/(n*n*LegendreFunction(nodes[i], n-1)**2))
    return w
    
def differentiation_matrix_LGR(n,reverse=True):
    tk_lgr = nodes_LGR(n,reverse)
    if reverse:
        tk_lgr = np.hstack((-1.0,tk_lgr))
    else:
        tk_lgr = np.hstack((tk_lgr, 1.0))
    D = np.zeros((n,n+1))
    for k in range(n):
        for i in range(n+1):
            if reverse:
                D[k,i] = lagrangeD(tk_lgr, i, tk_lgr[k+1])
            else:
                D[k,i] = lagrangeD(tk_lgr, i, tk_lgr[k])
    return D