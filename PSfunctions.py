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

from scipy import special
import numpy as np


def LegendreFunction(x, n):
    """ Legendre function of the first kind Pn(x).

    Args:
        x (float64) : argument of the Legendre function
        n (int) : degree of the Legendre function

    Returns:
        float64 : the value of the nth function at x
    """

    Legendre, Derivative = special.lpn(n, x)
    return Legendre[-1]
    
def lagrange(tn,k,t):
    """ Lagrange basis polynominal.
    
    Args:
        tn (ndarray) : data points
        k (int) : degree of the polynominal
        t (float64) : argument

    Returns:
        float64 : value of the kth polynominal at t
    """
    l = 1.0
    N = len(tn)
    for i in range(N):
        if i != k:
            l = l * (t - tn[i]) / (tn[k] - tn[i])
    return l

def lagrangeD(tn,k,t):
    """ Differentiation of Lagrange basis polynominal.
    
    Args:
        tn (ndarray) : data points
        k (int) : degree of the polynominal
        t (float64) : argument

    Returns:
        float64 : differential coefficient of the kth polynominal at t
    """
    N = len(tn)
    den = 1.0
    for i in range(N):
        if i!=k:
            den = den * (tn[k] - tn[i])
    num = 0.0
    for j in range(N):
        num_j = 1.0
        if j!=k:
            for i in range(N):
                if i!=k and i!=j:
                    num_j = num_j * (t - tn[i])
            num = num + num_j
    return num/den

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
    """ Legendre-Gauss(LG) points"""
    return special.roots_legendre(n)[0]

def weight_LG(n):
    """ Legendre-Gauss(LG) weights."""
    return special.roots_legendre(n)[1]

def differentiation_matrix_LG(n):
    """ Legendre-Gauss(LG) differentiation matrix."""
    tk_lg,_=special.roots_legendre(n)
    tk_lg = np.hstack((-1.0,tk_lg))
    D = np.zeros((n,n+1))
    for k in range(1,n+1):
        for i in range(n+1):
            D[k-1,i] = lagrangeD(tk_lg, i, tk_lg[k])
    return D

def nodes_LGR(n, reverse=True):
    """ Legendre-Gauss-Radau(LGR) points.
    
    Args:
        n (int) : number of degrees. (n >= 2)
        reverse (boolean) : type of LGR points. The return value
        includes -1 when reverse is false and it includes +1 when
        reverse is true.

    Returns:
        ndarray: LGR points.
    
    """

    roots, weight = special.j_roots(n-1, 0, 1)
    nodes = np.hstack((-1, roots))
    if reverse:
        return np.sort(-nodes)
    else:
        return nodes
    
def weight_LGR(n):
    """ Legendre-Gauss-Radau(LGR) weights."""
    nodes = nodes_LGR(n)
    w = np.zeros(0)
    for i in range(n):
        w = np.append(w, (1-nodes[i])/(n*n*LegendreFunction(nodes[i], n-1)**2))
    return w
    
def differentiation_matrix_LGR(n,reverse=True):
    """ Legendre-Gauss-Radau(LGR) differentiation matrix.
    
    Args:
        n (int) : number of degrees. (n >= 2)
        reverse (boolean) : type of LGR points. The LGR node
        includes -1 when reverse is false and it includes +1 when
        reverse is true.

    Returns:
        ndarray: differentiation matrix (n * n+1).
    
    """

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