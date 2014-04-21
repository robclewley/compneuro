"""Exploring Naka-Rushton functions

Section 2.3
"""

from __future__ import division
from PyDSTool import *
from matplotlib.pyplot import *

M = 100
N = 4
s = 50

# These functions are safe for negative inputs

def NR(p):
    """Vectorizable, can accept scalars or arrays"""
    # Could also use p**N, which works for arrays
    return np.asarray(p>0,int) * (M*np.power(p,N)/(np.power(s,N)+np.power(p,N)))


def NR_scalar(p):
    """Not vectorizable.
    pow is for scalars only, and if statement"""
    if p > 0:
        return M*pow(p,N)/(pow(s,N)+pow(p,N))
    else:
        return 0


if __name__ == '__main__':
    Ps = linspace(-20, 140, 50)
    N = 1
    Ss_1 = NR(Ps)
    N = 4
    Ss_4 = NR(Ps)

    plot(Ps, Ss_1, label='N=1')
    plot(Ps, Ss_4, label='N=4')

    legend(loc='lower right')
    show()