"""These routines assume 1D ODEs
"""

from numpy import linspace, zeros

def euler_step(DS, t, x, dt):
    """Use for PyDSTool generators given by DS argument
    """
    return x+dt*DS.Rhs(t, {DS.variables.keys()[0]: x})[0]

def euler_integrate(DS, t0, t1, dt, x0):
    ts = linspace(t0, t1, (t1-t0)/dt)
    xs = zeros(len(ts), float)
    x = x0
    for i, t in enumerate(ts):
        xs[i] = x
        x = euler_step(DS, t, x, dt)
    return ts, xs
