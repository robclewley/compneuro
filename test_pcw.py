from PyDSTool import *
from common_lib import *

thresh_ev = Events.makeZeroCrossEvent('x-0.5', 1,
                                       {'name': 'ev',
                                        'eventtol': 1e-4,
                                        'precise': True,
                                        'term': False},
                                       varnames=['x'])

unused_ev = Events.makeZeroCrossEvent('x-100', 1,
                                       {'name': 'ev2',
                                        'eventtol': 1e-4,
                                        'precise': True,
                                        'term': False},
                                       varnames=['x'])

DSargs = {'tdomain': [0,20],
          'pars': {'k':0, 'a':0},
          'algparams': {'init_step':0.01, 'strict':False},
          'name': 'pcwtest',
          'varspecs': {"x": "k*sin(t*pi)+a"},
          'events': [thresh_ev, unused_ev]
          }
testODE = Vode_ODEsystem(DSargs)

# Piecewise protocol
protocol = []

protocol.append({'ics': {'x': 0},
                 'pars': {'k': 0, 'a': 1},
                 'tdur': 1})

protocol.append({'ics': {'x': 0},
                 'pars': {'k': 0, 'a': 1},
                 'tdur': 1})

protocol.append({'ics': {'x': 0},
                 'pars': {'k': 0, 'a': 1},
                 'tdur': 1})

traj, pts = pcw_protocol(testODE, protocol)
plot(pts['t'], pts['x'])
show()