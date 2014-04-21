"""
Wilson-Cowan cortex models
Chapter 7.4
"""

from __future__ import division
from PyDSTool import *

# ---------------------------------

### User settings

# domain size (index number)
size = 200  #  ensure even

# stimulus controls
#width_microns = 1000   # 100 - 1600
stim_time = 5
# spatial frequency (integer 0-50) of stimulus
freq = 20
P0 = 31.5
Q0 = 32.5
amp = 10

bEE = 1.95
bEI = 1.4
bII = 2.2

sigEE = 40
sigEI = 60
sigII = 30

# Background input level to I cells
Q = 0

# End integration time
t1 = 300
# timestep in ms
dt = 0.5

# initial conditions (spatially homogeneous)
EE0 = 0.25
IN0 = 3.85


# ---------------------------------
# Convolution functions

def neural_conv(fltr, inputs):
    """
    Convolves fltr with inputs and removes extraneous values
    fltr is assumed to have an odd number of elements and be centered
    Replicates inputs for periodic boundary conditions
    """
    s = len(inputs)
    x = np.convolve(fltr, inputs)
    extra = int(floor(len(fltr)/2))
    return x[extra: len(x) - extra]

def circle_conv(fltr, inputs):
    """
    Convolves fltr with inputs and removes extraneous values
    fltr is assumed to have an odd number of elements and be centered
    Replicates inputs for periodic boundary conditions
    """
    s = len(inputs)
    x = np.convolve(fltr, concatenate((inputs, inputs, inputs)))
    extra = int(floor(len(fltr)/2))
    x = x[extra: len(x) - extra]
    return x[len(inputs): 2*len(inputs)]

# ---------------------------------

EE = EE0*ones(size)
IN = IN0*ones(size)

dx = 20   # microns
x = dx*arange(size)

# normalize freq by size of domain (microns)
freq_n = freq/max(x)

Del = 10

stim = ones(size)

# synaptic space
Xsyn = dx*(arange(31)-15)
# pre-computed distribution of synaptic weights
synEE = bEE*np.exp(-abs(Xsyn)/sigEE)
synEI = bEI*np.exp(-abs(Xsyn)/sigEI)
synII = bII*np.exp(-abs(Xsyn)/sigII)

num_points = int(ceil(t1/dt))
ts = linspace(0, t1, num_points)

# time array of states of all positions along domain
EEs = zeros((size, num_points))
IIs = zeros((size, num_points))


for i, t in enumerate(ts):
    if t < stim_time:
        P = P0*stim + amp*cos(2*pi*freq_n*x)
    else:
        P = P0*stim
    Q = Q0*stim
    EEresp = circle_conv(synEE, EE) - circle_conv(synEI, IN) + P
    EEresp = (EEresp * (EEresp > 0)) ** 2
    INresp = circle_conv(synEI, EE) - circle_conv(synII, IN) + Q
    INresp = (INresp * (INresp > 0)) ** 2
    # Euler step
    EE = EE + (dt/Del)*(-EE + 100*EEresp/(20*20 + EEresp))
    IN = IN + (dt/Del)*(-IN + 100*INresp/(40*40 + INresp))
    EEs[:,i] = EE
    IIs[:,i] = IN

def plot_fig(t, fignum=1):
    plt.figure(fignum)
    plt.clf()
    plt.xlabel('Position in microns')
    plt.ylabel('Spike rate activity')
    tix = find(ts, t, 1)
    plt.plot(EEs[:,tix], label='E')
    plt.plot(IIs[:,tix], label='I')
    plt.legend()

plot_fig(20)
plt.show()