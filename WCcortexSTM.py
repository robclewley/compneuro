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
width_microns = 100   # 100 gives single peak, 1000 gives two
stim_time = 10

bEE = 1.95
bEI = 1.4
bII = 2.2

sigEE = 40
sigEI = 60
sigII = 30

# Background input level to I cells
Q = 0

# End integration time
t1 = 80
# timestep in ms (approx)
dt = 0.5

# initial conditions (spatially homogeneous)
EE0 = 0
IN0 = 0


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

dx = 20.0   # microns
x = dx*arange(size)

Del = 10

stim = zeros(size)
# stim width
width = int(round(width_microns/(2.0*dx)))
stim[int(size/2 - width) : int(size/2 + width)] = ones(2*width)


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
    P = stim*(t < stim_time)
    EEresp = neural_conv(synEE, EE) - neural_conv(synEI, IN) + P
    EEresp = (EEresp * (EEresp > 0)) ** 2
    INresp = neural_conv(synEI, EE) - neural_conv(synII, IN) + Q
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
    plt.plot(x, EEs[:,tix], label='E')
    plt.plot(x, IIs[:,tix], label='I')
    plt.legend()

##plot_fig(40)
##
##plt.figure(2)
##plt.plot(ts,EEs[50,:])
##plt.plot(ts,IIs[50,:])
##
##plt.figure(3)
##plt.plot(Xsyn, synEE)
##plt.plot(Xsyn, synII)
##plt.plot(Xsyn, synEI)

plt.figure(1)
plt.show()


from time import sleep
for t in ts:
    plot_fig(t)
    plt.ylim([0,100])
    plt.text(5, 55, '%.3f'%t, fontsize='large')
    plt.draw()
    sleep(0.05)
