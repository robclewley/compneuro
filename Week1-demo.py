"""
Displaying a simple text-based grid
"""

from __future__ import division

from PyDSTool import *

# Script version

n = 5
#joiner = '+'

# first line
print "+" + "-"*n + "+" + "-"*n + "+"

# n lines down
for i in range(n):
    print "|" + " "*n + "|" + " "*n + "|"

# middle line
print "+" + "-"*n + "+" + "-"*n + "+"

# n lines down
for i in range(n):
    print "|" + " "*n + "|" + " "*n + "|"

# last line
print "+" + "-"*n + "+" + "-"*n + "+"
