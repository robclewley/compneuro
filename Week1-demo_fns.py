"""
Displaying a simple text-based grid
"""

from __future__ import division

from PyDSTool import *

# Functions version

n = 10
joiner = 'o'

def print_Hline(joiner, fill, n):
    print joiner + fill*n + joiner + fill*n + joiner


def print_box(joiner, fill, n):
    """
    Print a window-like box ...
    """

    # first line
    print_Hline(joiner, '-', n)

    # n lines down
    for i in range(n):
        print_Hline('|', fill, n)

    # middle line
    print_Hline(joiner, '-', n)

    # n lines down
    for i in range(n):
        print_Hline('|', fill, n)

    # last line
    print_Hline(joiner, '-', n)


print_box(joiner, " ", 10)