from __future__ import division    # ensures 1/2 is 0.5 not 0

# Lines starting with '#' are comments (not executed by python interpreter)

# Get access to all the library functions we need using import
# * is a wildcard (matches all available functions, constants, etc.)
from numpy import *
# linalg is a sub-package of numpy, referenced using the 'dot' notation
from numpy.linalg import *
# import specific plotting commands from matplotlib (via its pylab interface)
from pylab import plot, title, draw, show

print "Everything imported OK"


print "\nConsistent indentation is essential in python!"
# for loops: "i from 0 to 9" (10 total)
for i in range(10):
    print "For loop i=", i

# no end statement was needed because of the indentation change

print "Countdown ",
for i in [5,4,3,2,1,0]:
    print i,

# variables
x = 5
p = x
print p, x
x = 6
print p, x

# functions
def chatty_add(x, y):
    # %f is a format code to tell print to expect floating point number
    print "Adding %f and %f ->" % (x, y),
    # result variable is only local to the function, it disappears after
    # the call is complete
    result = x + y
    print "Result is", result
    return result

# assignment to the result of function call
# (arguments are evaluated before passing to function)
# ... is an example of function composition
sum = chatty_add(pi, sqrt(4))

# make some space in the output (\n encodes "newline")
print "\n"

# floating point numbers and common functions
print 0.1 * sqrt(10) + min([5,2]) / 6
print 2**4, 2**(-4), e*exp(-1)
print "Boolean operations return True or False:"
print "DOUBLE equals sign for comparison: x == 5 ->", x == 5

print "\nConditional statements in a function:"
def is_bigger(x):
    # no type declaration: python uses "weak" (non-strict) typing
    # you'll get an error raised if x has a type that won't compare with an
    # integer
    if x > 5:
        print "Argument was greater than 5"
    elif x == 5:
        print "Argument was equal to 5"
    else:
        print "Argument was less than 5"
    # no need for an 'end' statement -- that's what the indentation is for

print is_bigger(1)
print is_bigger(10)

print "\nWhile loop in a function to compute the floor of the log-2:"
def twolog(n):
    l = 0
    m = 2
    while m < n:
        l += 1  # increment l
        m *= 2  # double m
    # no need for an 'end' statement
    # without return statement, nothing is returned!
    return l


print "twolog(50.5) =", twolog(50.5)
print "Compare with regular log without floor: log2(50.5) =", log2(50.5)
print "\nN.B. Log base 10 is log(x) or log10(x)"


# lists are specified by square braces (these are NOT quite proper arrays/vectors)
# they can store anything, and keep their order.
menu = ['spam', 'eggs', 'ham']    # assign a list of strings
print "\n\nMenu list: ", menu
print "I forgot more spam. Let's append it to the list using menu.append"
# append is a "method" of the list object "menu"
menu.append('spam')
print "That's better: ", menu
print "(list entries need not be unique)"

print "\nN.B. '1.0' is a string, not a number"

# can mix types in python lists
assorted = [0, 3, 'hello', -1, sin(pi/2), 0.1, [1,2], min]
# and use them for iteration
for item in assorted:
    # \t represents a tab
    print type(item), '\t\t', item

# sorting on numeric lists is *in place*
nums = [3, 4, 1, 0, 10, 8]
print "\nUnsorted numbers:", nums
nums.sort()
print "They were sorted in place with nums.sort(), i.e. which did not return the new value"
print "Sorted numbers:", nums


print "\nLooping the classic way with a counter, displaying with a table\n"
print "Counter:\tType:\t\t\t\tValue:"
i = 0
# this is less efficient and less easy to read
for i in range(len(assorted)):
    # note lists can be indexed, and indices ***** start at 0 *****
    item = assorted[i]
    type_name = type(item).__name__
    # you can ignore this next line, which ensures even tabbing!
    tabs = max( (1, 4 - len(type_name) // 7) )
    # %i prints an integer
    print "i=%i" % i, "\t\t", type_name, "\t"*tabs, item

print "\nrange(5) is really just a list object:", range(5)
print "and range(5)[::-1] reverses it:", range(5)[::-1]
print "Extracting sub-lists: 10 numbers in the middle of range(100)"
print " -> range(100)[45:55] =", range(100)[45:55]


## Linear algebra and arrays from NumPy
print "\n\nThree proper numerical arrays (not lists):"

# these can only store numerical values
a = array([1., 2., 3.])
z = zeros(3, float)
o = 5 * ones(3, float)

print "a=", a, "  z=", z, "  o=", o

r = (a - z) / o

print "(a-z)/o =", r, ", with elements:", r[0], r[1], r[2]

# setting an element
r[0] = 10

# incrementing an element
r[1] += 5   # same as r[1] = r[1] + 5

print "Array equivalent of range is arange: arange(10) * 3.5 ->", arange(10) * 3.5

print "Concatenate arrays using concatenate( (a, b) ), as append is only for lists"
b = array([5, 6, 7])
# Notice the extra set of braces!
print "concatenate( (a, b) ) =", concatenate( (a, b) )
print "stacking: vstack( (a, b, b) ) =", vstack( (a, b, b) )
print "          hstack( (a, b) ) =", hstack( (a, b) )

print "\nMatrix:"
m = array([[3., 2., 0.],
           [0., -1., 0.5],
           [1., 0., 0.2]])

print "m =\n",m
print "Transpose of m is transpose(m) = m.T =\n", m.T
print "Inverse of m (if defined) is inv(m) =\n", inv(m)
print "Determinant is det(m) =", det(m), " ... non-zero in order that inv(m) exists"

print "Slicing vertically: m[:,2]=", m[:,2]
print "Slicing horizontally: m[0,:] = m[0] =", m[0]

# Only get *element-wise* multiplication with m * m (BE CAREFUL OF THAT)
print "Matrix multiplication: dot(m, 2*eye(3)) =\n", dot(m, 2*eye(3))
print " ... or apply matrix to vector: dot(m, array([1, 0, 1])) =", dot(m, array([1, 0, 1]))
print "Element-wise addition: m - m =\n", m - m

print "\nYou must create a ** COPY ** of a matrix or array before using it"
print "Trying b = m to create b from m, and setting first element to 100"
b = m
b[0,0] = 100
print "m[0,0] == 100? ", m[0,0] == 100
print "Oops! Let's try b = m.copy()"
lin
b = m.copy()
b[0,0] = -5
print "m[0,0] == b[0,0]? ", m[0,0] == b[0,0]
print "That's better (but we didn't go back and fix m)\n"

from numpy.random import random
print "Four uniformly random numbers from [0, 1) as a 2 x 2 matrix:\n", random((2,2))

# Many more examples with matlab equivalents at:
# http://www.scipy.org/NumPy_for_Matlab_Users



### PLOTTING
print "\n\nPlot a line of 2D data for sqrt function (using 'linspace'):"
x_data = concatenate( (linspace(0, 0.9, 10), linspace(1, 10, 20)) )
y_data = sqrt(x_data)  # function can accept arrays
print "x:", x_data, "\ny:", y_data

plot(x_data, y_data, 'ko-')   # uses same format codes as matlab
title('sqrt function')

plot(x_data, -y_data, 'ro-')  # fills in the negative part using the positive part
show()
