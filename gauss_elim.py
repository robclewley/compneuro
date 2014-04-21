"""
Written by Git user 'num3ric'
https://gist.github.com/1357315

Slightly modified to ensure float types by R. Clewley
"""
from __future__ import division
import numpy as np

def GENP(A, b):
    '''
Gaussian elimination with no pivoting.
% input: A is an n x n nonsingular matrix
% b is an n x 1 vector
% output: x is the solution of Ax=b.
'''
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)
    if b.size != n:
        raise ValueError("Invalid argument: incompatible sizes between A & b.", b.size, n)
    for pivot_row in xrange(n-1):
        for row in xrange(pivot_row+1, n):
            multiplier = A[row][pivot_row]/A[pivot_row][pivot_row]
            #the only one in this column since the rest are zero
            A[row][pivot_row] = multiplier
            for col in xrange(pivot_row + 1, n):
                A[row][col] = A[row][col] - multiplier*A[pivot_row][col]
            #Equation solution column
            b[row] = b[row] - multiplier*b[pivot_row]
    #print A
    #print b
    x = np.zeros(n)
    k = n-1
    x[k] = b[k]/A[k,k]
    while k >= 0:
        x[k] = (b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
        k = k-1
    return x

def GEPP(A, b):
    '''
Gaussian elimination with partial pivoting.
% input: A is an n x n nonsingular matrix
% b is an n x 1 vector
% output: x is the solution of Ax=b.
'''
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)
    if b.size != n:
        raise ValueError("Invalid argument: incompatible sizes between A & b.", b.size, n)
    # k represents the current pivot row. Since GE traverses the matrix in the upper
    # right triangle, we also use k for indicating the k-th diagonal column index.
    for k in xrange(n-1):
        #Choose largest pivot element below (and including) k
        maxindex = abs(A[k:,k]).argmax() + k
        if A[maxindex, k] == 0:
            raise ValueError("Matrix is singular.")
        #Swap rows
        if maxindex != k:
            A[[k,maxindex]] = A[[maxindex, k]]
            b[[k,maxindex]] = b[[maxindex, k]]
        for row in xrange(k+1, n):
            multiplier = A[row][k]/A[k][k]
            #the only one in this column since the rest are zero
            A[row][k] = multiplier
            for col in xrange(k + 1, n):
                A[row][col] = A[row][col] - multiplier*A[k][col]
            #Equation solution column
            b[row] = b[row] - multiplier*b[k]
    #print A
    #print b
    x = np.zeros(n)
    k = n-1
    x[k] = b[k]/A[k,k]
    while k >= 0:
        x[k] = (b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
        k = k-1
    return x

if __name__ == "__main__":
    A = np.array([[1.,-1.,1.,-1.],[1.,0.,0.,0.],[1.,1.,1.,1.],[1.,2.,4.,8.]])
    b = np.array([[14.],[4.],[2.],[2.]])
    print GENP(np.copy(A), np.copy(b))
    print GEPP(A,b)