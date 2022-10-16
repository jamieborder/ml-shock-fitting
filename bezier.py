import numpy as np

# https://pomax.github.io/bezierinfo/#decasteljau

m = 3

t = np.zeros((1,m+1))
A = np.zeros((m+1,m+1))

A[0,0] =  1.0
A[1,0] = -3.0
A[1,1] =  3.0
A[2,0] =  3.0
A[2,1] = -6.0
A[2,2] =  3.0
A[3,0] = -1.0
A[3,1] =  3.0
A[3,2] = -3.0
A[3,3] =  1.0

B = lambda t,P: (np.array([[1,t,t**2,t**3]]) @ A @ P)[0,:]

def genBezierPoints(p,n,t0=0.0,t1=1.0):
    '''
    m = 3
    p = np.zeros((m+1,2))
    '''
    ts = np.linspace(t0,t1,n)
    return np.array([B(t,p) for t in ts])
