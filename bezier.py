import matplotlib.pyplot as plt
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

op = np.zeros((m+1,2))
r = 1
op[0,:] = [-2*r, 0]
op[1,:] = [-2*r, r]
op[2,:] = [-r, 2.5*r]
op[3,:] = [0, 3*r]
ip = np.zeros((m+1,2))
r = 1
K = 0.5522847498
ip[0,:] = [-r, 0]
ip[1,:] = [-r, K*r]
ip[2,:] = [-K*r, r]
ip[3,:] = [ 0, r]

B = lambda t,P: (np.array([[1,t,t**2,t**3]]) @ A @ P)[0,:]

ts = np.linspace(0,1,100)
ol = np.array([B(t,op) for t in ts])
il = np.array([B(t,ip) for t in ts])

for i in range(100):
    p = np.zeros((m+1,2))
    R1 = 1
    # three controlling parameters
    R2 = np.random.random() * 0.2 + 0.9
    K1 = np.random.random()
    K2 = np.random.random() * 0.8
    # R2,K1,K2 = [float(s) for s in '1.0 0.60 0.8'.split(' ')]
    # R2 = 0.9723386671832537
    # K1 = 0.7238354991222685
    # K2 = 0.1433085789166241
    R2 = 1.089293
    K1 = 0.61379
    K2 = 0.00210
    #
    p[0,:] = [-R1, 0]
    p[1,:] = [-R1, K1*R1]
    p[2,:] = [-K2*R2, R2]
    p[3,:] = [ 0, R2]
    xy = np.array([B(t,p) for t in ts])
    plt.figure(1)
    plt.plot(xy[:,0],xy[:,1],'-',c=f'C{i}')
    plt.plot( p[:,0], p[:,1],'o',c=f'C{i}')
    print(R2,K1,K2)


plt.figure(1)
plt.plot(il[:,0],il[:,1],'r-',lw=3)
plt.plot(ol[:,0],ol[:,1],'b-',lw=3)
plt.plot(ip[:,0],ip[:,1],'ro')
plt.plot(op[:,0],op[:,1],'bo')
plt.axis('equal')
# plt.show(block=False)
plt.show()

