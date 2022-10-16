import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 16

# plotting shock shape to see what order polynomial should be used to fit to it - looks like 6th-order polynomial / Legendre / Chebyshev basis is the sweet spot (so 7 coefficients)

# for dev
from matplotlib.pyplot import *

# -1.444847941398620605e+00 0.000000000000000000e+00 0.000000000000000000e+00
pos = np.loadtxt('pos.dat')

# input dimensions:
# for this block, vertices: (41,6)
p = np.reshape(pos,(41,6,3))

if False:
    # visualise to find outer
    plt.figure(1)
    for i in range(p.shape[1]):
        plt.plot(p[:,i,0],p[:,i,1],'o-',c='k')

    plt.plot(p[:,0,0],p[:,0,1],'o-',c='r') # this is outer
    plt.plot(p[:,-1,0],p[:,-1,1],'o-',c='b')
    plt.show(block=False)


# map to polar coords
#
# x = r cos(theta)
# y = r sin(theta)
#
# theta = arctan(y/x)
# r = (x**2 + y**2)**0.5

ol = p[:,0,:]
theta = -np.arctan(ol[:,1] / ol[:,0])
r = (ol[:,0]**2 + ol[:,1]**2)**0.5

phi = 4.0 * theta / np.pi - 1.0

N = r.shape[0]
# 1, x, x^2, x^3, ...
# f(theta=t) = sum_{k=0}^{M} c_k * t**k
M = 3
Ms = [M for M in range(2,10)]
errs = np.zeros((len(Ms),2))
lgs = np.zeros((Ms[-1]+1,phi.shape[0]))
dlgs = np.zeros((Ms[-1]+1,phi.shape[0]))
for k in range(lgs.shape[0]):
    lc = np.zeros(lgs.shape[0])
    lc[k] = 1.0
    lgs[k,:] = np.polynomial.legendre.legval(phi,lc)
    dc = np.polynomial.legendre.legder(lc,m=1)
    dlgs[k,:] = np.polynomial.legendre.legval(phi,dc)
use_legendre = True
# monomial, legendre, chebyshev all give same errors!!!
for M in Ms:

    if use_legendre:
        A = lgs[:M+1,:].T
    else:
        A = np.zeros((N,M+1))
        for i in range(N):
            for k in range(A.shape[1]):
                A[i,k] = theta[i]**k

    b = r.copy()[:,None]

    # without constraint
    cs = np.linalg.solve(A.T @ A, A.T @ b)

    # could add constraint of zero gradient at start?
    # d/dt( f(theta=t=0) ) = sum_{k=1}^{M} k * c_k * x**(k-1) = 0

    if use_legendre:
        T = dlgs[:M+1,0][None,:]
    else:
        T = np.zeros((1,M+1))
        for k in range(1,T.shape[1]):
            T[0,k] = k * 0.0**(k-1)
        # only k = 1 is nonzero..
        T[0,1] = 1

    L = A.shape[1]+T.shape[0]
    X = np.zeros((L,L))
    Y = np.zeros((L,1))

    As = A.shape[1]
    X[:As,:As] = A.T @ A
    X[As,:T.shape[1]] = T
    X[:T.shape[1],As] = T.T[:,0]

    Y[:As,0] = A.T @ b[:,0]
    Y[As,0] = 0.0

    csc = np.linalg.solve(X, Y)[:cs.shape[0],:]

    if use_legendre:
        f = np.polynomial.legendre.legval(phi,cs[:,0])
        fc = np.polynomial.legendre.legval(phi,csc[:,0])
    else:
        f = np.zeros_like(theta)
        fc = np.zeros_like(theta)
        for k in range(A.shape[1]):
            f += cs[k] * theta**k
            fc += csc[k] * theta**k

    errs[M-Ms[0],0] = (np.sum((f-r)**2.)/r.shape[0])**0.5
    errs[M-Ms[0],1] = (np.sum((fc-r)**2.)/r.shape[0])**0.5

    plt.figure(2)
    plt.plot(theta,f ,'o-',mfc='none',c=f'C{M-Ms[0]}',label='fL_2={errs[M-Ms[0],0]}')
    plt.plot(theta,fc,'s-',mfc='none',c=f'C{M-Ms[0]}',label='fL_2={errs[M-Ms[0],1]}')

    plt.figure(3)
    plt.plot(theta,f -r,'o-',mfc='none',c=f'C{M-Ms[0]}',label='fL_2={errs[M-Ms[0],0]}')
    plt.plot(theta,fc-r,'s-',mfc='none',c=f'C{M-Ms[0]}',label='fL_2={errs[M-Ms[0],1]}')

plt.figure(2)
plt.title('fit comparison')
plt.plot(theta,r,'b.-',label='shock')
plt.plot(theta,np.ones_like(r),'r.-',label='surface')
# plt.plot(theta,f,'mo-',mfc='none')
# plt.plot(theta,fc,'ms-',mfc='none')
plt.xlim(0,np.pi/2)
plt.ylim(0,3)
plt.xticks([i*np.pi/8 for i in range(5)],
['0',r'$\frac{\pi}{8}$',r'$\frac{1\pi}{4}$',r'$\frac{3\pi}{8}$',r'$\frac{\pi}{2}$'])
plt.xlabel('theta')
plt.ylabel('radius')
plt.legend(loc='best')
plt.show(block=False)

plt.figure(3)
plt.title('fit error comparison')
plt.xlim(0,np.pi/2)
plt.xticks([i*np.pi/8 for i in range(5)],
['0',r'$\frac{\pi}{8}$',r'$\frac{1\pi}{4}$',r'$\frac{3\pi}{8}$',r'$\frac{\pi}{2}$'])
plt.xlabel('theta')
plt.ylabel('radius')
plt.legend(loc='best')
plt.show(block=False)

plt.figure(4)
plt.plot(Ms,errs[:,0],'ro-',mfc='none',label='LSQ')
plt.plot(Ms,errs[:,1],'ms-',mfc='none',label='cLSQ')
plt.xlabel('order, M')
plt.ylabel(r'$L_2$ error')
plt.legend(loc='best')
plt.show(block=False)
