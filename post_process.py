import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
    print('python3 vis.py {:06d}[,...]')
    # exit()

# loading a number of data files in this folder
# ns = [int(i) for i in sys.argv[1].split(',') if i != '']
ns = [2]
ld = 'data_/'

for i,idx in enumerate(ns):
    # idx = int(sys.argv[-1])

    datf = ld + f'run{idx:06d}.dat'

    f = open(datf,'r')
    R2 = float(f.readline().strip().replace(' ','').split('=')[-1])
    K1 = float(f.readline().strip().replace(' ','').split('=')[-1])
    K2 = float(f.readline().strip().replace(' ','').split('=')[-1])
    M  = float(f.readline().strip().replace(' ','').split('=')[-1])
    T  = float(f.readline().strip().replace(' ','').split('=')[-1])
    dat = np.loadtxt(f)

    # don't this residual is needed
    # resf = f'res{idx:06d}.dat'
    # res = np.loadtxt(resf,skiprows=1)

    # should input to NN be Bezier params or actual points?
    # or poly coeffs?

    # need poly coeffs for bow shock though (or points?)

    # map to polar coords
    #
    # x = r cos(theta)
    # y = r sin(theta)
    #
    # theta = arctan(y/x)
    # r = (x**2 + y**2)**0.5

    theta = -np.arctan(dat[:,1] / dat[:,0])
    r = (dat[:,0]**2 + dat[:,1]**2)**0.5

    phi = 4.0 * theta / np.pi - 1.0

    N = r.shape[0]

    # 1, x, x^2, x^3, ...
    # f(theta=t) = sum_{k=0}^{M} c_k * t**k

    M = 3

    A = np.zeros((N,M+1))
    for k in range(A.shape[1]):
        A[:,k] = theta[:]**k

    b = r.copy()[:,None]

    # without constraint
    cs = np.linalg.solve(A.T @ A, A.T @ b)

    # could add constraint of zero gradient at start?
    # d/dt( f(theta=t=0) ) = sum_{k=1}^{M} k * c_k * x**(k-1) = 0

    T = np.zeros((1,M+1))
    # for k in range(1,T.shape[1]):
        # T[0,k] = k * 0.0**(k-1)
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

    f = np.zeros_like(theta)
    fc = np.zeros_like(theta)
    for k in range(A.shape[1]):
        f += cs[k] * theta**k
        fc += csc[k] * theta**k

    plt.plot(theta,f,'bo')
    plt.plot(theta,r,'rs')
        
    # STUFF IS WRONG! ALSO NEED TO CALCULATE BEZIER AND MAP THEM
    # TO THIS DOMAIN


plt.show(block=False)
