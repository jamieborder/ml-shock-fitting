import matplotlib.pyplot as plt
import numpy as np
import os

ld = 'data_all/'

# plot_them = True
plot_them = False

params = np.empty([0,5])
shapes = np.empty([0,41*2])
coeffs = np.empty([0,7])

# loading all of them
res = os.popen(f'ls {ld}').read().strip().split('\n')
i = -1
for name in res:
    if 'run' in name:
        datf = ld + name
        i += 1
    else:
        continue

    f = open(datf,'r')

    # handle two different formats
    first_line = f.readline().strip().replace(' ','').split('=')
    old_format = True
    if first_line[0] == 'R1':
        old_format = False
    elif first_line[0] != 'R2':
        print('ERROR: in post-processing expected first line to start with `R1` or `R2`')
        exit()

    if not old_format:
        R1 = float(first_line[-1])
        R2 = float(f.readline().strip().replace(' ','').split('=')[-1])
        K1 = float(f.readline().strip().replace(' ','').split('=')[-1])
        K2 = float(f.readline().strip().replace(' ','').split('=')[-1])
        M  = float(f.readline().strip().replace(' ','').split('=')[-1])
        T  = 300.0
    else:
        R1 = 1.0
        R2 = float(first_line[-1])
        K1 = float(f.readline().strip().replace(' ','').split('=')[-1])
        K2 = float(f.readline().strip().replace(' ','').split('=')[-1])
        M  = float(f.readline().strip().replace(' ','').split('=')[-1])
        T  = float(f.readline().strip().replace(' ','').split('=')[-1])

    dat = np.loadtxt(f)

    # params = np.vstack((params,np.array([[R2,K1,K2,M,T]])))
    # params = np.vstack((params,np.array([[R1,R2,K1,K2,M,T]])))
    params = np.vstack((params,np.array([[R1,R2,K1,K2,M]])))

    # shapes = np.vstack((shapes,dat.flatten()[None,:]))
    shapes = np.vstack((shapes,dat.T.flatten()[None,:]))

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

    theta = -np.arctan(dat[:,1] / (dat[:,0]-1e-16))
    r = (dat[:,0]**2 + dat[:,1]**2)**0.5

    N = r.shape[0]

    # 1, x, x^2, x^3, ...
    # f(theta=t) = sum_{k=0}^{M} c_k * t**k

    M = 6

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

    if plot_them:
        plt.plot(theta,f ,'bo' ,ms=12)
        plt.plot(theta,fc,'rs' ,ms=12)
        plt.plot(theta,r ,'k.-',ms=12)
        print(np.sum(np.abs(f-r)),np.sum(np.abs(fc-r)))

    coeffs = np.vstack((coeffs,cs.T))

if plot_them:
    plt.show(block=False)

# save data to file
if True:
    np.savetxt('data/params.dat',params)
    np.savetxt('data/shapes.dat',shapes)
    np.savetxt('data/coeffs.dat',coeffs)
