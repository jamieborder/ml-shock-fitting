import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
    print('python3 vis.py {:06d}')
    exit()


# now generate starting shock and circle of radius 1:

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

ts = np.linspace(0,1,100)
ol = np.array([B(t,op) for t in ts])
il = np.array([B(t,ip) for t in ts])


# plotting all data files
ns = [int(i) for i in sys.argv[1].split(',') if i != '']
for i,idx in enumerate(ns):
    # idx = int(sys.argv[-1])

    datf = f'run{idx:06d}.dat'
    resf = f'res{idx:06d}.dat'

    f = open(datf,'r')
    R2 = float(f.readline().strip().replace(' ','').split('=')[-1])
    K1 = float(f.readline().strip().replace(' ','').split('=')[-1])
    K2 = float(f.readline().strip().replace(' ','').split('=')[-1])
    M  = float(f.readline().strip().replace(' ','').split('=')[-1])
    T  = float(f.readline().strip().replace(' ','').split('=')[-1])
    dat = np.loadtxt(f)

    res = np.loadtxt(resf,skiprows=1)

    plt.figure(1)
    plt.plot(dat[:,0],dat[:,1],'-',c=f'C{i}')

    plt.figure(2)
    plt.semilogy(res[:,0],res[:,3],'--',c=f'C{i}',label='mass')
    plt.semilogy(res[:,0],res[:,8],'-' ,c=f'C{i}',label='L2')

    p = np.zeros((m+1,2))
    R1 = 1
    #
    p[0,:] = [-R1, 0]
    p[1,:] = [-R1, K1*R1]
    p[2,:] = [-K2*R2, R2]
    p[3,:] = [ 0, R2]
    xy = np.array([B(t,p) for t in ts])
    plt.figure(1)
    plt.plot(xy[:,0],xy[:,1],'-',c=f'C{i}')
    plt.plot( p[:,0], p[:,1],'o',c=f'C{i}')


plt.figure(1)
plt.plot(il[:,0],il[:,1],'k-',lw=3)
plt.plot(ol[:,0],ol[:,1],'k-',lw=3)
plt.plot(ip[:,0],ip[:,1],'ko')
plt.plot(op[:,0],op[:,1],'ko')
plt.axis('equal')
# plt.show(block=False)
plt.show()
