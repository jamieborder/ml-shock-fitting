import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.cm as cm

from bezier import *

if len(sys.argv) < 2:
    print('python3 vis.py {:06d}')

ts = np.linspace(0,1,100)
R1 = 1


# plotting all data files
if len(sys.argv) > 1:
    if ':' in sys.argv[1]:
        sap = sys.argv[1].split(':')
        lb = int(sap[0])
        ub = int(sap[-1])
        ns = [i for i in range(lb,ub)]
    else:
        ns = [int(i) for i in sys.argv[1].split(',') if i != '']
    if len(sys.argv) > 2:
        paths = [i+'/' for i in sys.argv[2].split(',') if i != '']
else:
    ns = []
    ns = [i for i in range(100)]
    paths = ['data/','data_/','data_1/','data_2/','data_3/','data_4/','data_6/']

bcm = cm.coolwarm(np.linspace(0,0.45,len(ns)))
rcm = cm.coolwarm(np.linspace(0.55,1,len(ns)))

for path in paths:
    # path = 'data/'
    for i,idx in enumerate(ns):

        # idx = int(sys.argv[-1])

        datf = path + f'run{idx:06d}.dat'
        resf = path + f'res{idx:06d}.dat'

        f = open(datf,'r')
        R2 = float(f.readline().strip().replace(' ','').split('=')[-1])
        K1 = float(f.readline().strip().replace(' ','').split('=')[-1])
        K2 = float(f.readline().strip().replace(' ','').split('=')[-1])
        M  = float(f.readline().strip().replace(' ','').split('=')[-1])
        T  = float(f.readline().strip().replace(' ','').split('=')[-1])
        dat = np.loadtxt(f)

        res = np.loadtxt(resf,skiprows=1)

        plt.figure(1)
        # plt.plot(dat[:,0],dat[:,1],'-',c=f'C{i}')
        plt.plot(dat[:,0],dat[:,1],'-',c=rcm[i])

        plt.figure(2)
        plt.semilogy(res[:,0],res[:,3],'--',c=f'C{i}',label='mass')
        plt.semilogy(res[:,0],res[:,8],'-' ,c=f'C{i}',label='L2')

        p = np.zeros((m+1,2))
        #
        p[0,:] = [-R1, 0]
        p[1,:] = [-R1, K1*R1]
        p[2,:] = [-K2*R2, R2]
        p[3,:] = [ 0, R2]
        xy = genBezierPoints(p,ts.shape[0])
        plt.figure(1)
        # plt.plot(xy[:,0],xy[:,1],'-',c=f'C{i}')
        # plt.plot( p[:,0], p[:,1],'o',c=f'C{i}')
        plt.plot(xy[:,0],xy[:,1],'-',c=bcm[i])
        plt.plot( p[:,0], p[:,1],'o',c=bcm[i])


# generate starting shock and circle of radius 1:
op = np.zeros((m+1,2))
op[0,:] = [-2*R1, 0]
op[1,:] = [-2*R1, R1]
op[2,:] = [-R1, 2.5*R1]
op[3,:] = [0, 3*R1]
ip = np.zeros((m+1,2))
K = 0.5522847498
ip[0,:] = [-R1, 0]
ip[1,:] = [-R1, K*R1]
ip[2,:] = [-K*R1, R1]
ip[3,:] = [ 0, R1]

ol = genBezierPoints(op,ts.shape[0])
il = genBezierPoints(ip,ts.shape[0])


plt.figure(1)
plt.plot(il[:,0],il[:,1],'k-',lw=3)
plt.plot(ol[:,0],ol[:,1],'k-',lw=3)
plt.plot(ip[:,0],ip[:,1],'ko')
plt.plot(op[:,0],op[:,1],'ko')
plt.axis('equal')
# plt.show(block=False)
plt.show()
