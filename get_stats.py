import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy.io import FortranFile
from statistics import mean, median, stdev, variance, pstdev, pvariance
import os
import sys

PROCESSED = os.getenv('PROCESSED')
PLOT = os.getenv('PLOT')

if len(sys.argv) < 2:
    print('[PROCESSED=1] python3 get_stats.py [i [path]]')
    print('[PROCESSED=1] python3 get_stats.py [i,j,... [path]]')
    print('[PROCESSED=1] python3 get_stats.py [i:k [path]]')

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
    # paths = ['data/','data_/','data_1/','data_2/','data_3/','data_4/','data_6/','data_5/']
    # ns = [i for i in range(100)]
    paths = ['data_/']
    ns = [i for i in range(57)]


R2s = []
K1s = []
K2s = []
Ms = []
Ts = []

if not PROCESSED:
    for path in paths:
        for i,idx in enumerate(ns):
            datf = path + f'run{idx:06d}.dat'

            f = open(datf,'r')
            R2 = float(f.readline().strip().replace(' ','').split('=')[-1])
            K1 = float(f.readline().strip().replace(' ','').split('=')[-1])
            K2 = float(f.readline().strip().replace(' ','').split('=')[-1])
            M  = float(f.readline().strip().replace(' ','').split('=')[-1])
            T  = float(f.readline().strip().replace(' ','').split('=')[-1])

            R2s.append(R2)
            K1s.append(K1)
            K2s.append(K2)
            Ms.append(M)
            Ts.append(T)

else:
    R2s = None; K1s = None; K2s = None; Ms  = None; Ts  = None
    for path in paths:
        datf = path + f'params.dat'
        dat = np.loadtxt(datf)
        if R2s is None:
            R2s = dat[:,0]
            K1s = dat[:,1]
            K2s = dat[:,2]
            Ms  = dat[:,3]
            Ts  = dat[:,4]
        else:
            R2s = np.hstack((R2s,dat[:,0]))
            K1s = np.hstack((K1s,dat[:,1]))
            K2s = np.hstack((K2s,dat[:,2]))
            Ms  = np.hstack((Ms ,dat[:,3]))
            Ts  = np.hstack((Ts ,dat[:,4]))

print(f'some statistics;')
print(f'    total number of samples: {len(R2s)}')
for name,dat in zip(['R2','K1','K2','M','T'],[R2s,K1s,K2s,Ms,Ts]):
    print(f'    {name}:')
    print(f'        max, min, mean, median : {min(dat):6f}, {max(dat):6f}, {mean(dat):6f}, {median(dat):6f}')
    print(f'        std-deviation, variance: {pstdev(dat):6f}, {pvariance(dat):6f}')

if PLOT:
    plt.figure(1)
    plt.plot(R2s,K1s,'bo')
    plt.plot(R2s,K2s,'rs')
    # plt.plot(K1s,K2s,'rs')
    # plt.plot(R2s,Ms,'g*')
    # plt.plot(R2s,Ts,'m*')
    plt.show()
