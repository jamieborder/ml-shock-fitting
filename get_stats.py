import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy.io import FortranFile
from statistics import mean, median, stdev, variance, pstdev, pvariance

paths = ['data/','data_/','data_1/','data_2/','data_3/','data_4/','data_6/','data_5/']
ns = [i for i in range(100)]


R2s = []
K1s = []
K2s = []
Ms = []
Ts = []

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


print(f'some statistics;')
print(f'    total number of samples: {len(R2s)}')
for name,dat in zip(['R2','K1','K2','M','T'],[R2s,K1s,K2s,Ms,Ts]):
    print(f'    {name}:')
    print(f'        max, min, mean, median : {min(dat):6f}, {max(dat):6f}, {mean(dat):6f}, {median(dat):6f}')
    print(f'        std-deviation, variance: {pstdev(dat):6f}, {pvariance(dat):6f}')
