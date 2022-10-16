import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

# some controlling parameters
nsims = 3

Mmin = 2.0
Mmax = 8.0

Tmin = 300.0
Tmax = 800.0

# R1 constant, 1.0
# R2, K1, K2 randomnly set

res = os.popen('mkdir -p tmp').read()
res = os.popen('cp template/* tmp/').read()

rng = np.random.default_rng(13)

stat = open('progress.dat','w',buffering=1)

for i in range(nsims):
    now = datetime.datetime.now().time()
    stat.write(f'{i:4d} @ {now}... ')
    # generate random params for Bezier curve representing geometry
    R2 = rng.random() * 0.2 + 0.9
    K1 = rng.random()
    K2 = rng.random() * 0.8
    M  = rng.random() * (Mmax - Mmin) + Mmin
    T  = rng.random() * (Tmax - Tmin) + Tmin
    # record those params in a file for sim
    with open('tmp/input.lua','w') as f:
        f.write(f'R2 = {R2}\n')
        f.write(f'K1 = {K1}\n')
        f.write(f'K2 = {K2}\n')
        f.write(f'M  = {M}\n')
        f.write(f'T  = {T}\n')
        f.close()
    #
    res = os.popen('cd tmp && bash run.sh').read()
    #
    # now want to save shock shape - outside of block 0
    # input dimensions for this block, vertices: (41,6)
    p = np.reshape(np.loadtxt('tmp/plot/cyl-sf-b0000-t0001.vtu',
        skiprows=4,max_rows=246),(41,6,3))[:,0,:-1]
    with open('tmp/input.lua','a') as f:
        np.savetxt(f,p)
    #
    # now move that file for safekeeping
    sn = 'run{:06d}.dat'.format(i)
    names = os.popen('ls data/').read()
    j = 0
    while sn in names and j < 5:
        sn = sn[:-4]+'_.dat'
        j += 1
    if j == 5:
        print('warning: overwriting data...')
    res = os.popen(f'mv tmp/input.lua data/{sn}').read()
    #
    # also record the residual
    res = os.popen(f'mv tmp/config/cyl-sf-residuals.txt data/{sn.replace("run","res")}').read()
    # and delete that folder
    res = os.popen('rm -r tmp/config').read()
    #
    # record time taken
    now = datetime.datetime.now().time()
    stat.write(f'done @ {now}\n')
