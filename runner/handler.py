import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

### nohup python3 handler.py &
### disown #1

# some controlling parameters
nsims = 100

Mmin = 2.0
Mmax = 8.0

# R1, R2, K1, K2 randomnly set

var = '_20'
# var = ''

tmp = 'tmp' + var
data = 'data' + var
prog = 'progress' + var + '.dat'

res = os.popen(f'mkdir -p {tmp}').read()
res = os.popen(f'mkdir -p {data}').read()
res = os.popen(f'cp template/* {tmp}/').read()

rng = np.random.default_rng(13)

stat = open(f'{prog}','a',buffering=1)

# get last number in directory
res = os.popen(f'ls {data}').read().split('\n')
idx = 0
for i,fn in enumerate(res):
    if 'run' in fn:
        idx = max(idx,int(fn[3:9])+1)

for i in range(nsims):
    now = datetime.datetime.now().time()
    stat.write(f'{i:4d} @ {now}...\n')
    #
    # generate random params for Bezier curve representing geometry
    #
    # R2 = rng.random() * 0.2 + 0.9
    # K1 = rng.random()
    # K2 = rng.random() * 0.8
    #
    # R2 = rng.random() * 0.3 + 0.9
    # K1 = rng.random() * 1.1
    # K2 = rng.random()
    #
    # R2 = rng.random() * 0.3 + 0.9
    # K1 = rng.random() * 1.1
    # K2 = rng.random() * 0.2
    #
    # T  = rng.random() * (Tmax - Tmin) + Tmin
    #
    # R1 = rng.random() * 0.4 + 0.8
    # R2 = rng.random() * 0.6 + 0.6
    # K1 = rng.random() * 0.3
    # K2 = rng.random() * 0.3
    #
    # R1 = rng.random() * 0.8 + 0.8
    # R2 = rng.random() * 0.6 + 0.6
    # K1 = rng.random() * 0.3
    # K2 = rng.random() * 0.3
    #
    R1 = rng.random() * 0.8 + 0.8
    R2 = rng.random() * 0.6 + 0.6
    K1 = rng.random() * 0.3 + 0.2
    K2 = rng.random()
    #
    M  = rng.random() * (Mmax - Mmin) + Mmin
    # record those params in a file for sim
    with open(f'{tmp}/input.lua','w') as f:
        f.write(f'R1 = {R1}\n')
        f.write(f'R2 = {R2}\n')
        f.write(f'K1 = {K1}\n')
        f.write(f'K2 = {K2}\n')
        f.write(f'M  = {M}\n')
        f.close()
    #
    res = os.popen(f'cd {tmp} && bash run.sh').read()
    #
    # now want to save shock shape - outside of block 0
    # input dimensions for this block, vertices: (41,6)
    p = np.reshape(np.loadtxt(f'{tmp}/plot/cyl-sf-b0000-t0001.vtu',
        skiprows=4,max_rows=246),(41,6,3))[:,0,:-1]
    with open(f'{tmp}/input.lua','a') as f:
        np.savetxt(f,p)
    #
    # now move that file for safekeeping
    sn = 'run{:06d}.dat'.format(idx)
    idx += 1
    names = os.popen(f'ls {data}/').read()
    j = 0
    while sn in names and j < 5:
        sn = sn[:-4]+'_.dat'
        j += 1
    if j == 5:
        print('warning: overwriting data...')
    res = os.popen(f'mv {tmp}/input.lua {data}/{sn}').read()
    #
    # also record the residual
    res = os.popen(f'mv {tmp}/config/cyl-sf-residuals.txt {data}/{sn.replace("run","res")}').read()
    # and delete that folder
    res = os.popen(f'rm -r {tmp}/config').read()
    #
    # record time taken
    now = datetime.datetime.now().time()
    stat.write(f'     @ {now} finised\n')
