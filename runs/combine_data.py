import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy.io import FortranFile
import os


# loop all the different data files and join into a single directory

res = os.popen(f'mkdir -p data').read()

# paths = ['data_0/','data_1/','data_2/','data_3/','data_4/','data_5/','data_6/','data_7']
paths = []

# now loop the new datasets and cp the files to the next available name
for path in paths:

    ids = {}
    rids = {}
    res = os.popen(f'ls data').read().strip().split('\n')
    for name in res:
        if 'run' in name:
            idx = int(name[3:9])
            if idx in ids.keys():
                print('error: duplicate run numbers in data/')
                exit()
            ids[idx] = name
                
        if 'res' in name:
            ridx = int(name[3:9])
            if ridx in rids.keys():
                print('error: duplicate run numbers in data/')
                exit()
            rids[ridx] = name

    if len(ids) != len(rids):
        print('num run id != num res id  in data/')
        print('lens:  ', len(ids), len(rids))
        exit()

    if len(ids) > 0 and max(ids.keys()) != max(rids.keys()):
        print('next run id != next res id  in data/')
        print('maxes: ', max(ids.keys()), max(rids.keys()))
        exit()

    # now have the next unused {run,res}id
    nidx = 0 if len(ids.keys()) == 0 else max(ids.keys()) + 1

    # need to store an old number -> new number map so that res and run get the same number
    idmap = {}

    res = os.popen(f'ls {path}').read().strip().split('\n')
    # 'res000000.dat', 'run000000.dat', ...
    for name in res:
        if 'run' in name or 'res' in name:
            idx = int(name[3:9])
            if idx not in idmap.keys():
                idmap[idx] = nidx
                nidx += 1
            nn = name[:3] + f'{idmap[idx]:06d}' + name[9:]
            res = os.popen(f'cp {path}/{name} data/{nn}').read()
