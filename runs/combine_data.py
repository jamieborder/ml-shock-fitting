import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy.io import FortranFile
import os

# set env variable for input data folders
paths = os.getenv('INPUT').strip().split(',')
if paths is None:
    print('ERROR: no INPUT (env var) folders supplied')
    exit()

# set env variable for output data folder
output_folder = os.getenv('OUTPUT').strip()
if output_folder is None:
    print('ERROR: no OUTPUT (env var) folder supplied')
    exit()


# loop all the different data files and join into a single directory

res = os.popen(f'mkdir -p {output_folder}').read()

# paths = ['data_0/','data_1/','data_2/','data_3/','data_4/','data_5/','data_6/','data_7']
#paths = []

# now loop the new datasets and cp the files to the next available name
for path in paths:

    ids = {}
    rids = {}
    res = os.popen(f'ls {output_folder}').read().strip().split('\n')
    for name in res:
        if 'run' in name:
            idx = int(name[3:9])
            if idx in ids.keys():
                print(f'error: duplicate run numbers in {output_folder}/')
                exit()
            ids[idx] = name

        if 'res' in name:
            ridx = int(name[3:9])
            if ridx in rids.keys():
                print(f'error: duplicate run numbers in {output_folder}/')
                exit()
            rids[ridx] = name

    if len(ids) != len(rids):
        print(f'num run id != num res id  in {output_folder}/')
        print('lens:  ', len(ids), len(rids))
        exit()

    if len(ids) > 0 and max(ids.keys()) != max(rids.keys()):
        print(f'next run id != next res id  in {output_folder}/')
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
            res = os.popen(f'cp {path}/{name} {output_folder}/{nn}').read()
