import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 16

# generating surface shapes and converting from polar to cartesian

# for dev
from matplotlib.pyplot import *

# -1.444847941398620605e+00 0.000000000000000000e+00 0.000000000000000000e+00
pos = np.loadtxt('pos.dat')

# input dimensions:
# for this block, vertices: (41,6)
p = np.reshape(pos,(41,6,3))

# map to polar coords
#
# x = r cos(theta)
# y = r sin(theta)
#
# theta = arctan(y/x)
# r = (x**2 + y**2)**0.5

ol = p[:,0,:]
theta = -np.arctan(ol[:,1] / ol[:,0])
r = (ol[:,0]**2 + ol[:,1]**2)**0.5

phi = 4.0 * theta / np.pi - 1.0

M = 4

mon = np.zeros((M+1,theta.shape[0]))
for k in range(M+1):
    mon[k,:] = theta**k

N = 100

ctheta = np.cos(theta)
stheta = np.sin(theta)

for n in range(N):
    cs = np.random.rand(M+1) * 0.1
    cs[0] = 1.0
    # cs[1] = 0.0
    rs = np.zeros(theta.shape[0])
    for k in range(cs.shape[0]):
        rs += cs[k] * mon[k,:]

    # and in cartesian
    x_ = rs * ctheta
    y_ = rs * stheta

    plt.figure(1)
    plt.plot(-x_,y_,'-',c=f'C{n%10}')

    plt.figure(2)
    plt.plot(theta,rs,'-',c=f'C{n%10}')


plt.figure(1)
plt.axis('equal')

plt.figure(2)
plt.title('fit comparison')
plt.plot(theta,np.ones_like(r),'r.-',label='surface')
plt.plot(theta,r,'b.-',label='shock')
plt.xlim(0,np.pi/2)
plt.ylim(0,3)
plt.xticks([i*np.pi/8 for i in range(5)],
['0',r'$\frac{\pi}{8}$',r'$\frac{1\pi}{4}$',r'$\frac{3\pi}{8}$',r'$\frac{\pi}{2}$'])
plt.xlabel('theta')
plt.ylabel('radius')
plt.legend(loc='best')
plt.show(block=False)

# d = Vector3:new{x=-1.5*radius, y=0}
# e = Vector3:new{x=-1.5*radius, y=radius}
# f = Vector3:new{x=-radius, y=2.0*radius}
# g = Vector3:new{x=0.0, y=3.0*radius}

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

Path = mpath.Path

fig, ax = plt.subplots()
r = 1
ps = [(-1.5*r, 0), (-1.5*r, r), (-r, 2*r), (0, 3*r)]
pp1 = mpatches.PathPatch(
    Path(
        ps,
         [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
    fc="none", transform=ax.transData)
pp2 = mpatches.PathPatch(
    Path.arc(0,90),
    fc="none", transform=ax.transData)

ax.add_patch(pp1)
ax.add_patch(pp2)
pa = [a[0] for a in ps]
pb = [a[1] for a in ps]
ax.plot([0], [0], "ro")
ax.plot(pa,pb,'bs')

plt.show()
