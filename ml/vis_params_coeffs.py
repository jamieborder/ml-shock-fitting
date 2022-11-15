
import sys
from matplotlib.pyplot import *
import numpy as np
import matplotlib.cm as cm

sys.path.append('..')
from bezier import *

rc(
    'font',
     **{
        'size':16,
        'family':'monospace',
        'monospace':['DejaVu Serif'],
        }
     )


params = np.loadtxt('../runs/data/params.dat')
coeffs = np.loadtxt('../runs/data/coeffs.dat')
#params = np.loadtxt('../runs/data-repeated/params.dat')
#coeffs = np.loadtxt('../runs/data-repeated/coeffs.dat')

print(np.unique(params,axis=0).shape[0], ' / ', params.shape[0], ' are unique')

bcm = cm.turbo(np.linspace(0.1,0.40,len(params)))
rcm = cm.turbo(np.linspace(0.60,0.9,len(params)))
np.random.shuffle(rcm)
np.random.shuffle(bcm)

if False:
    for i in range(params.shape[1]):
        figure(i)
        plot(params[:,i],'o',c=f'C{i}')
        #plot(np.sort(params[:,i]),'o',c=f'C{i}')

    for i in range(coeffs.shape[1]):
        figure(i+10)
        plot(coeffs[:,i],'o',c=f'C{i}')
        #plot(np.sort(params[:,i]),'o',c=f'C{i}')

    show(block=False)

fs = 10,10

if False:
    labels = 'R1,R2,K1,K2,M'.split(',')
    for i in range(params.shape[1]):
        figure(0,figsize=fs)
        plot(params[:,i],'o',c=f'C{i}',ms=10)
        xlabel('random iteration number, -')
        ylabel('labels[i]')
        tight_layout()
        #savefig(f'rand_{labels[i]}.png',dpi=300)
        #clf()
        #close()
        show()


if True:
    ts = np.linspace(0,1,100)

    figure(1,figsize=fs)
    for i in range(len(params)):
        R1,R2,K1,K2,M = params[i,:]
        #R2,K1,K2,M,T = params[i,:]; R1 = 1.0
        p = np.zeros((m+1,2))
        #
        p[0,:] = [-R1, 0]
        p[1,:] = [-R1, K1*R1]
        p[2,:] = [-K2*R2, R2]
        p[3,:] = [ 0, R2]
        xy = genBezierPoints(p,ts.shape[0])
        plot(xy[:,0],xy[:,1],'-',c=bcm[i])
        plot( p[:,0], p[:,1],'o',c=bcm[i])
        #
        #plot([p[0,0],dat[0,0]],[p[0,1],dat[0,1]],'-',c=rcm[i])
        #plot([p[-1,0],dat[-1,0]],[p[-1,1],dat[-1,1]],'-',c=rcm[i])

    show()
