#! /usr/bin/env python
# nn_sf.py
#  from template: nn-ThetaBetaM.py
#
# input : bow shock location from shock-fitting CFD simulations
# output: trained neural network
#         that approximates location of bow shock
#         given M and body geometry
#
# author Kyle Damm (May 2021)
# modified Jamie Border (Oct 2022)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
# PyTorch https://pytorch.org/
import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable

import re

sys.path.append('..')
from bezier import genBezierPoints
from nn_diagram_template import *

plt.rcParams['font.size'] = 16

class Scale():
    def __init__(self):
        self.max = 1.0
        self.min = 0.0
        self.mean = 0.0

    def min_max_mean(self, data):
        self.max = np.max(data)
        self.min = np.min(data)
        self.mean = np.mean(data);

    def normalize(self, data):
        if type(data) == float or type(data) == np.float64:
            data = (data-self.mean)/(self.max-self.min)
            return data
        else:
            data[:] = (data-self.mean)/(self.max-self.min)

    def denormalize(self, data):
        if type(data) == float or type(data) == np.float64:
            data = data*(self.max-self.min)+self.mean
            return data
        else:
            data[:] = data*(self.max-self.min)+self.mean

class Param():
    def __init__(self):
        self.data = np.array([])
        self.scale = Scale()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(5, 20)
        self.fc2 = Linear(20, 20)
        self.fc3 = Linear(20, 20)
        self.fc4 = Linear(20, 7)
        #self.fc1 = Linear(5, 10)
        #self.fc2 = Linear(10, 10)
        #self.fc3 = Linear(10, 10)
        #self.fc4 = Linear(10, 7)
        #self.fc1 = Linear(5, 10)
        #self.fc2 = Linear(10, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        return x

    def gen_diagram(self,filename):
        nn_str = self.__repr__().split('\n')
        sizes = []
        prev_size = -1
        for line in nn_str:
            if 'Linear' in line:
                m = re.search('in_features=([0-9]+)',line)
                if m is None:
                    print("failed to match `in_features' in <", line, ">")
                    return
                in_size = int(m.groups()[0])
                if prev_size > 0 and prev_size != in_size:
                    print(f"size mismatch: prev_size = {prev_size}, in_size = {in_size}")
                    return
                sizes.append(in_size)
                m = re.search('out_features=([0-9]+)',line)
                if m is None:
                    print("failed to match `out_features' in <", line, ">")
                    return
                prev_size = int(m.groups()[0])
        sizes.append(prev_size)
        #
        #
        first_nodes = ['x_'+str(i) for i in range(sizes[0])]
        first_nodes_str = ''
        for node in first_nodes:
            first_nodes_str += node + ' '
        first_nodes_str = first_nodes_str[:-1]
        sub_nodes = [['a_'+str(i)+'_'+str(j) for i in range(sizes[j])] for j in range(1,len(sizes))]
        sub_nodes_strs = []
        for sg in sub_nodes:
            sub_node_str = ''
            for node in sg:
                sub_node_str += node + ' '
            sub_nodes_strs.append(sub_node_str[:-1])
        #
        #
        subgs = ''
        for i in range(len(sizes)-1):
            subgs += template_sg.format(
                ID=i+1,
                COLOUR='red2' if i != len(sizes)-2 else 'seagreen2',
                NODES=sub_nodes_strs[i],
                LABEL='layer '+str(i+1) if i != len(sizes)-2 else 'outputs'
                )
        #
        cons = ''
        all_nodes = sub_nodes.copy()
        all_nodes.insert(0,first_nodes)
        #
        for i in range(len(sizes)-1):
            for j in range(sizes[i]):
                for k in range(sizes[i+1]):
                    cons += all_nodes[i][j] + ' -> ' + all_nodes[i+1][k] + '[arrowsize=0.5];\n'
            cons += '\n'
        #
        ds = template_nn.format(
                FIRST_NODES=first_nodes_str,
                SUBGRAPHS=subgs,
                CONNECTIONS=cons
                )
        #
        with open(filename,'w') as f:
            f.write(ds)
            f.close()
        os.popen(f'dot -Tpng -O {filename}')
        print(f'generated graph in {filename}.png')

class SFModel():
    def __init__(self, training_data_filenames=None, epochs=0, enable_gpu=False, train=True):
        self.enable_gpu = enable_gpu
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.inputs = [
                Param(),    # R1
                Param(),    # R2
                Param(),    # K1
                Param(),    # K2
                Param(),    # M
                ]
        self.outputs = [Param() for i in range(7)]  # coeffs

        if train:
            self.read_data(training_data_filenames)

        self.scaled_data = False
        self.trained = False

        # data structured for storing Tensors
        self.X = []
        self.y = []

        # neural network details
        # 1. instantiate the model
        self.model = Net()
        if self.enable_gpu:
            self.model.cuda()
        # 2. define the loss function
        self.critereon = MSELoss()
        # 3. define the optimizer
        self.optimizer = SGD(self.model.parameters(), lr=0.1)
        # 4. define the number of epochs
        self.nb_epochs = epochs
        # historic error norm data
        self.iteration = []
        self.error_norm = []
        self.weights_loaded = False

    def save_model(self,filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self,filename,on_gpu=False):
        if on_gpu:
            self.model.load_state_dict(torch.load(filename))
        else:
            self.model.load_state_dict(torch.load(filename,map_location=torch.device('cpu')))
        self.model.eval()
        self.weights_loaded = True

    def save_mmm(self,filename):
        data = np.zeros((len(self.inputs)+len(self.outputs),3))
        for i in range(len(self.inputs)):
            data[i,0] = self.inputs[i].scale.max
            data[i,1] = self.inputs[i].scale.min
            data[i,2] = self.inputs[i].scale.mean
        for i in range(len(self.outputs)):
            data[len(self.inputs)+i,0] = self.outputs[i].scale.max
            data[len(self.inputs)+i,1] = self.outputs[i].scale.min
            data[len(self.inputs)+i,2] = self.outputs[i].scale.mean
        np.savetxt(filename,data)

    def load_mmm(self,filename):
        data = np.loadtxt(filename,comments=['#'])
        for i in range(len(self.inputs)):
            self.inputs[i].scale.max  = data[i,0]
            self.inputs[i].scale.min  = data[i,1]
            self.inputs[i].scale.mean = data[i,2]
        for i in range(len(self.outputs)):
            self.outputs[i].scale.max  = data[len(self.inputs)+i,0]
            self.outputs[i].scale.min  = data[len(self.inputs)+i,1]
            self.outputs[i].scale.mean = data[len(self.inputs)+i,2]
        self.scaled_data = True

    def read_data(self, training_data_filenames):
        if training_data_filenames is None:
            print('ERROR: no valid training data filenames provided:', training_data_filenames)
            exit()
        param_fns,coeff_fns = training_data_filenames[0],training_data_filenames[1]

        if len(param_fns) == 1:
            # params -> [[R1,R2,K1,K2,M],...]
            params = np.loadtxt(param_fns[0])
            print(params.shape)
            for i in range(len(self.inputs)):
                self.inputs[i].data = params[:,i]

            # coeffs -> [[a0, a1, ..., a6],...] for a_n * x**n
            coeffs = np.loadtxt(coeff_fns[0])
            print(coeffs.shape)
            for i in range(len(self.outputs)):
                self.outputs[i].data = coeffs[:,i]
        else:
            print('WARNING: known issues here...')

            # need to know number of columns in params,coeffs already, to pre-allocate these arrays
            for i in range(len(self.inputs)):
                self.inputs[i].data = np.empty((0,5))
            for i in range(len(self.outputs)):
                self.outputs[i] = np.empty((0,7))

            for j in range(len(param_fns)):
                # params -> [[R2,K1,K2,M,T],...]
                params = np.loadtxt(param_fns[j])
                for i in range(len(self.inputs)):
                    self.inputs[i].data = np.vstack((self.inputs[i].data, params[:,i]))

                # coeffs -> [[a0, a1, ..., a6],...] for a_n * x**n
                coeffs = np.loadtxt(coeff_fns[j])
                for i in range(len(self.outputs)):
                    self.outputs[i].data = np.vstack((self.outputs[i].data, coeffs[:,i]))

    def scale_data(self):
        for var in self.inputs:
            var.scale.min_max_mean(var.data)
            var.scale.normalize(var.data)
        self.scaled_data = True

    def convert_data_to_tensor(self):
        self.X = np.array([var.data for var in self.inputs]).T
        self.X = Variable(Tensor(self.X))
        self.y = np.array([var.data for var in self.outputs]).T
        self.y = Variable(Tensor(self.y))
        if self.enable_gpu:
            self.X = self.X.cuda()
            self.y = self.y.cuda()

    def train_model(self):
        for i,epoch in enumerate(range(self.nb_epochs)):
            epoch_loss = 0
            y_pred = self.model(self.X)
            y_pred = torch.squeeze(y_pred)
            loss = self.critereon(y_pred, self.y)
            epoch_loss = loss.data
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iteration.append(epoch)
            self.error_norm.append(epoch_loss.cpu())
            if i % 100 == 0:
                print("Epoch: {} Loss: {}".format(epoch, epoch_loss))
        self.trained = True

    def prediction(self, inputs):
        if not self.trained and not self.weights_loaded:
            print("ERROR: a prediction was attempted before model has been trained.")
            exit()
        self.model.eval()
        scaled_inputs = [self.inputs[i].scale.normalize(inputs[i]) for i in range(len(inputs))]
        if self.enable_gpu:
            self.model.cpu()
        output_preds = self.model(Variable(Tensor(scaled_inputs)))
        scaled_output_preds = [self.outputs[i].scale.denormalize(float(output_preds.data[i])) for i in range(len(self.outputs))]
        return scaled_output_preds

    def plot_error(self):
        if not self.trained:
            print("ERROR: error norm plot cannot be generated before the model has been trained.")
            exit()
        plt.plot(self.iteration,self.error_norm)
        plt.grid(True, color='black', linestyle='--', linewidth=0.5)
        plt.yscale("log")
        plt.xlabel("Iterations/Epochs")
        plt.ylabel("Error norm")
        plt.show()


if __name__ == "__main__":

    #{GPU=1} {VIS=1} python3 nn_sf.py ../runs/data/params.dat, ../runs/data/coeffs.dat,

    # set env variable to run on GPU
    enable_gpu = os.getenv('GPU') != None

    # check if just wanting to vis
    vis_only = os.getenv('VIS') != None

    # grab training data filename
    if len(sys.argv) < 3:
        if not vis_only:
            print("NOTE: no training data files provided.")
            exit()
        else:
            filenames = ['','']
    else:
        params = sys.argv[1].strip().split(',')
        coeffs = sys.argv[2].strip().split(',')
        for filename in params:
            if (os.path.isfile(filename) == False):
                print("ERROR: invalid 'params.dat' training data file provided:", filename)
                exit()
        for filename in coeffs:
            if (os.path.isfile(filename) == False):
                print("ERROR: invalid 'coeffs.dat' training data file provided:", filename)
                exit()
        if len(params) != len(coeffs):
            print("ERROR: number of params and coeffs files supplied are different:",
                    len(params), '!=', len(coeffs))
            exit()

    # create the model and load the training data if training
    train = not vis_only
    sf_model = SFModel([params,coeffs], 100000, enable_gpu, train=train)

    if vis_only:
        sf_model.model.gen_diagram('nn_diagram')
        exit()

    # scale the training data
    sf_model.scale_data()

    # restructure data for network training
    sf_model.convert_data_to_tensor()


    # training model
    sf_model.train_model()

    # generate error convergence plot
    sf_model.plot_error()

    # save the model
    res = os.popen(f'ls ./models/').read().strip().split('\n')
    mids = []
    for name in res:
        if 'model' in name:
            mids.append(int(name[5:7]))
    mid = 0
    while mid in mids:
        mid += 1
    sf_model.save_model(f'./models/model{mid:02d}.pth')
    sf_model.save_mmm(f'./models/mmm{mid:02d}.dat')

    # exit()

    # test the model
    # input params: R2,K1,K2,M,T

    # this case doesn't fit that well
    lparams = '1.024010845383956037e+00 5.559842733496153100e-01 2.493385792650427979e-01 7.985859177691415844e+00 6.662489193564000516e+02'
    lcoeffs = '1.329053681099242601e+00 -2.381378425592319166e-03 1.815951977252002747e-01 -7.257403682239993437e-02 2.686813008048670737e-01 -1.739286381676704019e-01 5.769665095911536562e-02'

    # this one fits quite well
    # lparams = '1.031934264169858340e+00 4.557829600792653313e-01 5.861211734777412863e-01 4.867340185411659803e+00 3.623606666529556151e+02'
    # lcoeffs = '1.493551514491779475e+00 -4.820248358711151995e-03 3.362109836934503715e-01 -1.457855463284446484e-01 3.143132348758396755e-01 -1.968522945545726766e-01 6.791856727176236175e-02'

    inputs = [float(var) for var in lparams.split(' ')]
    prediction = sf_model.prediction(inputs)
    coeffs = [float(var) for var in lcoeffs.split(' ')]
    print("Prediction: {},{},{},{},{},{},{}".format(*prediction))
    print("Prediction: {},{},{},{},{},{},{}".format(*coeffs))

    # theta distribution for plotting
    thetas = np.linspace(0,1,100) * np.pi/2.0

    # radial coord for prediction (ps) and expected (es)
    ps = np.zeros_like(thetas)
    es = np.zeros_like(thetas)
    for i in range(len(prediction)):
        ps += prediction[i] * thetas**i
    for i in range(len(coeffs)):
        es += coeffs[i] * thetas**i

    tcm = cm.turbo(np.linspace(0.2,0.6,5))

    # and plotting them
    plt.figure(2)
    plt.plot(thetas,ps,'o-',label='prediction',c=tcm[0],mfc='none')
    plt.plot(thetas,es,'.-',label='simulation',c=tcm[2])

    # also map from polar to cartesian to visualise proper shock shape
    # x = r cos(theta)
    # y = r sin(theta)
    xps = -ps * np.cos(thetas)
    yps =  ps * np.sin(thetas)
    xes = -es * np.cos(thetas)
    yes =  es * np.sin(thetas)

    # and plotting them
    plt.figure(3)
    plt.plot(xps,yps,'o-',label='prediction',c=tcm[0],mfc='none')
    plt.plot(xes,yes,'.-',label='simulation',c=tcm[2])

    # it is a bit awkard but we can create a Bezier curve of the surface
    #  from the input params in cartesian space, and then map them to polar


    p = np.zeros((4,2))
    # input params: R2,K1,K2,M,T
    R1 = 1.0
    R2,K1,K2,M,T = inputs
    p[0,:] = [-R1, 0]
    p[1,:] = [-R1, K1*R1]
    p[2,:] = [-K2*R2, R2]
    p[3,:] = [ 0, R2]
    ts = np.linspace(0,1,thetas.shape[0])
    xy = genBezierPoints(p,ts.shape[0])

    # plot in cartesian space
    plt.figure(3)
    plt.plot(xy[:,0],xy[:,1],'k.-',label='surface')
    plt.plot([xy[0,0],xes[0]],[xy[0,1],yes[0]],'-',c=tcm[2],zorder=0)
    plt.plot([xy[-1,0],xes[-1]],[xy[-1,1],yes[-1]],'-',c=tcm[2],zorder=0)

    # and map to polar
    #   theta = arctan(y/x)
    #   r = (x**2 + y**2)**0.5
    stheta = -np.arctan(xy[:,1] / (xy[:,0] - 1e-16))
    sr = (xy[:,0]**2 + xy[:,1]**2)**0.5

    # and plot
    plt.figure(2)
    plt.plot(stheta,sr,'k.-',label='surface')


    abs_error = np.sum(np.abs(ps - es))
    print("Absolute error: {}".format(abs_error))
    l2_error = (np.sum((ps - es)**2) / ps.shape[0])**0.5
    print("L2 error: {}".format(l2_error))

    plt.figure(2,figsize=(8,8))
    plt.title('polar domain')
    plt.xlim(0,np.pi/2)
    plt.xlabel('theta')
    plt.ylabel('radius')
    plt.xticks([i*np.pi/8 for i in range(5)],
    ['0',r'$\frac{\pi}{8}$',r'$\frac{1\pi}{4}$',r'$\frac{3\pi}{8}$',r'$\frac{\pi}{2}$'])
    plt.legend(loc='best')

    plt.figure(3,figsize=(8,8))
    plt.title('cartesian domain')
    plt.axis('equal')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show(block=False)
    # plt.show()
    plt.show()
