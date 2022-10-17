#! /usr/bin/env python
# nn-sf.py
#  from template: nn-ThetaBetaM.py
#
# input: theta-beta-M training data
# output: trained neural network
#         that approximates beta
#         given theta and M
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

sys.path.append('..')
from bezier import genBezierPoints

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
        if (type(data) == float or type(data) == np.float64):
            data = (data-self.mean)/(self.max-self.min)
            return data
        else:
            data[:] = (data-self.mean)/(self.max-self.min)

    def denormalize(self, data):
        if (type(data) == float or type(data) == np.float64):
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

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class SFModel():
    def __init__(self, training_data_path, epochs, enable_gpu=False):
        self.enable_gpu = enable_gpu
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.inputs = [
                Param(),    # R2
                Param(),    # K1
                Param(),    # K2
                Param(),    # M
                Param(),    # T
                ]
        self.outputs = [Param() for i in range(7)]  # coeffs
        #
        self.read_data(training_data_path)
        self.scaled_data = False
        self.trained = False
        # data structured for storing Tensors
        self.X = []
        self.y = []
        # neural network details
        # 1. instantiate the model
        self.model = Net()
        if (self.enable_gpu):
            self.model.cuda()
        # 2. define the loss function
        self.critereon = MSELoss()
        # 3. define the optimizer
        self.optimizer = SGD(self.model.parameters(), lr=0.2)
        # 4. define the number of epochs
        self.nb_epochs = epochs
        # historic error norm data
        self.iteration = []
        self.error_norm = []

    def read_data(self, filenames):
        # params -> [[R2,K1,K2,M,T],...]
        params = np.loadtxt(filenames[0])
        for i in range(len(self.inputs)):
            self.inputs[i].data = params[:,i]

        # coeffs -> [[a0, a1, ..., a6],...] for a_n * x**n
        coeffs = np.loadtxt(filenames[1])
        for i in range(len(self.outputs)):
            self.outputs[i].data = coeffs[:,i]

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
        if (self.enable_gpu):
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
        if (self.trained == False):
            print("ERROR: a prediction was attempted before model has been trained.")
            exit()
        self.model.eval()
        scaled_inputs = [self.inputs[i].scale.normalize(inputs[i]) for i in range(len(inputs))]
        if (self.enable_gpu):
            self.model.cpu()
        output_preds = self.model(Variable(Tensor(scaled_inputs)))
        scaled_output_preds = [self.outputs[i].scale.denormalize(float(output_preds.data[i])) for i in range(len(self.outputs))]
        return scaled_output_preds

    def plot_error(self):
        if (self.trained == False):
            print("ERROR: error norm plot cannot be generated before the model has been trained.")
            exit()
        plt.plot(self.iteration,self.error_norm)
        plt.grid(True, color='black', linestyle='--', linewidth=0.5)
        plt.yscale("log")
        plt.xlabel("Iterations/Epochs")
        plt.ylabel("Error norm")
        plt.show()

    # def visualise_model(self, mach_max, mach_min, theta_max, theta_min, npts):
        # (mach_numbers, theta_angles) = np.meshgrid(np.linspace(mach_min, mach_max, num=npts),
                                                   # np.linspace(theta_min, theta_max, num=npts))
        # # analytical beta
        # beta_analytic = np.zeros((npts,npts))
        # # model prediction beta
        # beta_model = np.zeros((npts,npts))
        # for i in range(npts):
            # for j in range(npts):
                # beta = analytic.shock_angle_from_ThetaBetaM(mach_numbers[i][j],
                                                        # theta_angles[i][j],
                                                        # gamma)
                # beta_analytic[i][j] = beta
                # beta = self.prediction(theta_angles[i][j],
                                       # mach_numbers[i][j])
                # beta_model[i][j] = beta

        # # 3d model comparison plot
        # fig = plt.figure(num=1, clear=True)
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax.plot_surface(theta_angles, mach_numbers, beta_analytic, cmap=cm.magma,label='analytic')
        # ax.plot_surface(theta_angles, mach_numbers, beta_model, cmap=cm.viridis,label='NN model')
        # #ax.legend()
        # ax.set(xlabel=r'$\theta$', ylabel='M', zlabel=r'$\beta$', title=r'$\theta-\beta-M$')
        # fig.tight_layout()
        # plt.show()

        # # model comparison plot - 2d slice
        # plt.plot(mach_numbers[50], beta_analytic[50], label='analytic')
        # plt.plot(mach_numbers[50], beta_model[50], label='NN model')
        # plt.grid(True, color='black', linestyle='--', linewidth=0.5)
        # plt.legend()
        # title = "Mach vs " + r'$\beta$' + " for " + r'$\theta$' + " = " + str(theta_angles[50][0])
        # plt.title(title)
        # plt.xlabel("Mach")
        # plt.ylabel(r'$\beta$')
        # plt.show()

        # # 3d model error plot
        # beta_error = abs(beta_analytic-beta_model)/beta_analytic * 100
        # fig = plt.figure(num=1, clear=True)
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax.plot_surface(theta_angles, mach_numbers, beta_error, cmap=cm.viridis)
        # ax.set(xlabel=r'$\theta$', ylabel='M', zlabel=r'$\beta$ error',
               # title=r'$\theta-\beta-M$')
        # fig.tight_layout()
        # plt.show()


# ====================================
# MAIN PROGRAM
# ====================================

if __name__ == "__main__":

    #{GPU=1} python3 nn-sf.py ../runs/data/params.dat,../runs/data/coeffs.dat

    # grab training data filename
    if (len(sys.argv) < 2):
        print("NOTE: no training data file provided.")
        exit()
    filenames = sys.argv[1].strip().split(',')
    for filename in filenames:
        if (os.path.isfile(filename) == False):
            print("ERROR: invalid training data file provided.")
            exit()

    # set env variable to run on GPU
    enable_gpu = os.getenv('GPU') != None

    # load the training data
    sf_model = SFModel(filenames, 100000, enable_gpu)

    # scale the training data
    sf_model.scale_data()

    # restructure data for network training
    sf_model.convert_data_to_tensor()

    # training model
    sf_model.train_model()

    # generate error convergence plot
    sf_model.plot_error()

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

    # visualise the model
    # sf_model.visualise_model(1.5, 5.0, 5.0, 45.0, 100)

plt.figure(2)
plt.title('polar domain')
plt.xlim(0,np.pi/2)
plt.xlabel('theta')
plt.ylabel('radius')
plt.xticks([i*np.pi/8 for i in range(5)],
['0',r'$\frac{\pi}{8}$',r'$\frac{1\pi}{4}$',r'$\frac{3\pi}{8}$',r'$\frac{\pi}{2}$'])
plt.legend(loc='best')

plt.figure(3)
plt.title('cartesian domain')
plt.axis('equal')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')

plt.show(block=False)
# plt.show()
plt.show()
