#! /usr/bin/env python
# nn-ThetaBetaM.py
#
# input: theta-beta-M training data
# output: trained neural network
#         that approximates beta
#         given theta and M
#
# author Kyle Damm (May 2021)

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

class ThetaBetaMVariable():
    def __init__(self):
        self.data = np.array([])
        self.scale = Scale()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(2, 10)
        self.fc2 = Linear(10, 10)
        self.fc3 = Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BetaShockModel():
    def __init__(self, training_data_path, epochs, enable_gpu=False):
        self.enable_gpu = enable_gpu
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.theta = ThetaBetaMVariable()
        self.beta  = ThetaBetaMVariable()
        self.M     = ThetaBetaMVariable()
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
        self.optimizer = SGD(self.model.parameters(), lr=0.1)
        # 4. define the number of epochs
        self.nb_epochs = epochs
        # historic error norm data
        self.iteration = []
        self.error_norm = []

    def read_data(self, filename):
        data = np.loadtxt(filename)
        self.theta.data = data[:,0]
        self.beta.data  = data[:,1]
        self.M.data     = data[:,2]

    def scale_data(self):
        self.theta.scale.min_max_mean(self.theta.data)
        self.theta.scale.normalize(self.theta.data)
        self.beta.scale.min_max_mean(self.beta.data)
        self.beta.scale.normalize(self.beta.data)
        self.M.scale.min_max_mean(self.M.data)
        self.M.scale.normalize(self.M.data)
        self.scaled_data = True

    def convert_data_to_tensor(self):
        print('theta:', self.theta.data.shape)
        print('M    :', self.M.data.shape)
        self.X = np.array((self.theta.data,self.M.data)).T
        print('X    :', self.X.shape)
        self.X = Variable(Tensor(self.X))
        self.y = self.beta.data
        print('y    :', self.y.shape)
        self.y = Variable(Tensor(self.y))
        if (self.enable_gpu):
            self.X = self.X.cuda()
            self.y = self.y.cuda()
        # exit()

    def train_model(self):
        for epoch in range(self.nb_epochs):
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
            print("Epoch: {} Loss: {}".format(epoch, epoch_loss))
        self.trained = True

    def prediction(self, theta, M):
        if (self.trained == False):
            print("ERROR: a prediction was attempted before model has been trained.")
            exit()
        self.model.eval()
        scaled_theta = self.theta.scale.normalize(theta)
        scaled_M = self.M.scale.normalize(M)
        if (self.enable_gpu):
            self.model.cpu()
        beta_pred = self.model(Variable(Tensor([scaled_theta, scaled_M])))
        scaled_beta_pred = self.beta.scale.denormalize(float(beta_pred.data[0]))
        print('beta_pred:', beta_pred.shape)
        print('scaled_beta_pred:', scaled_beta_pred.shape)
        exit()
        return scaled_beta_pred

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

    def visualise_model(self, mach_max, mach_min, theta_max, theta_min, npts):
        (mach_numbers, theta_angles) = np.meshgrid(np.linspace(mach_min, mach_max, num=npts),
                                                   np.linspace(theta_min, theta_max, num=npts))
        # analytical beta
        beta_analytic = np.zeros((npts,npts))
        # model prediction beta
        beta_model = np.zeros((npts,npts))
        for i in range(npts):
            for j in range(npts):
                beta = analytic.shock_angle_from_ThetaBetaM(mach_numbers[i][j],
                                                        theta_angles[i][j],
                                                        gamma)
                beta_analytic[i][j] = beta
                beta = self.prediction(theta_angles[i][j],
                                       mach_numbers[i][j])
                beta_model[i][j] = beta

        # 3d model comparison plot
        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(theta_angles, mach_numbers, beta_analytic, cmap=cm.magma,label='analytic')
        ax.plot_surface(theta_angles, mach_numbers, beta_model, cmap=cm.viridis,label='NN model')
        #ax.legend()
        ax.set(xlabel=r'$\theta$', ylabel='M', zlabel=r'$\beta$', title=r'$\theta-\beta-M$')
        fig.tight_layout()
        plt.show()

        # model comparison plot - 2d slice
        plt.plot(mach_numbers[50], beta_analytic[50], label='analytic')
        plt.plot(mach_numbers[50], beta_model[50], label='NN model')
        plt.grid(True, color='black', linestyle='--', linewidth=0.5)
        plt.legend()
        title = "Mach vs " + r'$\beta$' + " for " + r'$\theta$' + " = " + str(theta_angles[50][0])
        plt.title(title)
        plt.xlabel("Mach")
        plt.ylabel(r'$\beta$')
        plt.show()

        # 3d model error plot
        beta_error = abs(beta_analytic-beta_model)/beta_analytic * 100
        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(theta_angles, mach_numbers, beta_error, cmap=cm.viridis)
        ax.set(xlabel=r'$\theta$', ylabel='M', zlabel=r'$\beta$ error',
               title=r'$\theta-\beta-M$')
        fig.tight_layout()
        plt.show()


# ====================================
# MAIN PROGRAM
# ====================================

if __name__ == "__main__":

    # grab training data filename
    if (len(sys.argv) < 2):
        print("NOTE: no training data file provided.")
        exit()
    filename = sys.argv[1]
    if (os.path.isfile(filename) == False):
        print("ERROR: invalid training data file provided.")
        exit()

    # set env variable to run on GPU
    enable_gpu = os.getenv('GPU') != None

    # load the training data
    thetaBetaM = BetaShockModel(filename, 500, enable_gpu)

    # scale the training data
    thetaBetaM.scale_data()

    # restructure data for network training
    thetaBetaM.convert_data_to_tensor()

    # training model
    thetaBetaM.train_model()

    # generate error convergence plot
    thetaBetaM.plot_error()

    # test the model (assume fluid is air)
    import analytic_shock_angle as analytic
    gamma = 1.4
    M = 2.5; theta = 15.7
    beta = analytic.shock_angle_from_ThetaBetaM(M, theta, gamma)
    prediction = thetaBetaM.prediction(theta, M)
    print("Prediction: {}".format(prediction))
    print("Expected: {}".format(beta))
    abs_error = abs(prediction-beta)
    print("Absolute error: {}".format(abs_error))
    rel_error = abs_error/beta
    print("Relative error: {}".format(rel_error))

    # visualise the model
    thetaBetaM.visualise_model(1.5, 5.0, 5.0, 45.0, 100)
