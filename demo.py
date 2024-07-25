#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import torch
import h5py
import functools
import math
import os
import sys
import math
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import io
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import parameters
import utils
from models.hdsn import Step1_SSN as step1_ssn
from models.hdsn import Step2_phasor as step2_phasor
from models.aan import AAN as step3_isn
from utils import process_signal
import time


print('current time:', datetime.now())

## Specify the gpu for testing
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
n_gpu = torch.cuda.device_count()
print('Using %s GPUs for testing......' % (n_gpu))
sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('The device is:', device)

## Define the network
net1 = step1_ssn()
algorithm = step2_phasor()
net2 = step3_isn()


## Load data
sample_size_test = parameters.sample_size_test
batchsize = parameters.batchsize
lr = parameters.learning_rate
sample_id = parameters.sample_id

## Load weights
net1 = net1.to(device)
net1 = torch.nn.DataParallel(net1)
net1.load_state_dict(torch.load(parameters.net1_weight_path))

algorithm.todev(device)

net2 = torch.nn.DataParallel(net2)
net2.load_state_dict(torch.load(parameters.net2_weight_path))
net2 = net2.to(device)

print('Testing on dataset:', parameters.dataset_name)
if parameters.dataset_name == 'Experiment18m':
    X0 = np.empty(
        [sample_size_test, 1, parameters.time_span, parameters.resolution_down_hor, parameters.resolution_down_ver])
    Y2 = np.zeros([sample_size_test, 1, parameters.time_span, parameters.sig_super_scale * parameters.resolution_down_hor,
                   parameters.sig_super_scale * parameters.resolution_down_ver])
    for group in range(sample_size_test):
        Sig = io.loadmat('./dataset_18m/%d.mat' % (group + 1))['sig']
        x0, y2 = process_signal(Sig, [32, 32, 512], 4)
        X0[group,:,:,:,:] = x0.reshape([1, 1, 512, 8, 8])
        Y2[group,:,:,:,:] = y2.reshape([1, 1, 512, 32, 32])


## Make folder to store results
if not os.path.exists(parameters.Output_path):
    os.mkdir(parameters.Output_path)

img_folder = './image-%s/' % (parameters.dataset_name)
if not os.path.exists(img_folder):
    os.mkdir(img_folder)

start = time.time()
for count in range(sample_size_test):
    print('sample:', count)
    Xt0 = X0[count, :, :, :, :] / np.max(X0[count, :, :, :, :])

    with torch.no_grad():
        Xt0 = Variable(torch.from_numpy(Xt0)).reshape(1, 1, parameters.time_span, parameters.resolution_down_ver,
                                                      parameters.resolution_down_hor)
        Xt0 = Xt0.to(device)
        Xt0 = Xt0.type(torch.cuda.FloatTensor)

        Y_hat2_1 = net1(Xt0)
        Y_hat2_2 = algorithm(Y_hat2_1)
        Y_hat2_3 = net2(Y_hat2_2)

        Y1 = Y_hat2_1.data.cpu().numpy()
        Y2 = Y_hat2_2.data.cpu().numpy()
        Y3 = Y_hat2_3.data.cpu().numpy()

    Xt0 = Xt0.data.cpu().numpy()

    io.savemat('%shighresolution_sig_%s.mat' % (parameters.Output_path, count + 1),
               {'highresolution_sig': np.transpose(np.squeeze(Y1), (2, 1, 0))}, do_compression=True)
    io.savemat('%sresult32_%s.mat' % (parameters.Output_path, count + 1),
               {'result': np.squeeze(Y2)}, do_compression=True)
    io.savemat('%sresult256_%s.mat' % (parameters.Output_path, count + 1),
               {'result': np.squeeze(Y3)}, do_compression=True)

    figure_name = img_folder + 'count_%s' % (count + 1) + 'png'
    extent = [0, 1, 1, 0]
    plt.figure(figsize=(12, 12))
    colour = 'hot'

    current_32 = np.squeeze(Y2[:, :, :, :])
    current_256 = np.squeeze(Y3[:, :, :, :])

    plt.subplot(1, 2, 1)
    plt.title('albedo_32x32', fontsize=18)
    plt.imshow(np.squeeze(current_32) / np.max(current_32), vmax=1, vmin=0, extent=extent, cmap=colour)
    plt.yticks(size=15)
    plt.xticks(size=15)

    plt.subplot(1, 2, 2)
    plt.title('albedo_256x256', fontsize=18)
    plt.imshow(np.squeeze(current_256) / np.max(current_256), vmax=1, vmin=0, extent=extent, cmap=colour)
    plt.yticks(size=15)
    plt.xticks(size=15)

    plt.tight_layout()
    plt.savefig(figure_name)
    plt.close()


print('total time with storage:')
print(time.time() - start)