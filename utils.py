import os
import numpy as np
import h5py
from torch.utils.data import Dataset
from datetime import datetime
from torch import nn
import parameters
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def process_signal(input_sig, input_size, downsample_ratio):
    assert input_size[0] == input_size[1]
    index = np.arange(input_size[0])
    index = index[1:input_size[0]:downsample_ratio]
    downsample_sig = input_sig[index, :, :]
    downsample_sig = downsample_sig[:, index, :]

    ##normalize
    # downsample_sig=downsample_sig/np.max(downsample_sig.flatten())
    downsample_sig = downsample_sig / np.max(input_sig.flatten())
    input_sig = input_sig / np.max(input_sig.flatten())
    return np.transpose(downsample_sig, (2, 0, 1)), np.transpose(input_sig, (2, 0, 1))