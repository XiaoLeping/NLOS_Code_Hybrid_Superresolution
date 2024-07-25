import torch
import torch.nn.functional as F
import torch.nn as nn
from models.ssn import SSN as sig_super_net
from models.tfphasor import phasor as Phasor_solver
import parameters

class Step1_SSN(nn.Module):

    def __init__(self):
        super(Step1_SSN, self).__init__()
        ## define the signal superresolution network
        self.sig_super = sig_super_net()

    def forward(self, input_signal):
        ## obtain the high-resolution signal
        sig_super = self.sig_super(input_signal)

        return sig_super


class Step2_phasor(nn.Module):

    def __init__(self):
        super(Step2_phasor, self).__init__()
        ## define the phasor solver
        self.Phasor_solver = Phasor_solver(spatial=parameters.sig_super_scale * parameters.resolution_down_ver,
                                           crop=parameters.time_span, bin_len=parameters.bin_len,
                                           wall_size=parameters.wall_size,
                                           sampling_coeff=parameters.sampling_coeff, cycles=parameters.cycles)

    def todev(self, dev):
        self.Phasor_solver.todev(dev, 1)

    def forward(self, sig_super):
        ## obtain the reconstructed albedo of Phasor solver
        Phasor_albedo = self.Phasor_solver(sig_super, [0] * parameters.batchsize,
                                           [parameters.time_span] * parameters.batchsize)

        Phasor_albedo = torch.max(Phasor_albedo, dim=2)[0]
        Phasor_albedo_max = torch.max(Phasor_albedo, dim=2, keepdim=True)[0]
        Phasor_albedo_max = torch.max(Phasor_albedo_max, dim=3, keepdim=True)[0]
        Phasor_albedo_max[Phasor_albedo_max < 0] = 0
        Phasor_albedo_max[Phasor_albedo_max == 0] = 1e-6
        Phasor_albedo = torch.div(Phasor_albedo, Phasor_albedo_max)  # 做归一化

        return Phasor_albedo

