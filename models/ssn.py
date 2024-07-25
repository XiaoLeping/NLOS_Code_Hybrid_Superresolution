## This code is adopted from the paper "Non-line-of-sight Imaging with Signal Superresolution Network".
import functools
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import parameters

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv3d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


# Attention Branch
class AttentionBranch(nn.Module):

    def __init__(self, nf, k_size=3):

        super(AttentionBranch, self).__init__()
        self.k1 = nn.Conv3d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.k2 = nn.Conv3d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv3d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv3d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):
        
        y = self.k1(x)
        y = self.lrelu(y)
        y = self.k2(y)
        y = self.sigmoid(y)


        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class AAB(nn.Module):

    def __init__(self, nf, reduction=4, K=2, t=30):
        super(AAB, self).__init__()
        self.t = t
        self.K = K

        self.conv_first = nn.Conv3d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv3d(nf, nf, kernel_size=1, bias=False)        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction, self.K, bias=False),
        )
        
        # attention branch
        self.attention = AttentionBranch(nf)  
        # non-attention branch
        self.non_attention = nn.Conv3d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False) 

    def forward(self, x):
        residual = x
        a, b, c, d, e = x.shape

        x = self.conv_first(x)
        x = self.lrelu(x)

        # Attention Dropout
        y = self.avg_pool(x).view(a,b)
        y = self.ADM(y)
        ax = F.softmax(y/self.t, dim = 1)

        attention = self.attention(x)
        non_attention = self.non_attention(x)
    
        x = attention * ax[:,0].view(a,1,1,1,1) + non_attention * ax[:,1].view(a,1,1,1,1)
        x = self.lrelu(x)

        out = self.conv_last(x)
        out += residual

        return out



class SSN(nn.Module):
    
    def __init__(self):
        super(SSN, self).__init__()
        
        in_nc  = 1
        out_nc = 1
        nf  = 40
        unf = 24
        nb  = 16
        scale = 4

        # AAB
        AAB_block_f = functools.partial(AAB, nf=nf)
        self.scale = scale
        
        ### first convolution
        self.conv_first = nn.Conv3d(in_nc, nf, 3, 1, 1, bias=True)
        
        ### main blocks
        self.AAB_trunk = make_layer(AAB_block_f, nb)
        self.trunk_conv = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        
        #### upsampling
        self.upconv1 = nn.Conv3d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv3d(unf, unf, 3, 1, 1, bias=True)
        
        
        self.upconv2 = nn.Conv3d(unf, unf, 3, 1, 1, bias=True)
        self.att2 = PA(unf)
        self.HRconv2 = nn.Conv3d(unf, unf, 3, 1, 1, bias=True)
            
        self.conv_last = nn.Conv3d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        for para in self.modules():
            if isinstance(para,nn.Conv3d):
                nn.init.orthogonal_(para.weight)
                if para.bias is not None:
                    para.bias.data.zero_()

    def forward(self, x):

        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.AAB_trunk(fea))
        fea = fea + trunk
        
        fea = self.upconv1(F.interpolate(fea, size=[parameters.time_span,2*parameters.resolution_down_ver,2*parameters.resolution_down_hor], mode='trilinear'))#trilinear
        fea = self.lrelu(self.att1(fea))
        fea = self.lrelu(self.HRconv1(fea))
        fea = self.upconv2(F.interpolate(fea, size=[parameters.time_span,4*parameters.resolution_down_ver,4*parameters.resolution_down_hor], mode='trilinear'))
        fea = self.lrelu(self.att2(fea))
        fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)
        ILR = F.interpolate(x, size=[parameters.time_span,4*parameters.resolution_down_ver,4*parameters.resolution_down_hor], mode='trilinear', align_corners=False)
        out = out + ILR
        return out