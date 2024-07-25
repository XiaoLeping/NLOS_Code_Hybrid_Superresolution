import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from models.helper import definePsf, resamplingOperator, waveconvparam, waveconv
import parameters


try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fftn(x, dim=(1, 2, 3))
        r = torch.stack((t.real, t.imag), -1)
        return r

    def ifft(x, d):

        t = torch.fft.ifftn(torch.complex(x[:, :, :, :, 0], x[:, :, :, :, 1]), dim=(1, 2, 3))
        # return t.real ** 2 + t.imag ** 2
        return t


class phasor(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 sampling_coeff=2.0, \
                 cycles=5):
        super(phasor, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        self.sampling_coeff = sampling_coeff
        self.cycles = cycles
        
        self.parpareparam()
    
    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0;
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution
        
        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        
        wall_size = self.wall_size
        bin_resolution = self.bin_resolution
        
        sampling_coeff = self.sampling_coeff
        cycles = self.cycles
        
        ######################################################
        # Step 0: define virtual wavelet properties
        # s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        # sampling_coeff = 2;  # scale the size of the virtual wavelength (usually 2, optionally 3 for noisy scenes)
        # virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        # cycles = 5;  # number of wave cycles in the wavelet, typically 4-6
        
        s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        self.virtual_wavelength = virtual_wavelength
        
        virtual_cos_wave_k, virtual_sin_wave_k = \
        waveconvparam(bin_resolution, virtual_wavelength, cycles)
        
        virtual_cos_sin_wave_2xk = np.stack([virtual_cos_wave_k, virtual_sin_wave_k], axis=0)
        
        # use pytorch conv to replace matlab conv
        self.virtual_cos_sin_wave_inv_2x1xk = torch.from_numpy(virtual_cos_sin_wave_2xk[:, ::-1].copy()).unsqueeze(1)
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf(sptial_grid, temprol_grid, slope)
        fpsf = np.fft.fftn(psf)
        invpsf = np.conjugate(fpsf)
        
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
    def todev(self, dev, dnum):
        self.virtual_cos_sin_wave_inv_2x1xk_todev = self.virtual_cos_sin_wave_inv_2x1xk.to(dev)
        self.datapad_2Dx2Tx2Hx2W = torch.zeros((2 * dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32, device=dev)
        
        self.mtx_MxM_todev = self.mtx_MxM.to(dev)
        self.mtxi_MxM_todev = self.mtxi_MxM.to(dev)
        
        self.invpsf_real_todev = self.invpsf_real.to(dev)
        self.invpsf_imag_todev = self.invpsf_imag.to(dev)

        if parameters.n_gpu == 2:  ##suppose there is two gpus, add more copies if needed
            self.virtual_cos_sin_wave_inv_2x1xk_todev_cudacopy = self.virtual_cos_sin_wave_inv_2x1xk_todev.to('cuda:1')
            self.datapad_2Dx2Tx2Hx2W_copy = self.datapad_2Dx2Tx2Hx2W.to('cuda:1')
            self.mtx_MxM_todev_copy = self.mtx_MxM_todev.to('cuda:1')
            self.mtxi_MxM_todev_copy = self.mtxi_MxM_todev.to('cuda:1')
            self.invpsf_real_todev_copy = self.invpsf_real_todev.to('cuda:1')
            self.invpsf_imag_todev_copy = self.invpsf_imag_todev.to('cuda:1')

    def forward(self, feture_bxdxtxhxw, tbes, tens):
        
        # 1 padd data with zero
        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape
        for tbe, ten in zip(tbes, tens):
            assert tbe >= 0
            assert ten <= self.crop
        dev = feture_bxdxtxhxw.device
        
        featpad_bxdxtxhxw = []
        for i in range(bnum):
            featpad_1xdxt1xhxw = torch.zeros((1, dnum, tbes[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxt2xhxw = torch.zeros((1, dnum, self.crop - tens[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxtxhxw = torch.cat([featpad_1xdxt1xhxw, feture_bxdxtxhxw[i:i + 1], featpad_1xdxt2xhxw], dim=2)
            featpad_bxdxtxhxw.append(featpad_1xdxtxhxw)
        featpad_bxdxtxhxw = torch.cat(featpad_bxdxtxhxw, dim=0)
        
        # 2 params
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop
        tnum = self.crop
        
        ####################################################
        # 3 run lct
        # assert bnum == 1
        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, tnum, hnum, wnum)
        
        ############################################################
        # Step 1: convolve measurement volume with virtual wave
        
        data_BDxHxWxT = data_BDxTxHxW.permute(0, 2, 3, 1)
        data_BDHWx1xT = data_BDxHxWxT.reshape(-1, 1, tnum)
        knum = self.virtual_cos_sin_wave_inv_2x1xk.shape[2]

        if self.virtual_cos_sin_wave_inv_2x1xk_todev.device == dev:
            phasor_data_cos_sin_BDHWx2x1T = F.conv1d(data_BDHWx1xT, self.virtual_cos_sin_wave_inv_2x1xk_todev, padding=knum // 2)
        else:
            phasor_data_cos_sin_BDHWx2x1T = F.conv1d(data_BDHWx1xT, self.virtual_cos_sin_wave_inv_2x1xk_todev_cudacopy, padding=knum // 2)


        if knum % 2 == 0:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T[:, :, 1:]
        else:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T
        
        data_BDxHxWx2xT = data_BDHWx2xT.reshape(bnum * dnum, hnum, wnum, 2, tnum)
        data_2xBDxTxHxW = data_BDxHxWx2xT.permute(3, 0, 4, 1, 2)
        data_2BDxTxHxW = data_2xBDxTxHxW.reshape(2 * bnum * dnum, tnum, hnum, wnum)
        
        #############################################################    
        # Step 2: transform virtual wavefield into LCT domain
        
        # datapad_2BDx2Tx2Hx2W = torch.zeros((2 * bnum * dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        if self.datapad_2Dx2Tx2Hx2W.device == dev:
            datapad_2Dx2Tx2Hx2W = self.datapad_2Dx2Tx2Hx2W
        else:
            datapad_2Dx2Tx2Hx2W = self.datapad_2Dx2Tx2Hx2W_copy
        # create new variable
        datapad_B2Dx2Tx2Hx2W = datapad_2Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)
        # actually, because it is all zero so it is ok
        datapad_2BDx2Tx2Hx2W = datapad_B2Dx2Tx2Hx2W

        if self.mtx_MxM_todev.device == dev:
            left = self.mtx_MxM_todev
        else:
            left = self.mtx_MxM_todev_copy

        right = data_2BDxTxHxW.view(2 * bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(2 * bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        datapad_2BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2
        
        ###########################################################3
        # Step 3: convolve with backprojection kernel
        
        datafre = rfft(datapad_2BDx2Tx2Hx2W, 3)
        datafre_real = datafre[:, :, :, :, 0]
        datafre_imag = datafre[:, :, :, :, 1]

        if self.invpsf_real_todev.device == dev:
            re_real = datafre_real * self.invpsf_real_todev - datafre_imag * self.invpsf_imag_todev
            re_imag = datafre_real * self.invpsf_imag_todev + datafre_imag * self.invpsf_real_todev
        else:
            re_real = datafre_real * self.invpsf_real_todev_copy - datafre_imag * self.invpsf_imag_todev_copy
            re_imag = datafre_real * self.invpsf_imag_todev_copy + datafre_imag * self.invpsf_real_todev_copy


        refre = torch.stack([re_real, re_imag], dim=4)
        re = ifft(refre, 3)
        re = torch.stack((re.real, re.imag), -1)
        volumn_2BDxTxHxWx2 = re[:, :temprol_grid, :sptial_grid, :sptial_grid, :]

        ########################################################################
        # Step 4: compute phasor field magnitude and inverse LCT
        
        cos_real = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :, 0]
        cos_imag = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :, 1]
        
        sin_real = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :, 0]
        sin_imag = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :, 1]
        
        sum_real = cos_real ** 2 - cos_imag ** 2 + sin_real ** 2 - sin_imag ** 2
        sum_image = 2 * cos_real * cos_imag + 2 * sin_real * sin_imag
        
        tmp = (torch.sqrt(sum_real ** 2 + sum_image ** 2) + sum_real) / 2
        # numerical issue
        tmp = F.relu(tmp, inplace=False)
        sqrt_sum_real = torch.sqrt(tmp)
        
        #####################################################################
        if self.mtxi_MxM_todev.device == dev:
            left = self.mtxi_MxM_todev
        else:
            left = self.mtxi_MxM_todev_copy
        right = sqrt_sum_real.view(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        ########################################################################
        # do we force to be > 0?
        # volumn_BDxTxHxW = F.relu(tmp2, inplace=False)
        volumn_BDxTxHxW = tmp2
        
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)

        if parameters.dataset_name == 'Stanford_dynamic':
            volumn_BxDxTxHxW = torch.transpose(volumn_BxDxTxHxW, 3, 4)

        elif parameters.dataset_name == 'testset':
            volumn_BxDxTxHxW = torch.transpose(volumn_BxDxTxHxW,3,4)

        elif parameters.dataset_name == 'Experiment18m':
            volumn_BxDxTxHxW = torch.rot90(volumn_BxDxTxHxW, k=1, dims=[3, 4])
            volumn_BxDxTxHxW = torch.transpose(volumn_BxDxTxHxW,3,4)
            volumn_BxDxTxHxW = torch.rot90(volumn_BxDxTxHxW, k=2, dims=[3, 4])

        return volumn_BxDxTxHxW