
import pdb
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# total variance loss
from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)


class PSFGrid(nn.Module):
    def __init__(self, channel, levels, psfVsizes, for_test=False):
        super(PSFGrid, self).__init__()
        self.channel = channel
        self.levels = levels
        self.psfVsizes = psfVsizes
        assert(len(self.levels) == len(self.psfVsizes))
        Volume = []
        for i in range(len(self.levels)):
            if for_test:
                psfV_numpy = self.create_synthetic_psf(levels[i], psfVsizes[i])
                psfVol = Parameter(1e-3*torch.from_numpy(psfV_numpy), requires_grad=False)
            else:
                psfVol = Parameter(torch.zeros(1, channel, levels[i], psfVsizes[i], psfVsizes[i]).normal_(mean=0, std=0.0001), requires_grad=True)
            Volume.append(psfVol)
        self.psfVolume = nn.ParameterList(Volume)
    
    def create_circular_mask(self, h, w, center=None, radius=None):
        '''
            To validate PSF Volume and our algorithm (for debugging)
        '''
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask
        
    def create_synthetic_psf(self, level, psfVsize, minsize=0.2, maxsize=0.8):
        '''
            To validate PSF Volume and our algorithm (for debugging)
            generate cone shape psfVolume
        '''
        min_radius = psfVsize * minsize * 0.5
        max_radius = psfVsize * maxsize * 0.5
        radius = np.linspace(min_radius, max_radius, level)
        psfV_numpy = np.zeros((1, 1, level, psfVsize, psfVsize))
        for i in range(level):
            psfV_numpy[:, :, i] = self.create_circular_mask(psfVsize, psfVsize, (psfVsize // 2, psfVsize // 2), radius[i]) * 1.0
        return np.float32(psfV_numpy)
    
    def scale_volume_grid(self, new_size):
        
        # only apply to single grid case
        if len(self.levels) == 1:
            self.psfVolume = nn.ParameterList([Parameter(F.interpolate(self.psfVolume[0].data, size=tuple(new_size), mode='trilinear', align_corners=True), requires_grad=True)])
    
    def total_variation_add_grad(self, wx, wy, wz, dense_mode=True):
        for volume in self.psfVolume:
            if volume.grad is not None:
                # add total variation loss
                total_variation_cuda.total_variation_add_grad(volume, volume.grad, wx, wy, wz, dense_mode)  # type: ignore
    
    def forward(self, xyz):
        '''
            xyz: (B, C, H, W, 3)
            interpolate given xyz coordinate
        '''
        output = []
        for volume in self.psfVolume:
            output.append(F.grid_sample(volume, xyz, mode='bilinear', padding_mode='border', align_corners=True))
        return torch.cat(output, dim=1)