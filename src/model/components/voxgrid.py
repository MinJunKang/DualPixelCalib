
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange


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
    def __init__(self, channel, level, psfVsize, bound_xy, min_depth, max_depth, for_test=False):
        super(PSFGrid, self).__init__()
        self.channel = channel
        self.level = level
        self.psfVsize = psfVsize
        
        if for_test:
            psfV_numpy = self.create_synthetic_psf(level, psfVsize)
            self.psfVolume = Parameter(1e-3*torch.from_numpy(psfV_numpy), requires_grad=False)
        else:
            self.psfVolume = Parameter(torch.zeros(1, self.channel, level, psfVsize, psfVsize).normal_(mean=0, std=0.0001), requires_grad=True)
        
        self.register_buffer("coord_min", torch.FloatTensor([-bound_xy,  -bound_xy,  min_depth]))
        self.register_buffer("coord_max", torch.FloatTensor([bound_xy,  bound_xy,  max_depth]))
    
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
    
    def create_synthetic_psf(self, level, psfVsize, minsize=0.1, maxsize=0.5):
        '''
            To validate PSF Volume and our algorithm (for debugging)
            generate cone shape psfVolume
        '''
        min_radius = psfVsize * minsize
        max_radius = psfVsize * maxsize
        radius = np.linspace(min_radius, max_radius, level)
        psfV_numpy = np.zeros((1, 1, level, psfVsize, psfVsize))
        for i in range(level):
            psfV_numpy[:, :, i] = self.create_circular_mask(psfVsize, psfVsize, (psfVsize // 2, psfVsize // 2), radius[i]) * 1000
        return np.float32(psfV_numpy)
    
    def scale_volume_grid(self, new_size):
        # only apply to single grid case
        self.psfVolume = nn.ParameterList([Parameter(F.interpolate(self.psfVolume.data, size=tuple(new_size), mode='trilinear', align_corners=True), requires_grad=True)])
    
    def total_variation_add_grad(self, wx, wy, wz, dense_mode=True):
        if self.psfVolume.grad is not None:
            # add total variation loss
            total_variation_cuda.total_variation_add_grad(self.psfVolume, self.psfVolume.grad, wx, wy, wz, dense_mode)  # type: ignore
    
    def forward(self, points):
        '''
            points: (B, C, H, W, 3)
            interpolate given xyz coordinate
        '''
        npoints_c = (points - self.coord_min) / (self.coord_max - self.coord_min)
        output = F.grid_sample(self.psfVolume, npoints_c, mode='bilinear', padding_mode='border', align_corners=True)
        return output.abs()
    
    def compute_grid(self, device=None):
        if device is None: 
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        z_grid, y_grid, x_grid = torch.meshgrid([torch.linspace(-1.0, 1.0, self.level), 
                                                torch.linspace(-1.0, 1.0, self.psfVsize), 
                                                torch.linspace(-1.0, 1.0, self.psfVsize)])
        grid = torch.stack((x_grid, y_grid, z_grid), dim=-1).float().to(device)
        grid = rearrange(grid, 'd h w c -> 1 d h w c')
            
        return grid