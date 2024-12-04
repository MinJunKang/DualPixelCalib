import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange, repeat
from .hashgrid import HashPSFGrid

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
    def __init__(self, 
                 coarse_level: int,
                 coarse_size: int,
                 bound_xy: float,
                 min_value: float,
                 max_value: float,
                 num_level: int = 16,
                 min_res: int = 16, 
                 max_res: int = 1024,
                 log2_hashmap_size: int = 19,
                 features_per_level: int = 2):
        super(PSFGrid, self).__init__()
        self.in_dim = 3
        self.psfVsize = coarse_size
        self.type = 'mixed'
        
        self.psfVolume_fine = HashPSFGrid(self.in_dim, bound_xy, min_value, max_value, num_level, min_res, max_res, log2_hashmap_size, features_per_level)
        self.psfVolume_coarse = Parameter(torch.zeros(1, self.psfVolume_fine.out_dim, coarse_level, coarse_size, coarse_size).normal_(mean=0, std=0.0001), requires_grad=True)
        self.out_dim = self.psfVolume_fine.out_dim
        
        self.register_buffer("coord_min", torch.FloatTensor([-bound_xy,  -bound_xy,  min_value]))
        self.register_buffer("coord_max", torch.FloatTensor([bound_xy,  bound_xy,  max_value]))
        
    def scale_volume_grid(self, new_world_size):
        if self.out_dim == 0:
            self.psfVolume_coarse = nn.Parameter(torch.zeros([1, self.out_dim, *new_world_size]))
        else:
            self.psfVolume_coarse = nn.Parameter(
                F.interpolate(self.psfVolume_coarse.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))
    
    def total_variation_add_grad(self, wx, wy, wz, dense_mode=True):
        if self.psfVolume_coarse.grad is not None:
            # add total variation loss
            total_variation_cuda.total_variation_add_grad(self.psfVolume_coarse, self.psfVolume_coarse.grad, wx, wy, wz, dense_mode)  # type: ignore
    
    def forward(self, points):
        '''
            points: (N, 3)
            interpolate given xyz coordinate
        '''
        shape = points.shape[:-1]
        npoints_c = points.reshape(1, 1, 1, -1, 3)
        npoints_c = ((npoints_c - self.coord_min) / (self.coord_max - self.coord_min)).flip((-1,)) * 2 - 1
        output_coarse = F.grid_sample(self.psfVolume_coarse, npoints_c, mode='bilinear', padding_mode='border', align_corners=True)
        output_coarse = output_coarse.reshape(self.out_dim,-1).T.reshape(*shape,self.out_dim)
        output_fine = self.psfVolume_fine(points)
        output = output_coarse + output_fine
        
        if self.out_dim == 1:
            output = output.squeeze(-1)
        return output