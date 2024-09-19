
# To be implemented following "https://github.com/theNded/torch-ash.git" (hash collision free grid)

import numpy as np
import torch
import torch.nn as nn
from ash import BoundedSparseDenseGrid


# Hash based PSF Grid
class HashPSFGrid(nn.Module):
    
    def __init__(self,
                 out_dim: int,
                 bound_xy: float,
                 min_depth: float,
                 max_depth: float,
                 num_embeddings: int = 10000,
                 grid_dim: int = 16,
                 sparse_grid_dim: int = 128) -> None:
        super().__init__()
        self.in_dim = 3
        self.out_dim = out_dim
        self.type = 'hash'
        self.grid = BoundedSparseDenseGrid(in_dim=self.in_dim, 
                                           num_embeddings=num_embeddings, 
                                           embedding_dim=self.out_dim, 
                                           grid_dim=grid_dim, 
                                           sparse_grid_dim=sparse_grid_dim, 
                                           bbox_min=torch.FloatTensor([-bound_xy, -bound_xy, min_depth]),
                                           bbox_max=torch.FloatTensor([bound_xy, bound_xy, max_depth]),
                                           device='cuda')
        # self.grid.dense_init_()
        self.register_buffer("coord_min", torch.FloatTensor([-bound_xy,  -bound_xy,  min_depth]))
        self.register_buffer("coord_max", torch.FloatTensor([bound_xy,  bound_xy,  max_depth]))
        
    def initialize_grid(self, points):
        # this part should be implemented
        pass
     
    def initialize(self, bbox):
        if hasattr(self, "bbox"):
            return
        self.coord_min = bbox[0]
        self.coord_max = bbox[1]
        self.bbox = bbox
        
    def forward(self, points):
        assert points.ndim == 2
        npoints_c = (points - self.coord_min) / (self.coord_max - self.coord_min)  # [0, 1]
        feature, mask = self.grid(npoints_c.clamp(min=0, max=1), interpolation='linear')
        import pdb; pdb.set_trace()
        return feature