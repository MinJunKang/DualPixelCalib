

import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn


# Hash based PSF Grid
class HashPSFGrid(nn.Module):
    
    def __init__(self, 
                 in_dim: int, 
                 bound_xy: float,
                 min_depth: float,
                 max_depth: float,
                 num_level: int = 16,
                 min_res: int = 16, 
                 max_res: int = 1024,
                 log2_hashmap_size: int = 19,
                 features_per_level: int = 2) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = int(features_per_level * num_level)
        self.type = 'hash'
        
        # Hash Initialization
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_level - 1))
        
        self.encoder = tcnn.Encoding(
            n_input_dims=self.in_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_level,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": growth_factor,
                "interpolation": "Linear"
            },
        )
        self.register_buffer("coord_min", torch.FloatTensor([-bound_xy,  -bound_xy,  min_depth]))
        self.register_buffer("coord_max", torch.FloatTensor([bound_xy,  bound_xy,  max_depth]))
        
    def initialize(self, bbox):
        if hasattr(self, "bbox"):
            return
        self.coord_min = bbox[0]
        self.coord_max = bbox[1]
        self.bbox = bbox
        
    def forward(self, points):
        assert points.ndim == 2
        npoints_c = (points - self.coord_min) / (self.coord_max - self.coord_min)  # [0, 1]
        feature = self.encoder(npoints_c.clamp(min=0, max=1))
        return feature

    