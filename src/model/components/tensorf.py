

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class TensoRF(nn.Module):
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int, 
                 bound_xy: float,
                 min_value: float,
                 max_value: float,
                 num_level: int = 3,  # 4
                 depth_res: int = 64,
                 min_res: int = 64,   # 32
                 max_res: int = 256):
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_level = num_level
        
        self.plane_coefs = nn.ParameterList()
        res_multi = np.linspace(min_res, max_res, num_level)
        for i in range(num_level):
            res = int(res_multi[i])
            plane_coef = torch.nn.Parameter(0.01 * torch.randn((1, self.out_dim, res, res)), requires_grad=True)
            self.plane_coefs.append(plane_coef)
        self.line_coef = torch.nn.Parameter(0.01 * torch.randn((1, self.out_dim, depth_res, 1)), requires_grad=True)
        
        self.register_buffer("coord_min", torch.FloatTensor([-bound_xy,  -bound_xy,  min_value]))
        self.register_buffer("coord_max", torch.FloatTensor([bound_xy,  bound_xy,  max_value]))
    
    def initialize(self, bbox):
        if hasattr(self, "bbox"):
            return
        self.coord_min = bbox[0]
        self.coord_max = bbox[1]
        self.bbox = bbox
        
    def forward(self, points):
        assert points.ndim == 2
        npoints_c = (points - self.coord_min) / (self.coord_max - self.coord_min) * 2 - 1  # [-1, 1]
        npoints_p = npoints_c[..., :2]
        npoints_d = npoints_c[..., 2:]
        
        coordinate_plane = rearrange(npoints_p, 'b d -> 1 b 1 d')
        coordinate_line = rearrange(torch.cat((torch.zeros_like(npoints_d), npoints_d), dim=-1), 'b d -> 1 b 1 d')
        
        plane_feats = []
        for i in range(self.num_level):
            plane_feat = F.grid_sample(self.plane_coefs[i], coordinate_plane, align_corners=True).view(-1, *points.shape[:1])
            plane_feats.append(plane_feat)
        plane_feats = torch.stack(plane_feats, dim=0)
        line_feats = F.grid_sample(self.line_coef, coordinate_line, align_corners=True).view(-1, *points.shape[:1])
        sigma_feature = torch.sum(plane_feats * line_feats[None], dim=0)
        
        return sigma_feature.T