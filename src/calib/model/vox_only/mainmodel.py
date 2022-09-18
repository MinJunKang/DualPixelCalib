

import os
import pdb
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops.einops import rearrange
from kornia.filters import Laplacian, spatial_gradient, GaussianBlur2d

import pytorch_lightning as pl


from src.calib.extern.pacnet.pac import conv2d  # TODO: change to own implementation (spatially varying conv)

# from src.calib.utils.metric import CalibMetric
from src.calib.utils.loader import normalize, masked_mean
from src.calib.psflearner import optimizer_selector
# from src.calib.utils.visualizer import visualize_PSFVolume, visualize_PSFVolume_test, visualize_samples


class PSFVolume(pl.LightningModule):
    
    def __init__(self, data, opt):
        super(PSFVolume, self).__init__()
        
        # save hyperparameters
        self.save_hyperparameters(ignore='data')
        
        # parameters
        self.opt = opt
        self.multi_res = opt.model_cfg.multi_res
        self.max_level = opt.model_cfg.level
        self.max_psfV_size = opt.model_cfg.psfV_size
        self.max_psfV_size = self.max_psfV_size + 1 if self.max_psfV_size % 2 == 0 else self.max_psfV_size  # odd number
        self.max_psf_uvsize = np.abs(data['disp_range']).max() * opt.model_cfg.patch_margin
        self.depth_range = [data['depth_range'][0][0], data['depth_range'][1][0]]
        
        # define psf volume
        self.psfV_scale = 1.0
        self.level = int(self.max_level * self.psfV_scale)
        self.psfV_size = self.max_psfV_size  # initially use the largest size
        self.psf_volume = Parameter(torch.zeros(1, 1, self.level, self.psfV_size, self.psfV_size), requires_grad=True)  # [L, P, P]
        
        # filters for gradient map
        self.gradient_method = 'sobel'  # option : sobel, laplacian
        self.laplace = Laplacian(kernel_size=3, border_type='constant')
        self.gaussian = GaussianBlur2d(kernel_size=(7, 7), sigma=(1.5, 1.5), border_type='constant')
        
        # register parameters
        self.register_parameter('min_depth', Parameter(torch.tensor(self.depth_range[0]), requires_grad=False))
        self.register_parameter('max_depth', Parameter(torch.tensor(self.depth_range[1]), requires_grad=False))
        
        # visualization setting
        self.record_epoch = opt.model_cfg.record_epoch
        self.num_vis = opt.model_cfg.num_vis
        
    def model_setting(self, cfg):
        
        # visualizer setting
        self.sampled_index = cfg['vis_idx']
        self.modelpath = cfg['modelpath']
        self.resultpath = cfg['resultpath']
        
        # callback setting
        pdb.set_trace()
        
    def create_circular_mask(self, h, w, center=None, radius=None):
        '''
            To validate PSF Volume and our algorithm
        '''
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask
        
    def create_synthetic_psf(self, level, minsize=0.2, maxsize=0.8):
        '''
            To validate PSF Volume and our algorithm
        '''
        min_radius = self.psfV_size * minsize * 0.5
        max_radius = self.psfV_size * maxsize * 0.5
        radius = np.linspace(min_radius, max_radius, level)
        psfV_numpy = np.zeros((1, 1, level, self.psfV_size, self.psfV_size))
        for i in range(level):
            psfV_numpy[:, :, i] = self.create_circular_mask(self.psfV_size, self.psfV_size, (self.psfV_size // 2, self.psfV_size // 2), radius[i]) * 1.0
        return np.float32(psfV_numpy)
        
    def gradient_apply(self, feat, clean, mask=None):
        
        # compute gradient map
        if self.gradient_method == 'laplacian':
            gradient = self.laplace(feat)
        elif self.gradient_method == 'sobel':
            gradient = torch.sqrt(torch.square(spatial_gradient(feat)).sum(dim=2))
        else:
            gradient = feat
        gradient = normalize(self.gaussian(gradient), mask)
        
        # fill the centers (fill hole)
        medianvalue = torch.median(gradient[:, clean[0] > 0], dim=1)[0].view(-1, 1)
        gradient[:, clean[0] > 0] = medianvalue
        gradient = normalize(self.gaussian(gradient), mask)
        return gradient
    
    def CharbonnierLoss_L1(self, x, scale=0.1):
        # alpha=1: Charbonnier/pseudo-Huber loss.
        assert torch.is_tensor(x)
        
        # This will be used repeatedly.
        squared_scaled_x = (x / scale) ** 2
        return torch.pow(squared_scaled_x + 1., 0.5) - 1.
    
    def CharbonnierLoss_L2(self, x, scale=0.1):
        # alpha=2: L2 loss.
        assert torch.is_tensor(x)
        
        # This will be used repeatedly.
        squared_scaled_x = (x / scale) ** 2

        # The loss when alpha == 2.
        return 0.5 * squared_scaled_x
    
    def SSIM(self, x, y, conf=None):
        """ Compute the structural similarity index between two images

        Args:
            x: (n_batch, n_dim, nx, ny) input image
            y: (n_batch, n_dim, nx, ny) input image
            conf: (n_batch, n_dim, nx, ny) input confidence map

        Returns:
            (float) structural similarity measure
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = torch.nn.AvgPool2d(3, 1)(x)
        mu_y = torch.nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = torch.nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = torch.nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        if conf is not None:
            return torch.clamp((1 - SSIM) / 2, 0, 1) * torch.nn.AvgPool2d(3, 1)(conf)
        else:
            return torch.clamp((1 - SSIM) / 2, 0, 1)
        
    def calc_psfVbound(self, K_mat):
        '''
            For PSF Volume, we should define max_xy size,
            which is the max_xy of closest patch (min_depth).
            For this patch, the following equation should be hold.
            K'[u, v, 1] = K[u + du, v + dv, 1], du, dv is the uv coord of patch[0, 0]
            closest patch coords : [0 ~ max_psfV_size, 0 ~ max_psfV_size, min_depth]
            After tedious calculation, max_xy = max(max_psfV_size / fx, max_psfV_size / fy) * min_depth
        '''
        return torch.max(self.max_psf_size / K_mat[:, 0, 0], self.max_psf_size / K_mat[:, 1, 1]) * self.min_depth  # [B]
    
    def active_sampling(self, gradient, mask, use_sampling=True):
        
        if self.training and use_sampling:
            score = gradient[mask]
            b_ids = torch.stack(torch.where(mask), axis=0)
            num_sampled = int(len(score) * self.num_sampled_ratio)
            
            # randomly select k samples based on score
            prob = score / score.sum()
            with torch.no_grad():
                idx = np.random.choice(np.arange(len(prob)), num_sampled, p=prob.cpu().numpy())
                selected_idx = b_ids[:, idx]
        else:
            selected_idx = torch.stack(torch.where(torch.ones_like(mask)), axis=0)
            num_sampled = int(selected_idx.shape[-1] * self.num_sampled_ratio)
            
        return selected_idx, num_sampled
    
    def forward(self, batch, record_results=True):
        
        # normalize input
        coord = batch['uv_coord']  # [B, 2, H, W]
        depth, mask = batch['depth'], batch['depth'] > 0  # [B, 1, H, W]
        clean = normalize(batch['clean'], mask)  # [B, C, H, W]
        blur = normalize(batch['blur'], mask)  # [B, C, H, W]
        iK_mat = torch.linalg.inv(batch['intmat'].squeeze(1))  # [B, 3, 3]  -> inverse intrinsic matrix
        max_psfV_size_xy = self.calc_max_psfVsize(batch['intmat'].squeeze(1))
        
        # shape define
        b, c, h, w = clean.shape
        
        # mask aware sparse sampling
        
        
        # get gradient map (weight map)
        gradient = self.gradient_apply(blur, clean, mask)  # [B, C, H, W]
        gradient = gradient * batch['weight']  # reweighting by LDS
        
        # coordinate from (u, v, 1) to (x, y, z)
        norm_coord = torch.cat([coord, torch.ones_like(depth)], dim=1)  # [B, 3, H, W]
        if self.method == 'ortho':
            depth_ = masked_mean(depth.view(b, -1), mask.view(b, -1), 1).view(b, 1, 1, 1).repeat(1, 1, h, w)
            xyz_coord = torch.einsum('bij,bjhw->bihw', iK_mat, norm_coord) * depth_  # [B, 3, H, W]
        else:
            xyz_coord = torch.einsum('bij,bjhw->bihw', iK_mat, norm_coord) * depth  # [B, 3, H, W]
        
        # Unfolding
        coord_unfold = F.unfold(xyz_coord, self.prj_psf_size, 1, self.prj_psf_size // 2, 1)  # [B, 3 * PSFsize * PSFsize, H * W]
        coord_unfold = rearrange(coord_unfold, 'b (c kh kw) l -> b c kh kw l', kh=self.prj_psf_size, kw=self.prj_psf_size)  # [B, 3, PSFsize, PSFsize, H * W]
        coord_unfold_center = coord_unfold[:, :2, self.prj_psf_size // 2, self.prj_psf_size // 2, :].view(b, 2, 1, 1, -1)  # [B, 2, 1, 1, H * W]
        mask_unfold = (coord_unfold[:, -1] > 0).unsqueeze_(1).repeat(1, 3, 1, 1, 1)  # [B, 3, PSFsize, PSFsize, H * W]
        
        # normalize grid [-1, 1]
        max_psfV_size_xy = max_psfV_size_xy.view(b, 1, 1, 1, 1)
        coord_unfold[:, :2] = (coord_unfold[:, :2] - coord_unfold_center) / ((max_psfV_size_xy - 1) / 2)
        coord_unfold[:, 2] = (coord_unfold[:, 2] - self.min_depth) / (self.max_depth - self.min_depth) * 2.0 - 1.0  # type: ignore
        
        # masking unwanted area (depth == 0)
        coord_unfold = coord_unfold.masked_fill_(~mask_unfold, -2.0)
        coord_unfold = coord_unfold.permute(0, 4, 2, 3, 1)  # [B, H * W, PSFsize, PSFsize, 3]
        
        # weight kernel from PSF Volume
        psfVolume = self.psf_volume.repeat(b, 1, 1, 1, 1)  # [B, C, Level, PSFVolume_H, PSFVolume_W]
        kernel = F.grid_sample(psfVolume, coord_unfold, mode='bilinear', padding_mode='border', align_corners=True)
        kernel = torch.abs(kernel.view(b, c, h, w, self.prj_psf_size, self.prj_psf_size))  # [B, C, H, W, PSFsize, PSFsize]
        kernel_ = kernel.permute(0, 1, 4, 5, 2, 3)  # [B, C, PSFsize, PSFsize, h, w]
        
        # spatially varying conv
        output = conv2d(clean, kernel_, kernel_size=self.prj_psf_size, stride=1, padding=self.prj_psf_size // 2, dilation=1)
        
        loss = dict()
        if self.training:
            # Reconstruction Loss (Blur Loss)
            loss.update({'loss_main': self.params['arg'].loss_weights[0] * torch.mean(self.CharbonnierLoss_L2(gradient * (blur - output), self.scale))})
            
            # SSIM Loss
            loss.update({'loss_ssim': self.params['arg'].loss_weights[1] * torch.mean(self.SSIM(blur, output, gradient))})
            
            # L1 regularization Loss
            loss.update({'loss_l1': self.params['arg'].loss_weights[2] * torch.mean(torch.norm(kernel.view(-1, self.prj_psf_size ** 2), p=1, dim=-1))})  # type: ignore
        
    def configure_optimizers(self):
        # optimizer and schedular
        opt = optimizer_selector(self.parameters(), self.opt)
        
        if self.params['arg'].use_hierarchical:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, milestones=self.params['milestones'], 
                                                             gamma=0.5, last_epoch=-1)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=self.params['arg'].epoch // 4, 
                                                        gamma=0.5, last_epoch=-1)
        
        return {'optimizer': opt, 'lr_scheduler': scheduler}
    
    
