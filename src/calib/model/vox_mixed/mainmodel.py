

import os
import pdb
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops.einops import rearrange, repeat
from torch_scatter import segment_coo
from kornia.filters import Laplacian, spatial_gradient, GaussianBlur2d

import pytorch_lightning as pl

from src.calib.model.grid import PSFGrid
from src.calib.extern.pacnet.pac import conv2d  # TODO: change to own implementation (spatially varying conv)

# from src.calib.utils.metric import CalibMetric
from src.calib.utils.loader import normalize, masked_mean
from src.calib.psflearner import optimizer_selector
# from src.calib.utils.visualizer import visualize_PSFVolume, visualize_PSFVolume_test, visualize_samples


def positional_encoding(positions, freqs):
    freq_bands = 2**torch.arange(freqs, dtype=torch.float32, device=positions.device)*np.pi  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class MLP_Fea(torch.nn.Module):
    def __init__(self,inChanel, feape=6, featureC=128):
        super(MLP_Fea, self).__init__()

        self.in_mlpC = 2 * feape * inChanel + inChanel + 3
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 1)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, features):
        indata = [pts, features]
        
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)
        psf = self.mlp(mlp_in)
        psf = torch.abs(psf)

        return psf


class PSFVolume(pl.LightningModule):
    
    def __init__(self, data, opt):
        super(PSFVolume, self).__init__()
        
        # save hyperparameters
        self.save_hyperparameters(ignore='data')
        
        # parameters
        self.opt = opt
        
        # define volume level
        self.levels = [int(opt.model_cfg.level * scale / opt.model_cfg.scales[-1]) for scale in opt.model_cfg.scales]
        
        # define volume size
        psfV_sizes = [int(opt.model_cfg.psfV_minsize * scale / opt.model_cfg.scales[0]) for scale in opt.model_cfg.scales]
        self.psfV_sizes = [psfV_size + 1 if psfV_size % 2 == 0 else psfV_size for psfV_size in psfV_sizes]  # odd number
        
        # define kernel size in uv space
        self.max_psf_uvsize = np.abs(data['disp_range']).max() * opt.model_cfg.patch_margin
        # define depth range to apply
        self.depth_range = [data['depth_range'][0][0], data['depth_range'][1][0]]
        
        # coarse to fine learning
        self.stage = 0  # progressive training stage (0 ~ N - 1)
        self.psfV_scale = self.opt.model_cfg.scales[self.stage] / opt.model_cfg.scales[0]
        
        # define PSF Volume
        if opt.model_cfg.strategy == 'singlegrid':
            self.psf_volume = PSFGrid(opt.model_cfg.vox_channel, [self.levels[0]], [self.psfV_sizes[0]]) # [C, L, P, P]
            self.in_channel = opt.model_cfg.vox_channel
        elif opt.model_cfg.strategy == 'multigrid':
            self.psf_volume = PSFGrid(opt.model_cfg.vox_channel, self.levels, self.psfV_sizes) # [C, L, P, P]
            self.in_channel = opt.model_cfg.vox_channel * len(self.levels)
        else:
            raise NotImplementedError
        
        # filters for gradient map
        self.gradient_method = 'sobel'  # option : sobel, laplacian
        self.laplace = Laplacian(kernel_size=3, border_type='constant')
        self.gaussian = GaussianBlur2d(kernel_size=(7, 7), sigma=(1.5, 1.5), border_type='constant')
        
        # MLPs for PSF volume
        self.mlp_FE = MLP_Fea(self.in_channel, feape=opt.model_cfg.feape, featureC=opt.model_cfg.mlp_channel)
        
        # register parameters
        self.register_parameter('min_depth', Parameter(torch.tensor(self.depth_range[0]), requires_grad=False))
        self.register_parameter('max_depth', Parameter(torch.tensor(self.depth_range[1]), requires_grad=False))
        
        # visualization setting
        self.record_epoch = opt.model_cfg.record_epoch
        self.num_vis = opt.model_cfg.num_vis
        
    def model_setting(self, cfg):
        
        # visualizer setting
        self.img_size = cfg['img_size']
        self.sampled_index = cfg['vis_idx']
        self.modelpath = cfg['modelpath']
        self.resultpath = cfg['resultpath']
        self.psf_prj_scale = np.array(cfg['psf_prj_scale'])
        self.psf_prj_size = np.ceil(self.max_psf_uvsize * self.psf_prj_scale)
        self.psf_prj_size[self.psf_prj_size % 2 == 0] += 1  # odd number
        
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
        fx, fy = K_mat[:, 0, 0], K_mat[:, 1, 1]
        return torch.max(self.max_psf_uvsize / fx, self.max_psf_uvsize / fy) * self.max_depth  # [B]
    
    def prepare_inputs(self, batch):
        
        # inputs
        clean = batch['clean'][self.stage]
        blur = batch['blur'][self.stage]
        focus = batch['focus'][self.stage]
        depth = batch['depth'][self.stage]
        normal = batch['normal'][self.stage]
        intmat = batch['intmat'][self.stage]
        uvcoord = batch['uv_coord'][self.stage]
        
        # masking and calculate bounds
        mask = depth > 0
        invKmat = torch.inverse(intmat).squeeze(1) # [B, 3, 3]
        psfV_max_xy = self.calc_psfVbound(intmat.squeeze(1))
        gradient = self.gradient_apply(blur, clean, mask) * batch['weight'][self.stage]  # reweighting by LDS
        mask = mask & (gradient > self.opt.model_cfg.grad_thres)  # mask by gradient map
        mask = mask & (uvcoord[:, 0:1] < self.img_size[1]) & (uvcoord[:, 1:2] < self.img_size[0]) & (uvcoord[:, 0:1] > 0) & (uvcoord[:, 1:2] > 0)  # mask by image boundary
        gradient = gradient * mask.float()
        
        # normalize input
        clean = normalize(clean, mask)  # [B, C, H, W]
        blur = normalize(blur, mask)  # [B, C, H, W]
        
        return clean, blur, focus, depth, normal, mask, gradient, invKmat, psfV_max_xy, uvcoord
    
    def forward(self, batch, record_results=True):
        
        # prepare inputs
        psf_prj_sz = int(self.psf_prj_size[self.stage])
        clean, blur, focus, depth, normal, mask, gradient, invKmat, psfV_max_xy, uvcoord = self.prepare_inputs(batch)
        
        # masking
        batchnum = clean.shape[0]
        index = torch.arange(clean.numel(), device=clean.device)  # [B * 1 * H * W]
        index_masked = index.view(*clean.shape).permute(0, 2, 3, 1)[mask.squeeze(1)]  # [N, 1]
        
        # coordinate from (u, v, 1) to (x, y, z)
        norm_coord = torch.cat([uvcoord, torch.ones_like(depth)], dim=1)  # [B, 3, H, W]
        xyz_coord = torch.einsum('bij,bjhw->bihw', invKmat, norm_coord) * depth  # [B, 3, H, W]
        
        # Unfolding
        clean_unfold = F.unfold(clean, psf_prj_sz, 1, psf_prj_sz // 2, 1)  # [B, 1 * PSFsize * PSFsize, H * W]
        clean_unfold = rearrange(clean_unfold, 'b (c kh kw) l -> b l kh kw c', kh=psf_prj_sz, kw=psf_prj_sz)  # [B, H * W, PSFsize, PSFsize, 1]
        coord_unfold = F.unfold(xyz_coord, psf_prj_sz, 1, psf_prj_sz // 2, 1)  # [B, 3 * PSFsize * PSFsize, H * W]
        coord_unfold = rearrange(coord_unfold, 'b (c kh kw) l -> b c kh kw l', kh=psf_prj_sz, kw=psf_prj_sz)  # [B, 3, PSFsize, PSFsize, H * W]
        coord_unfold_center = rearrange(coord_unfold[:, :2, psf_prj_sz // 2, psf_prj_sz // 2], 'b c l -> b c 1 1 l')  # [B, 2, 1, 1, H * W]
        
        # normalize grid [-1, 1]
        coord_unfold[:, :2] = (coord_unfold[:, :2] - coord_unfold_center) / (psfV_max_xy[:, None, None, None, None] / 2)
        coord_unfold[:, 2] = (coord_unfold[:, 2] - self.min_depth) / (self.max_depth - self.min_depth) * 2.0 - 1.0  # type: ignore
        coord_unfold = rearrange(coord_unfold, 'b c kh kw l -> b l kh kw c')  # [B, H * W, PSFsize, PSFsize, 3]
        
        # masking unwanted area
        coord_unfold = coord_unfold[mask.view(batchnum, -1)]  # [N, PSFsize, PSFsize, 3]
        clean_unfold = clean_unfold[mask.view(batchnum, -1)]  # [N, PSFsize, PSFsize, 1]
        
        # sample kernel from PSF Volume
        kernel_feat = self.psf_volume(coord_unfold[None])  # [1, C, N, PSFsize, PSFsize]
        coord_unfold = rearrange(coord_unfold, 'n kh kw c -> n (kh kw) c')
        kernel_feat = rearrange(kernel_feat, '1 c n kh kw -> n (kh kw) c')
        clean_unfold = rearrange(clean_unfold, 'n kh kw c -> n (kh kw) c')
        kernel = self.mlp_FE(coord_unfold, kernel_feat)
        
        # spatially varying conv
        output = segment_coo(src=(kernel * clean_unfold).sum(dim=1).view(-1), 
                             index=index_masked.view(-1), 
                             out=torch.zeros([clean.numel()], device=clean.device), 
                             reduce='sum')
        output = output.view(*clean.shape)
        
        loss = dict()
        # Reconstruction Loss (Reblur Loss)
        loss.update({'loss_main': self.opt.model_cfg.loss_weights[0] * torch.mean(self.CharbonnierLoss_L2(gradient * (blur - output)))})
        
        # SSIM loss
        loss.update({'loss_ssim': self.opt.model_cfg.loss_weights[1] * torch.mean(self.SSIM(blur, output, gradient))})
        
        # L1 regularization Loss
        loss.update({'loss_l1': self.opt.model_cfg.loss_weights[2]* torch.mean(torch.norm(kernel, p=1, dim=1))})  # type: ignore
        
        if not self.training:
            '''
                Render full psf volume for visualization
            '''
            psfVolume_full = None
        else:
            psfVolume_full = None
        
        if record_results:
            results = {'mask': mask, 'gradient': gradient, 'convolved': output, 'psfVolume': psfVolume_full}
        else:
            results = None
        
        return loss, results
    
    def training_step(self, batch, batch_idx):
        
        loss, _ = self.forward(batch, record_results=False)
        
        loss_total = 0.0
        for key in loss.keys():
            self.log(key, loss[key].detach(), prog_bar=True)
            loss_total = loss_total + loss[key]
        self.log('total_loss', loss_total.detach(), prog_bar=True)
        
        return loss_total
    
    def validation_step(self, batch, batch_idx):
        if (self.current_epoch + 1) % self.record_epoch == 0:
            loss, results = self.forward(batch)
            
            # visualize volume
            # if batch_idx == 0:
            #     visualize_PSFVolume(self.psf_volume[0, 0], self.min_depth, self.max_depth, self.params['savepath'], self.current_epoch + 1)
                
            # # calculate metrics
            # metrics = self.metric_logger.measure(batch, results)
                
            # # visualize results
            # if batch_idx in self.sampled_index:
            #     sample_name = '_sample%04d_rmse_%.5f' % (batch_idx + 1, metrics['rmse'])
            #     visualize_samples(batch, results, self.params['savepath'], self.current_epoch + 1, sample_name)
                
    def validation_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.record_epoch == 0:
            # metric evaluation in this step
            print('Metric results summary : ')
            # results, t = self.metric_logger.get_value(use_chart=True)
            # if t is not None:
            #     print(t.draw())
    
    def configure_optimizers(self):
        # optimizer and schedular
        optimizer = optimizer_selector(self.parameters(), self.opt)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, 
                                                      lr_lambda=lambda epoch: self.opt.model_cfg.lrate_decay ** (epoch / self.opt.epoch), 
                                                      last_epoch=-1, verbose=False)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
