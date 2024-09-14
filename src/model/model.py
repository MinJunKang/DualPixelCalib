from typing import Any, Dict, Tuple

import random
from copy import deepcopy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_scatter import segment_coo
from einops import rearrange, repeat
from lightning import LightningModule
from kornia.filters import Laplacian, spatial_gradient, GaussianBlur2d
from src.model.utils.common import normalize, unfolding
from kornia.losses import charbonnier_loss, ssim_loss, total_variation
from src.extern.pacnet.pac import conv2d
from torchmetrics import MeanMetric


class PSFVolumeModule(LightningModule):
    
    def __init__(
        self,
        output_dir: str,
        meta_data: Dict[str, Any],
        model_cfg: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        diffusion_model: nn.Module = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore='diffusion_model')
        
        # define volume boundary condition
        depth_range = deepcopy(meta_data['depth_range'])
        self.patchSize = meta_data['patchSize_px']
        self.kernel_size = math.ceil(self.patchSize * model_cfg.patchRatio)
        self.kernel_size = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1
        self.kernel_uv_size, self.bound_xy = self.calc_kernel_size(meta_data['umtx'], self.kernel_size, meta_data['px_ratio_max'], meta_data['xy_ratio_max'])
        
        # diffusion model
        if model_cfg.w_diffusion and diffusion_model is not None:
            self.diffusion_model = diffusion_model
            self.diffusion_model.requires_grad_(False)
        else:
            self.diffusion_model = None
        
        # define psf volume and mlp
        self.blur_volume = self.create_network(model_cfg, self.bound_xy / 2, depth_range)
        
        # filters for gradient map
        self.gradient_method = model_cfg.gradient_method  # option : sobel, laplacian
        self.laplace = Laplacian(kernel_size=3, border_type='constant')
        self.gaussian = GaussianBlur2d(kernel_size=(7, 7), sigma=(2.0, 2.0), border_type='constant')
        
        # register parameters
        self.register_parameter('min_depth', Parameter(torch.tensor(depth_range[0]), requires_grad=False))
        self.register_parameter('max_depth', Parameter(torch.tensor(depth_range[1]), requires_grad=False))
        
        # log metric and loss
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
    def create_network(self, model_cfg, bound_xy, depth_range):
        architecture = nn.ModuleDict()
        
        # feature network
        featnet = model_cfg.psf_volume.psf_feat(bound_xy=bound_xy, min_depth=depth_range[0], max_depth=depth_range[1])
        architecture['featnet'] = featnet
        
        # embedding and mlp network
        if 'embedder' in model_cfg.psf_volume:
            embedder = model_cfg.psf_volume.embedder(input_dims=model_cfg.psf_volume.psf_mlp.keywords['in_dim'])
            mlpnet = model_cfg.psf_volume.psf_mlp(in_dim=embedder.out_dim + featnet.out_dim)
            architecture['mlpnet'] = mlpnet
            architecture['embedder'] = embedder
        else:
            mlpnet = model_cfg.psf_volume.psf_mlp(in_dim=model_cfg.psf_volume.psf_mlp.keywords['in_dim'] + featnet.out_dim)
            architecture['mlpnet'] = mlpnet
        
        return architecture
        
    def calc_kernel_size(self, K_mat, kernel_size, px_ratio_max, xy_ratio_max):
        '''
        Args:
            Kmat: camera matrix
            kernel_size : kernel size within patch of template
            px_ratio_max: max(image_px / template_px) >> use for kernel_uvsize
            xy_ratio_max: max((image_px / template_px) * depth) >> use for bound_xy
        Returns:
            For PSF Volume, we should define bound_xy size,
            which is the bound_xy of farthest patch (max_depth).
            For this patch, the following equation should be hold.
            K'[u, v, 1] = K[u + du, v + dv, 1], du, dv is the uv coord of patch[0, 0]
            farthest patch coords : [0 ~ kernel_uvsize, 0 ~ kernel_uvsize, max_depth]
            bound_xy = max(kernel_uvsize / fx, kernel_uvsize / fy) * depth
            kernel_uvsize = kernel_size * (image_px / template_px)
            bound_xy = kernel_size * ((image_px / template_px) * depth) / min(fx, fy)
            >> max(bound_xy) = (kernel_size / min(fx, fy)) * xy_ratio_max
            >> max(kernel_uvsize) = kernel_size * px_ratio_max
        '''
        fx, fy = K_mat[0, 0], K_mat[1, 1]
        bound_xy = kernel_size * (xy_ratio_max / min(fx, fy))
        kernel_uvsize = math.ceil(kernel_size * px_ratio_max)
        kernel_uvsize = kernel_uvsize if kernel_uvsize % 2 == 1 else kernel_uvsize + 1
        return kernel_uvsize, bound_xy
    
    @torch.no_grad()
    def gradient_apply(self, feat, clean, mask):
        assert self.gradient_method in ['sobel', 'laplacian', 'none'], 'Invalid gradient method'
        # compute gradient map
        if self.gradient_method == 'laplacian':
            gradient = self.laplace(feat)
        elif self.gradient_method == 'sobel':
            gradient = torch.sqrt(torch.square(spatial_gradient(feat)).sum(dim=2))
        else:
            gradient = feat
        
        # fill the centers (fill hole)
        mask_rgb = repeat(mask, 'b 1 h w -> b c h w', c=3) > 0
        medianvalue = torch.median(gradient[(clean > 0) & mask_rgb]).item()
        holevalue = medianvalue * torch.ones_like(gradient)
        gradient = torch.where((clean > 0) & mask_rgb, holevalue, gradient)
        gradient = normalize(self.gaussian(gradient), mask_rgb)
        
        return gradient
    
    def forward_psf_volume(self, pts3d, uv_coord, mask, network):
        kernel_channel = network['mlpnet'].out_dim
        batch, n, kh, kw, c = pts3d.shape
        masked_pts3d, masked_uv_coord = pts3d[mask].reshape(-1, 3), uv_coord[mask].reshape(-1, 2)
        feat = network['featnet'](masked_pts3d)
        norm_coords = torch.cat([masked_uv_coord, masked_pts3d], dim=1)
        norm_coords[..., 2:-1] = (norm_coords[..., 2:-1] / (self.bound_xy / 2))
        norm_coords[..., -1] = (norm_coords[..., -1] - self.min_depth) / (self.max_depth - self.min_depth) * 2. - 1.
        
        if 'embedder' in network:
            norm_coords = network['embedder'](norm_coords)
            
        kernel_w = network['mlpnet'](torch.cat([norm_coords, feat], dim=1))
        indices = torch.arange(batch * n * kh * kw, device=pts3d.device).view(batch, n, kh, kw)
        kernel_w = segment_coo(src=kernel_w, 
                               index=indices[mask].reshape(-1), 
                               out=torch.zeros((batch * n * kh * kw, kernel_channel)).type_as(kernel_w), 
                               reduce='sum')
        kernel_w = rearrange(kernel_w, '(b n kh kw) c -> b n kh kw c', b=batch, n=n, kh=kh, kw=kw)
        
        return kernel_w[..., :kernel_channel // 2], kernel_w[..., kernel_channel // 2:]
        
    def model_step(self, batch):
        loss, output = dict(), dict()
        
        # preprocessing
        clean_img = batch['clean']
        gradient_left = self.gradient_apply(batch['left'], clean_img, batch['mask'])
        gradient_right = self.gradient_apply(batch['right'], clean_img, batch['mask'])
        
        # create valid mask
        with torch.no_grad():
            ones_kernel = torch.ones((clean_img.shape[1], clean_img.shape[1], self.kernel_size, self.kernel_size), device=clean_img.device)
            mask = F.conv2d(clean_img, weight=ones_kernel, stride=1, padding=self.kernel_size // 2, dilation=1) > 0
            mask_gradient = rearrange((mask.type_as(clean_img) * batch['mask']).sum(dim=1), 'b h w -> b (h w)')
            gradient_left = gradient_left * batch['weight']
            gradient_right = gradient_right * batch['weight']
            output.update({'mask_gradient': rearrange(mask_gradient > 0, 'b (h w) -> b h w', h=self.patchSize, w=self.patchSize)})
            
            # apply diffusion if enabled
            if self.diffusion_model is not None:
                resized_left = F.interpolate(batch['left'], size=self.diffusion_model.image_size, mode='bilinear', align_corners=True)
                resized_right = F.interpolate(batch['right'], size=self.diffusion_model.image_size, mode='bilinear', align_corners=True)
                left_diff = self.diffusion_model.model_predictions(resized_left, torch.tensor([10]).to(resized_left.device))
                right_diff = self.diffusion_model.model_predictions(resized_right, torch.tensor([10]).to(resized_right.device))
                blur_left_img = F.interpolate(left_diff.pred_x_start, size=self.patchSize, mode='bilinear', align_corners=True)
                blur_right_img = F.interpolate(right_diff.pred_x_start, size=self.patchSize, mode='bilinear', align_corners=True)
            else:
                blur_left_img = batch['left']
                blur_right_img = batch['right']
            
        # normalize input
        if self.hparams.model_cfg.normalize_input:
            mask_rgb = repeat(batch['mask'], 'b 1 h w -> b c h w', c=3) > 0
            clean_img = normalize(clean_img, mask_rgb)  # normalize
            blur_left_img = normalize(blur_left_img, mask_rgb)  # normalize
            blur_right_img = normalize(blur_right_img, mask_rgb)  # normalize
        
        " learn to blur "
        # unfolding
        uvcoord_unfold = unfolding(batch['uv_coord'], kernel_size=self.kernel_size)  # already distorted uv coordinates
        pts3d_unfold = unfolding(batch['3d_coord'], kernel_size=self.kernel_size)  # 3d scene points
        mask_unfold = unfolding(rearrange(mask_gradient, 'b (h w) -> b 1 h w', h=self.patchSize, w=self.patchSize), kernel_size=self.kernel_size)  # mask
        mask_unfold = mask_unfold[..., 0] > 0
        
        # normalized uv coord
        uvcoord_unfold[..., 0] = (uvcoord_unfold[..., 0] - self.hparams.meta_data.mtx[0, 2]) / self.hparams.meta_data.mtx[0, 0]
        uvcoord_unfold[..., 1] = (uvcoord_unfold[..., 1] - self.hparams.meta_data.mtx[1, 2]) / self.hparams.meta_data.mtx[1, 1]
        
        # kernel sampling from psf volume
        pts3d_unfold[..., :2] = pts3d_unfold[..., :2] - pts3d_unfold[:, :, self.kernel_size // 2:self.kernel_size // 2 + 1, self.kernel_size // 2:self.kernel_size // 2 + 1, :2]
        kernel_left, kernel_right = self.forward_psf_volume(pts3d_unfold, uvcoord_unfold, mask_unfold, self.blur_volume)
        
        # reblurred image
        kernel_left = rearrange(kernel_left, 'b (h w) kh kw c -> b c kh kw h w', h=self.patchSize, w=self.patchSize)
        kernel_right = rearrange(kernel_right, 'b (h w) kh kw c -> b c kh kw h w', h=self.patchSize, w=self.patchSize)
        reblurred_left = conv2d(clean_img, kernel_left, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2, dilation=1)
        reblurred_right = conv2d(clean_img, kernel_right, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2, dilation=1)
        
        # loss terms (recon loss, SSIM loss, L1 regularize loss, tv loss)
        loss_recon_left = torch.mean(gradient_left * charbonnier_loss(reblurred_left, blur_left_img, reduction='none'))
        loss_recon_right = torch.mean(gradient_right * charbonnier_loss(reblurred_right, blur_right_img, reduction='none'))
        loss.update({'loss_recon': loss_recon_left + loss_recon_right})
        loss_ssim_left = torch.mean(gradient_left * ssim_loss(reblurred_left, blur_left_img, window_size=11, reduction='none'))
        loss_ssim_right = torch.mean(gradient_right * ssim_loss(reblurred_right, blur_right_img, window_size=11, reduction='none'))
        loss.update({'loss_ssim': loss_ssim_left + loss_ssim_right})
        loss_regl1_left = torch.mean(torch.norm(rearrange(kernel_left, 'b c kh kw h w -> b c (kh kw) h w'), p=1, dim=2))
        loss_regl1_right = torch.mean(torch.norm(rearrange(kernel_right, 'b c kh kw h w -> b c (kh kw) h w'), p=1, dim=2))
        loss.update({'loss_regl1': loss_regl1_left + loss_regl1_right})
        loss_tv_left_img = torch.mean(total_variation(reblurred_left, reduction='mean'))
        loss_tv_right_img = torch.mean(total_variation(reblurred_right, reduction='mean'))
        # loss_tv_left = torch.mean(total_variation(total_variation(rearrange(kernel_left, 'b c kh kw h w -> b c h w kh kw'), reduction='mean'), reduction='sum'))
        # loss_tv_right = torch.mean(total_variation(total_variation(rearrange(kernel_right, 'b c kh kw h w -> b c h w kh kw'), reduction='mean'), reduction='sum'))
        # loss.update({'loss_tv': loss_tv_left + loss_tv_right + loss_tv_left_img + loss_tv_right_img})
        loss.update({'loss_tv': loss_tv_left_img + loss_tv_right_img})
        
        # Volumetric symmetric loss at center of image
        if self.hparams.model_cfg.use_symmetric_loss:
            
            depths = torch.linspace(self.min_depth, self.max_depth, self.hparams.model_cfg.level).to(clean_img.device)
            xy_coords = torch.linspace(-1/2*self.bound_xy, 1/2*self.bound_xy, self.kernel_uv_size).to(clean_img.device)
            xyz_coords = torch.stack(torch.meshgrid(xy_coords, xy_coords, depths, indexing='xy'), dim=-1)  # [kernel_size_uv, kernel_size_uv, level, 3]
            xyz_coords = rearrange(xyz_coords, 'h w l c -> (h w l) c')
            feat_dp = self.blur_volume['featnet'](xyz_coords)
            norm_coords = torch.cat([torch.zeros_like(xyz_coords[..., :2]), xyz_coords], dim=-1)
            norm_coords[..., 2:-1] = (norm_coords[..., 2:-1] / (self.bound_xy / 2))
            norm_coords[..., -1] = (norm_coords[..., -1] - self.min_depth) / (self.max_depth - self.min_depth) * 2. - 1.
    
            if 'embedder' in self.blur_volume:
                norm_coords = self.blur_volume['embedder'](norm_coords)
                
            kernel_w = self.blur_volume['mlpnet'](torch.cat([norm_coords, feat_dp], dim=1))
            kernel_w = rearrange(kernel_w, '(h w l) c -> h w l c', h=self.kernel_uv_size, w=self.kernel_uv_size)
                
            kernel_w_left = kernel_w[..., :self.blur_volume['mlpnet'].out_dim // 2]
            kernel_w_right = kernel_w[..., self.blur_volume['mlpnet'].out_dim // 2:]
            
            # symmetric loss
            loss.update({'loss_sym': charbonnier_loss(kernel_w_left, torch.flip(kernel_w_right, [1]), reduction='sum')})
        output.update({'left': reblurred_left, 'right': reblurred_right, 'left_gt': blur_left_img, 'right_gt': blur_right_img, 'clean': clean_img})
                
        return loss, output
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
    
    def training_step(self, batch, batch_idx: int):
        loss, preds = self.model_step(batch)
        
        loss_total = 0.0
        for key in loss:
            for key_weight in self.hparams.model_cfg.loss_weight:
                if key_weight in key:
                    loss_total += loss[key] * self.hparams.model_cfg.loss_weight[key_weight]
                    self.log(f'train/{key}', loss[key], on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss(loss_total)
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
                    
        return loss_total
    
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass
    
    def validation_step(self, batch, batch_idx: int):
        loss, preds = self.model_step(batch)
        
        loss_total = 0.0
        for key in loss:
            for key_weight in self.hparams.model_cfg.loss_weight:
                if key_weight in key:
                    loss_total += loss[key] * self.hparams.model_cfg.loss_weight[key_weight]
                    self.log(f'val/{key}', loss[key], on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss(loss_total)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # save result
        if batch_idx in [351, 363, 370, 375, 406, 435, 485, 547]:
            clean = batch['clean'][0].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
            mask_gradient = repeat(preds['mask_gradient'][0].cpu().detach().numpy().clip(0, 1), 'h w -> h w 3')
            left_pred = preds['left'][0].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
            right_pred = preds['right'][0].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
            left_gt = preds['left_gt'][0].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
            right_gt = preds['right_gt'][0].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
            pts_plots = np.hstack([clean, mask_gradient, left_pred, left_gt, right_pred, right_gt])
            self.logger.log_image(key=f'Source | Mask | left_pred | left_gt | right_pred | right_gt [sample {batch_idx}]', images=[pts_plots])
    
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass
    
    def test_step(self, batch, batch_idx: int):
        pass
    
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        net_params, encoding_params = [], []
        for (name, param) in self.trainer.model.named_parameters():
            if 'diffusion_model' in name:
                continue
            elif 'volume' in name:
                if 'encoder' in name:
                    encoding_params.append(param)
                else:
                    net_params.append(param)
        params = [{'params': encoding_params, 'lr': self.hparams.optimizer.keywords['lr'] * 5.},
                  {'params': net_params, 'lr': self.hparams.optimizer.keywords['lr']}]
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
        
        
        