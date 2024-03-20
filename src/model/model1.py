from typing import Any, Dict, Tuple

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
        
        if not 'w_diffusion' in model_cfg or diffusion_model is None: model_cfg.w_diffusion = False
        self.save_hyperparameters(logger=False, ignore='diffusion_model')
        
        # define volume boundary condition
        self.bound_xy, self.bound_xy_max = self.calc_psfVbound(meta_data['umtx'], model_cfg.psf_volume.patch_uvsize, meta_data['depth_range'], meta_data['focal_distance'])
        
        # diffusion model
        if model_cfg.w_diffusion:
            self.diffusion_model = diffusion_model
            self.diffusion_model.requires_grad_(False)
        
        # define psf volume and mlp
        self.blur_volume = self.create_network(model_cfg, self.bound_xy / 2, meta_data['depth_range'])
        if model_cfg.use_deblur_volume:
            self.deblur_volume = self.create_network(model_cfg, self.bound_xy / 2, meta_data['depth_range'])
        else:
            self.deblur_volume = None
        
        # filters for gradient map
        self.gradient_method = 'sobel'  # option : sobel, laplacian
        self.laplace = Laplacian(kernel_size=3, border_type='constant')
        self.gaussian = GaussianBlur2d(kernel_size=(7, 7), sigma=(2.0, 2.0), border_type='constant')
        
        # register parameters
        self.register_parameter('min_depth', Parameter(torch.tensor(meta_data['depth_range'][0]), requires_grad=False))
        self.register_parameter('max_depth', Parameter(torch.tensor(meta_data['depth_range'][1]), requires_grad=False))
        
        # log metric and loss
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
    def create_network(self, model_cfg, bound_xy, depth_range):
        
        # feature network
        featnet = model_cfg.psf_volume.psf_feat(bound_xy=bound_xy, min_depth=depth_range[0], max_depth=depth_range[1], num_levels=model_cfg.level)
        
        # embedding and mlp network
        if 'embedder' in model_cfg.psf_volume:
            embedder = model_cfg.psf_volume.embedder(input_dims=model_cfg.psf_volume.psf_mlp.keywords['in_dim'])
            mlpnet = model_cfg.psf_volume.psf_mlp(in_dim=embedder.out_dim + featnet.out_dim)
            return nn.ModuleDict({'featnet': featnet, 'mlpnet': mlpnet, 'embedder': embedder})
        else:
            mlpnet = model_cfg.psf_volume.psf_mlp(in_dim=model_cfg.psf_volume.psf_mlp.keywords['in_dim'] + featnet.out_dim)
            return nn.ModuleDict({'featnet': featnet, 'mlpnet': mlpnet})
        
    def calc_psfVbound(self, K_mat, patch_uvsize, depth_range, focal_distance):
        '''
            For PSF Volume, we should define max_xy size,
            which is the max_xy of farthest patch (max_depth).
            For this patch, the following equation should be hold.
            K'[u, v, 1] = K[u + du, v + dv, 1], du, dv is the uv coord of patch[0, 0]
            farthest patch coords : [0 ~ patch_uvsize, 0 ~ patch_uvsize, max_depth]
            max_xy = max(patch_uvsize / fx, patch_uvsize / fy) * max_depth
        '''
        fx, fy = K_mat[0, 0], K_mat[1, 1]
        base_xy_size = max(patch_uvsize / fx, patch_uvsize / fy)
        disp_idx = np.argmax([abs(1 / depth_range[0] - 1 / focal_distance), abs(1 / depth_range[1] - 1 / focal_distance)]).item()
        return base_xy_size * depth_range[disp_idx], base_xy_size * depth_range[1]
    
    @torch.no_grad()
    def gradient_apply(self, feat, clean, mask):
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
        
        return gradient, mask_rgb
    
    def forward_psf_volume(self, pts3d, uv_coord, mask, network):
        kernel_channel = network['mlpnet'].out_dim
        batch, n, kh, kw, c = pts3d.shape
        masked_pts3d, masked_uv_coord = pts3d[mask].reshape(-1, 3), uv_coord[mask].reshape(-1, 2)
        feat = network['featnet'](masked_pts3d)
        coords = torch.cat([masked_uv_coord, masked_pts3d], dim=1)
        if 'embedder' in network:
            coords = network['embedder'](coords)
        kernel_w = network['mlpnet'](torch.cat([coords, feat], dim=1))
        indices = torch.arange(batch * n * kh * kw, device=pts3d.device).view(batch, n, kh, kw)
        kernel_w = segment_coo(src=kernel_w, 
                               index=indices[mask].reshape(-1), 
                               out=torch.zeros((batch * n * kh * kw, kernel_channel)).type_as(kernel_w), 
                               reduce='sum')
        kernel_w = rearrange(kernel_w, '(b n kh kw) c -> b n kh kw c', b=batch, n=n, kh=kh, kw=kw)
        return kernel_w[..., :kernel_channel // 2], kernel_w[..., kernel_channel // 2:]
        
    def model_step(self, batch):
        loss, output = dict(), dict()
        for n in range(self.hparams.meta_data.num_level):
            if n != 1:  # for now, only process the middle level because of memory issue
                continue
            
            patchSize = batch[f'clean_{n}'].shape[2]
            kernel_size = int(self.hparams.meta_data.patchRatio[n] * self.bound_xy_max)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            gradient_left, mask_left = self.gradient_apply(batch[f'left_{n}'], batch[f'clean_{n}'], batch[f'mask_{n}'])
            gradient_right, mask_right = self.gradient_apply(batch[f'right_{n}'], batch[f'clean_{n}'], batch[f'mask_{n}'])
            
            # create valid mask
            with torch.no_grad():
                ones_kernel = torch.ones((batch[f'clean_{n}'].shape[1], batch[f'clean_{n}'].shape[1], kernel_size, kernel_size), device=batch[f'clean_{n}'].device)
                mask = F.conv2d(batch[f'clean_{n}'], weight=ones_kernel, stride=1, padding=kernel_size // 2, dilation=1) > 0
                mask_gradient = rearrange((mask.float() * (mask_left.float() + mask_right.float())).mean(dim=1), 'b h w -> b (h w)') > 0
                gradient_left = gradient_left * batch[f'weight_{n}']
                gradient_right = gradient_right * batch[f'weight_{n}']
                output.update({f'mask_gradient_{n}': rearrange(mask_gradient, 'b (h w) -> b h w', h=patchSize, w=patchSize)})
                
            # apply diffusion if enabled
            if self.hparams.model_cfg.w_diffusion:
                # self.diffusion_model.sampling_timesteps
                left_diff = self.diffusion_model.model_predictions(batch[f'left_{n}'], torch.tensor([10]).to(batch[f'left_{n}'].device))
                right_diff = self.diffusion_model.model_predictions(batch[f'right_{n}'], torch.tensor([10]).to(batch[f'right_{n}'].device))
                clean_img = batch[f'clean_{n}']
                blur_left_img = left_diff.pred_x_start
                blur_right_img = right_diff.pred_x_start
            else:
                clean_img = batch[f'clean_{n}']
                blur_left_img = batch[f'left_{n}']
                blur_right_img = batch[f'right_{n}']
                
            # normalize input
            if self.hparams.model_cfg.normalize_input:
                mask_rgb = repeat(batch[f'mask_{n}'], 'b 1 h w -> b c h w', c=3) > 0
                clean_img = normalize(clean_img, mask_rgb)  # normalize
                blur_left_img = normalize(blur_left_img, mask_rgb)  # normalize
                blur_right_img = normalize(blur_right_img, mask_rgb)  # normalize
            
            " learn to blur "
            # unfolding
            uvcoord_unfold = unfolding(batch[f'uv_coord_{n}'], kernel_size=kernel_size)
            pts3d_unfold = unfolding(batch[f'3d_coord_{n}'], kernel_size=kernel_size)
            
            # uv coord normalization
            uvcoord_unfold[..., 0] = (uvcoord_unfold[..., 0] - self.hparams.meta_data.umtx[0, 2]) / self.hparams.meta_data.umtx[0, 0]
            uvcoord_unfold[..., 1] = (uvcoord_unfold[..., 1] - self.hparams.meta_data.umtx[1, 2]) / self.hparams.meta_data.umtx[1, 1]
            
            # kernel sampling from psf volume
            pts3d_unfold[..., :2] = pts3d_unfold[..., :2] - pts3d_unfold[:, :, kernel_size // 2:kernel_size // 2 + 1, kernel_size // 2:kernel_size // 2 + 1, :2]
            kernel_left, kernel_right = self.forward_psf_volume(pts3d_unfold, uvcoord_unfold, mask_gradient, self.blur_volume)
            
            # reblurred image
            kernel_left = rearrange(kernel_left, 'b (h w) kh kw c -> b c kh kw h w', h=patchSize, w=patchSize)
            kernel_right = rearrange(kernel_right, 'b (h w) kh kw c -> b c kh kw h w', h=patchSize, w=patchSize)
            reblurred_left = conv2d(clean_img, kernel_left, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, dilation=1)
            reblurred_right = conv2d(clean_img, kernel_right, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, dilation=1)
            
            # loss terms (recon loss, SSIM loss, L1 regularize loss, tv loss)
            loss_recon_left = torch.mean(gradient_left * charbonnier_loss(reblurred_left, blur_left_img, reduction='none'))
            loss_recon_right = torch.mean(gradient_right * charbonnier_loss(reblurred_right, blur_right_img, reduction='none'))
            loss.update({f'loss_recon_{n}_b': loss_recon_left + loss_recon_right})
            loss_ssim_left = torch.mean(gradient_left * ssim_loss(reblurred_left, blur_left_img, window_size=11, reduction='none'))
            loss_ssim_right = torch.mean(gradient_right * ssim_loss(reblurred_right, blur_right_img, window_size=11, reduction='none'))
            loss.update({f'loss_ssim_{n}_b': loss_ssim_left + loss_ssim_right})
            loss_regl1_left = torch.mean(torch.norm(rearrange(kernel_left, 'b c kh kw h w -> b c (kh kw) h w'), p=1, dim=2))
            loss_regl1_right = torch.mean(torch.norm(rearrange(kernel_right, 'b c kh kw h w -> b c (kh kw) h w'), p=1, dim=2))
            loss.update({f'loss_regl1_{n}_b': loss_regl1_left + loss_regl1_right})
            loss_tv_left_img = torch.mean(total_variation(reblurred_left, reduction='mean'))
            loss_tv_right_img = torch.mean(total_variation(reblurred_right, reduction='mean'))
            # loss_tv_left = torch.mean(total_variation(total_variation(rearrange(kernel_left, 'b c kh kw h w -> b c h w kh kw'), reduction='mean'), reduction='sum'))
            # loss_tv_right = torch.mean(total_variation(total_variation(rearrange(kernel_right, 'b c kh kw h w -> b c h w kh kw'), reduction='mean'), reduction='sum'))
            # loss.update({f'loss_tv_{n}_b': loss_tv_left + loss_tv_right + loss_tv_left_img + loss_tv_right_img})
            loss.update({f'loss_tv_{n}_b': loss_tv_left_img + loss_tv_right_img})
            # loss.update({f'loss_tv_{n}_b': loss_tv_left + loss_tv_right})
            
            # following the ICCP20 paper
            if self.hparams.model_cfg.use_symmetric_loss:
                convolved1 = conv2d(blur_left_img, kernel_right, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, dilation=1)
                convolved2 = conv2d(blur_right_img, kernel_left, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, dilation=1)
                loss.update({f'loss_symmetric_{n}_b': charbonnier_loss(convolved1, convolved2, reduction='mean')})
            output.update({f'left_{n}': reblurred_left, f'right_{n}': reblurred_right, f'left_gt_{n}': blur_left_img, f'right_gt_{n}': blur_right_img, f'clean_{n}': clean_img})
            
            if self.hparams.model_cfg.use_deblur_volume:
                " learn to deblur "
                
                # kernel sampling from psf volume
                kernel_left_deblur, kernel_right_deblur = self.forward_psf_volume(pts3d_unfold, uvcoord_unfold, mask_gradient, self.deblur_volume)
                
                # deblurred image
                kernel_left_deblur = rearrange(kernel_left_deblur, 'b (h w) kh kw c -> b c kh kw h w', h=patchSize, w=patchSize)
                kernel_right_deblur = rearrange(kernel_right_deblur, 'b (h w) kh kw c -> b c kh kw h w', h=patchSize, w=patchSize)
                deblurred_left = conv2d(blur_left_img, kernel_left_deblur, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, dilation=1)
                deblurred_right = conv2d(blur_right_img, kernel_right_deblur, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, dilation=1)
            
                # loss terms (recon loss, SSIM loss, L1 regularize loss)
                loss_recon_left_d = torch.mean(gradient_left * charbonnier_loss(deblurred_left, clean_img, reduction='none'))
                loss_recon_right_d = torch.mean(gradient_right * charbonnier_loss(deblurred_right, clean_img, reduction='none'))
                loss.update({f'loss_recon_{n}_d': loss_recon_left_d + loss_recon_right_d})
                loss_ssim_left_d = torch.mean(gradient_left * ssim_loss(deblurred_left, clean_img, window_size=11, reduction='none'))
                loss_ssim_right_d = torch.mean(gradient_right * ssim_loss(deblurred_right, clean_img, window_size=11, reduction='none'))
                loss.update({f'loss_ssim_{n}_d': loss_ssim_left_d + loss_ssim_right_d})
                loss_regl1_left_d = torch.mean(torch.norm(rearrange(kernel_left_deblur, 'b c kh kw h w -> b c (kh kw) h w'), p=1, dim=2))
                loss_regl1_right_d = torch.mean(torch.norm(rearrange(kernel_right_deblur, 'b c kh kw h w -> b c (kh kw) h w'), p=1, dim=2))
                loss.update({f'loss_regl1_{n}_d': loss_regl1_left_d + loss_regl1_right_d})
                loss_tv_left_img_d = torch.mean(total_variation(deblurred_left, reduction='mean'))
                loss_tv_right_img_d = torch.mean(total_variation(deblurred_right, reduction='mean'))
                loss_tv_left_d = torch.mean(total_variation(total_variation(rearrange(kernel_left_deblur, 'b c kh kw h w -> b c h w kh kw'), reduction='mean'), reduction='mean'))
                loss_tv_right_d = torch.mean(total_variation(total_variation(rearrange(kernel_right_deblur, 'b c kh kw h w -> b c h w kh kw'), reduction='mean'), reduction='mean'))
                loss.update({f'loss_tv_{n}_d': loss_tv_left_d + loss_tv_right_d + loss_tv_left_img_d + loss_tv_right_img_d})
                
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
        self.log('val/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
                    
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
            clean = batch['clean_1'][0].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
            mask_gradient = repeat(preds['mask_gradient_1'][0].cpu().detach().numpy().clip(0, 1), 'h w -> h w 3')
            left_pred = preds['left_1'][0].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
            right_pred = preds['right_1'][0].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
            left_gt = preds['left_gt_1'][0].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
            right_gt = preds['right_gt_1'][0].permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
            pts_plots = np.hstack([clean, mask_gradient, left_pred, left_gt, right_pred, right_gt])
            self.logger.log_image(key=f'Source | Mask | left_blurred | left_gt | right_blurred | right_gt_{batch_idx}', images=[pts_plots])
    
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
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
        
        
        