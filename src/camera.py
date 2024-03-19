

import cv2
import torch
import pickle
import hydra
import numpy as np
from tqdm import tqdm
from pathlib import Path
from einops import repeat, rearrange
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import lightning as L
from src.utils.visualizer import visualize_PSFVolume, save_as_gif
from src.model.utils.common import masked_mean, masked_minmax
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
# from scipy.signal import convolve2d
import torch.nn.functional as F
from torchvision.utils import save_image


# class to store camera related parameters
class CameraObject(object):
    
    def __init__(self, opts, load_path=None):
        
        self.opts = opts
        self.tag = int(10 * self.opts.calib.psfboard.target_focal)
        
        # camera parameters
        self.calib_data = self.loadCalibdata(load_path)
        
        # diffusion model for denoising patches
        if self.opts.calib.psfboard.use_diffusion_denoising:
            unet = Unet(
                dim = 64,
                dim_mults = (1, 2, 4, 8),
                flash_attn = True).cuda()
            self.diffusion = GaussianDiffusion(
                unet,
                image_size = 128,
                timesteps = 1000,           # number of steps
                # sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
                sampling_timesteps = 150    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
            ).cuda().eval()
            self.diffusion_ckpt_path = Path(load_path).parent / 'psfcalib' / 'diffusion_ckpt.pt'
    
    # load calibration data with learned PSF model (optional)
    def loadCalibdata(self, model_path=None):
        calib_data = {}
        if model_path is not None:
            model_path = Path(model_path).parent
            if (model_path / 'intcalib').is_dir():
                if (model_path / 'intcalib' / 'calib_data.pkl').is_file():
                    calib_data.update({'camera': pickle.load(open(model_path / 'intcalib' / 'calib_data.pkl', 'rb'))})
            if (model_path / 'psfcalib').is_dir():
                if (model_path / 'psfcalib' / f'training_data_{self.tag}.pkl').is_file():
                    calib_data.update({'training_data': pickle.load(open(model_path / 'psfcalib' / f'training_data_{self.tag}.pkl', 'rb'))})
        return calib_data
        
    # run intrinsic calibration with all-in-focus images
    def runIntrinsicCalib(self, board, observations: dict):
        
        # step1: detect boards
        observations = board.detectCalibBoard(observations)
        
        # step2: run initial intrinsic calibration
        calib_results, per_scene_results = board.runSingleCalib(observations)
        print('Reprojection error: ', sum(per_scene_results['ret']) / len(per_scene_results['ret']))
        
        # step3: propagate corners with view rejection
        observations = board.propagateCorners(observations, calib_results, per_scene_results)
        
        # step4: run intrinsic calibration with refined corners
        calib_results, per_scene_results = board.runSingleCalib(observations)
        print('Reprojection error: ', sum(per_scene_results['ret']) / len(per_scene_results['ret']))
        
        # step5: save calibration results
        self.calib_data['camera'] = calib_results
        pickle.dump(calib_results, open(Path(self.opts.paths.output_dir) / f'calib_data.pkl', 'wb'))
    
    # prepare data from observations for PSF calibration
    def preparePatches(self, board, observations: dict):
        
        training_data = {}
        
        # step1: detect boards and apply PnP to get location of boards
        observations = board.detectPSFBoard(observations, self.calib_data['camera'])
        
        # step2: align observed boards and extract patches
        training_data.update(board.extractPatches(observations, self.calib_data['camera']))
        
        # step3: extract meta data from patches
        training_data.update(self.estimateMetadata(board, training_data))
        
        # step4: save training data
        self.calib_data['training_data'] = training_data
        pickle.dump(training_data, open(Path(self.opts.paths.output_dir) / f'training_data_{self.tag}.pkl', 'wb'))
    
    # get meta data from observations
    def estimateMetadata(self, board, training_data: dict):
        fnumber = board.get_numerical_fnumber(self.opts.calib.psfboard.target_focal)
        focal_mm = self.opts.calib.observation.focal_mm
        aperture = max(self.calib_data['camera']['mtx'][0, 0], self.calib_data['camera']['mtx'][1, 1]) / fnumber
        
        # get depth range
        mask_start = (training_data['patch_uv'][0][..., 0] >= 0) & (training_data['patch_uv'][0][..., 1] >= 0)
        mask_end = (training_data['patch_uv'][0][..., 0] <= self.calib_data['camera']['image_size'][1] - 1) & (training_data['patch_uv'][0][..., 1] <= self.calib_data['camera']['image_size'][0] - 1)
        mask = torch.tensor(mask_start & mask_end & (training_data['patch_3d'][0][..., -1] > 0))
        depth = torch.tensor(training_data['patch_3d'][0][..., -1])
        depth_mean = masked_mean(depth.reshape(-1), mask.reshape(-1), dim=0)
        depth_min, depth_max = masked_minmax(depth.reshape(-1), mask.reshape(-1), dim=0)
        
        return {'focal_mm': focal_mm, 'aperture': aperture, 'fnumber': fnumber, 'depth_mean': depth_mean.item(), 'depth_min': depth_min.item(), 'depth_max': depth_max.item()}
    
    # train PSF calibration model
    def trainPSFCalibModel(self):
        
        # seed everything
        L.seed_everything(self.opts.seed, workers=True)
        
        # learn diffusion model
        use_diffusion_denoising = self.opts.calib.psfboard.use_diffusion_denoising
        if use_diffusion_denoising:
            if not self.diffusion_ckpt_path.is_file():
                self.trainDiffusionModel()
            else:
                checkpoint = torch.load(str(self.diffusion_ckpt_path))
                self.diffusion.load_state_dict(checkpoint['model'], strict=False)
        
        # learn psf model
        datamodule = hydra.utils.instantiate(self.opts.get("data"), 
                                             model_cfg=self.opts.model.model_cfg, 
                                             calib_data=self.calib_data, 
                                             focal_distance=self.opts.calib.psfboard.focal_distance_mm,
                                             precision=str(self.opts.trainer.precision))
        diffusion_model = self.diffusion if use_diffusion_denoising else None
        
        # state model
        final_ckpt_path = Path(self.opts.callbacks.model_checkpoint.dirpath) / f'epoch_{(self.opts.trainer.max_epochs - 1):03d}.ckpt'
        if final_ckpt_path.is_file():
            import importlib
            module_path, class_name = self.opts.model._target_.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model = getattr(module, class_name).load_from_checkpoint(str(final_ckpt_path), strict=False)
        else:
            model = hydra.utils.instantiate(self.opts.model, 
                                            meta_data=datamodule.meta_data, 
                                            output_dir=self.opts.paths.output_dir, 
                                            diffusion_model=diffusion_model)
            callbacks = instantiate_callbacks(self.opts.get("callbacks"))
            logger = instantiate_loggers(self.opts.get("logger"))
            trainer = hydra.utils.instantiate(self.opts.trainer, callbacks=callbacks, logger=logger)
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=self.opts.get("load_ckpt"))
            
        # save PSF volume related results
        self.model = model
        # self.savePSFImage(model)
        # self.savePSFVolume(model)
    
    def trainDiffusionModel(self):
        
        self.diffusion.model.training = True
        
        batch_imgs = self.calib_data['training_data']['clean_patch'][1]
        batch_imgs = torch.Tensor(batch_imgs).permute(0, 3, 1, 2)
        batch_imgs = torch.split(batch_imgs, self.opts.calib.psfboard.batchsize)
        optim = torch.optim.Adam(self.diffusion.parameters(), lr=1e-4, betas=(0.9, 0.99))
        loss_diffusion = 0.0
        pbar = tqdm(range(self.opts.calib.psfboard.diffusion_epoch), desc=("loss=%.2f" % loss_diffusion))
        for epoch in pbar:
            for batch in batch_imgs:
                losses = self.diffusion(batch.cuda() / 255.0)
                losses.backward()
                optim.step()
                optim.zero_grad()
            loss_diffusion = float(losses.detach().cpu().numpy())
            pbar.set_description("loss=%.2f" % loss_diffusion)

            data = {
                'step': epoch,
                'model': self.diffusion.state_dict(),
                'opt': optim.state_dict(),
                'version': 'v0.1'
            }
        torch.save(data, str(self.diffusion_ckpt_path))
        
        self.diffusion.model.training = False

    @torch.no_grad()
    def savePSFImage(self, psf_model, level=64, splits=[8, 10], pad=60):
        psf_model.eval()
        
        h_img, w_img = self.calib_data['camera']['image_size']
        patch_size = psf_model.hparams.model_cfg.psf_volume.patch_uvsize
        min_depth, max_depth = psf_model.min_depth.item() + pad, psf_model.max_depth.item() - pad
        depths = torch.linspace(min_depth, max_depth, level).cuda()
        umtx = torch.from_numpy(np.float32(self.calib_data['camera']['umtx'])).cuda()
        
        result_img_left = []
        result_img_right = []
        uv_coords = torch.stack(torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size), indexing='xy'), dim=-1).cuda()
        for n_level in range(level):
            result_psf_rowsl, result_psf_rowsr = [], []
            for n in range(splits[0]):
                result_psf_colsl, result_psf_colsr = [], []
                for m in range(splits[1]):
                    u_coord = uv_coords[..., 0] + (w_img / splits[1] * (m + 1))
                    v_coord = uv_coords[..., 1] + (h_img / splits[0] * (n + 1))
                    uv_coord_patch = torch.stack([u_coord, v_coord], dim=-1)
                    
                    # uv coord normalization
                    uv_coord_patch[..., 0] = (uv_coord_patch[..., 0] - umtx[0, 2]) / umtx[0, 0]
                    uv_coord_patch[..., 1] = (uv_coord_patch[..., 1] - umtx[1, 2]) / umtx[1, 1]
                    
                    # xyz coord
                    xyz_coord_patch = torch.ones((patch_size, patch_size, 3), dtype=torch.float32).cuda()
                    xyz_coord_patch[..., 0] = u_coord
                    xyz_coord_patch[..., 1] = v_coord
                    xyz_coord_patch = torch.einsum('ij,klj->kli', torch.inverse(umtx), xyz_coord_patch) * depths[n_level]
                    xyz_coord_patch[..., :2] = xyz_coord_patch[..., :2] - xyz_coord_patch[patch_size // 2:patch_size // 2 + 1, patch_size // 2:patch_size // 2 + 1, :2]
                    
                    # get psf volume
                    feat = psf_model.blur_volume['featnet'].cuda()(xyz_coord_patch.reshape(-1, 3))
                    coords = torch.cat([uv_coord_patch.reshape(-1, 2), xyz_coord_patch.reshape(-1, 3)], dim=1)
                    if 'embedder' in psf_model.blur_volume:
                        coords = psf_model.blur_volume['embedder'].cuda()(coords)
                    kernel_w = psf_model.blur_volume['mlpnet'].cuda()(torch.cat([coords, feat], dim=1))
                    kernel_w = kernel_w.reshape(patch_size, patch_size, -1)
                    psfk_left, psfk_right = kernel_w[..., :3].cpu().numpy(), kernel_w[..., 3:].cpu().numpy()
                    result_psf_colsl.append((psfk_left - psfk_left.min()) / (psfk_left.max() - psfk_left.min()) * 255)
                    result_psf_colsr.append((psfk_right - psfk_right.min()) / (psfk_right.max() - psfk_right.min()) * 255)
                    
                result_psf_rowsl.append(np.hstack(result_psf_colsl))
                result_psf_rowsr.append(np.hstack(result_psf_colsr))
            result_psf_rowsl = np.vstack(result_psf_rowsl)
            result_psf_rowsr = np.vstack(result_psf_rowsr)
            result_img_left.append(result_psf_rowsl.astype('uint8'))
            result_img_right.append(result_psf_rowsr.astype('uint8'))
            
        # make gif video
        save_as_gif(result_img_left, Path(self.opts.paths.output_dir) / f'psfvolume_left.gif', duration=250)
        save_as_gif(result_img_right, Path(self.opts.paths.output_dir) / f'psfvolume_right.gif', duration=250)


    # def myfun(xe, patlg, patrg, kersize, border):
        
    #     # only flipped kernel symmetry cost
    #     l = convolve2d(patlg, np.fliplr(h), mode='same')
    #     r = convolve2d(patrg, h, mode='same')
    #     l = l[border:-border, border:-border]
    #     r = r[border:-border, border:-border]
    #     err = (l - r) / 255
    #     xerr = np.sum(err**2) / np.size(err)
    #     return xerr
    
    @torch.no_grad()
    def estimateDepthFromPSF(self, imglg, imgrg, imgcg, level=255, splits=[8, 10], pad=60):
        
        psf_model = self.model
        psf_model.eval()
        h_img, w_img = imglg.shape[:2]
        img_patch_size = 111
        kernel_size = 41
        stride = 33
        border = 25
        # h_img, w_img = self.calib_data['camera']['image_size']
        umtx = torch.from_numpy(np.float32(self.calib_data['camera']['umtx'])).cuda()

        img_l_patches = [imglg[y:y+img_patch_size,x:x+img_patch_size] 
                         for y in range(0,imglg.shape[0]-img_patch_size,stride) for x in range(0,imglg.shape[1]-img_patch_size,stride)]
        img_r_patches = [imgrg[y:y+img_patch_size,x:x+img_patch_size] 
                         for y in range(0,imgrg.shape[0]-img_patch_size,stride) for x in range(0,imgrg.shape[1]-img_patch_size,stride)]
        uv_coords = [torch.stack(torch.meshgrid(torch.arange(y, y+img_patch_size), torch.arange(x, x+img_patch_size)), dim=-1).cuda()
                     for y in range(0,imglg.shape[0]-img_patch_size,stride) for x in range(0,imglg.shape[1]-img_patch_size,stride)]
        
        uv_coords = torch.stack(uv_coords)

        uv_coords_norm = uv_coords.clone().cuda().to(torch.float32)
        uv_coords_norm[..., 0] = (uv_coords_norm[..., 0] - umtx[0, 2]) / umtx[0, 0]
        uv_coords_norm[..., 1] = (uv_coords_norm[..., 1] - umtx[1, 2]) / umtx[1, 1]


        calib_patch_size = self.model.hparams.model_cfg.psf_volume.patch_uvsize
        patch_size = img_patch_size
        min_depth, max_depth = self.model.min_depth.item() + pad, self.model.max_depth.item() - pad
        depths = torch.linspace(min_depth, max_depth, level).cuda()
        

        depth_level = torch.zeros([h_img, w_img], dtype=torch.long)
        depth_cost = torch.zeros([h_img, w_img])

        for n in tqdm(range(uv_coords_norm.shape[0])):
            
            xyz_coord_patch = torch.ones(( patch_size, patch_size, 3), dtype=torch.float32).cuda()
            xyz_coord_patch[..., 0] = uv_coords_norm[n, ...,0]
            xyz_coord_patch[..., 1] = uv_coords_norm[n, ...,1]
            xyz_coord_patch = torch.einsum('ij,klj->kli', torch.inverse(umtx), xyz_coord_patch)
            xyz_coord_patch = torch.einsum('ijk,d->dijk', xyz_coord_patch, depths)
            xyz_coord_patch[..., :2] = xyz_coord_patch[..., :2] - xyz_coord_patch[:,patch_size // 2:patch_size // 2 + 1, patch_size // 2:patch_size // 2 + 1, :2]

            # get psf volume
            feat = self.model.blur_volume['featnet'].cuda()(xyz_coord_patch.reshape(-1, 3))
            coords = torch.cat([uv_coords_norm[n].unsqueeze(0).repeat(255,1,1,1).reshape(-1, 2), xyz_coord_patch.reshape(-1, 3)], dim=1)
            if 'embedder' in self.model.blur_volume:
                coords = self.model.blur_volume['embedder'].cuda()(coords)
            kernel_w = self.model.blur_volume['mlpnet'].cuda()(torch.cat([coords, feat], dim=1))
            kernel_w = kernel_w.reshape(level, patch_size, patch_size, -1)
            # psfk_left, psfk_right = kernel_w[..., :3].cpu().numpy(), kernel_w[..., 3:].cpu().numpy()
            psfk_left, psfk_right = kernel_w[..., :3], kernel_w[..., 3:]
            # torch conv2d is actually cross-correlation, need to flip kernel to correctly perform conv
            psfk_left = torch.flip(psfk_left, [1, 2])
            psfk_right = torch.flip(psfk_right, [1, 2])


            cost_per_level = torch.zeros(level,1)
            l = F.conv2d(torch.Tensor(img_l_patches[n])[None,...].permute(0, 3, 1, 2).float().cuda(), 
                            psfk_left.permute(0, 3, 1, 2),
                            padding='same').squeeze()

            r = F.conv2d(torch.Tensor(img_r_patches[n])[None,...].permute(0, 3, 1, 2).float().cuda(), 
                            psfk_right.permute(0, 3, 1, 2),
                            padding='same').squeeze()
                # l[...,rgb] = convolve2d(img_l_patches[n][...,rgb], psfk_left[n_level,...,rgb], mode='same')
                # r[...,rgb] = convolve2d(img_r_patches[n][...,rgb], psfk_right[n_level,...,rgb], mode='same')
            l = l[:,border:-border, border:-border]
            r = r[:,border:-border, border:-border]
            err = (l - r) / 255
            xerr = torch.sum(err**2,dim=(1,2)) / torch.numel(err[0])
            cost_per_level[:,0] = xerr
            fill_coords = uv_coords[n,
                                    patch_size//2 - stride//2:patch_size//2+stride//2 + 1, 
                                    patch_size//2 - stride//2:patch_size//2+stride//2 + 1].reshape(-1, 2).permute(1,0).cpu()
            fval, min_level = torch.min(cost_per_level,0)
            depth_cost[fill_coords[0,:],fill_coords[1,:]] = fval
            depth_level[fill_coords[0,:],fill_coords[1,:]] = min_level

        print('done')    
        # make gif video
        save_image(depth_cost/depth_cost.max(), 'output_cost.png')
        save_image(depth_level/255, 'output_depth.png')
        
    
    @torch.no_grad()
    def savePSFVolume(self, psf_model, level=32, pad=60):
        psf_model.eval()
        
        patch_size = psf_model.hparams.model_cfg.psf_volume.patch_uvsize
        min_depth, max_depth = psf_model.min_depth.item() + pad, psf_model.max_depth.item() - pad
        depths = torch.linspace(min_depth, max_depth, level).cuda()
        
        # create uv coords
        uv_coords = torch.stack(torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size), indexing='xy'), dim=-1).cuda()
        uv_coords[..., 0] = (uv_coords[..., 0] - 0.5 * patch_size) / self.calib_data['camera']['umtx'][0, 0]
        uv_coords[..., 1] = (uv_coords[..., 1] - 0.5 * patch_size) / self.calib_data['camera']['umtx'][1, 1]
        uv_coords = repeat(uv_coords, 'x y c -> x y l c', l=level)
        
        xy_coords = torch.linspace(-1/2*psf_model.bound_xy, 1/2*psf_model.bound_xy, patch_size).cuda()
        xyz_coords = torch.stack(torch.meshgrid(xy_coords, xy_coords, depths, indexing='xy'), dim=-1)  # [patch_size, patch_size, level, 3]
        
        # get psf volume
        feat = psf_model.blur_volume['featnet'].cuda()(xyz_coords.reshape(-1, 3))
        coords = torch.cat([uv_coords.reshape(-1, 2), xyz_coords.reshape(-1, 3)], dim=1)
        if 'embedder' in psf_model.blur_volume:
            coords = psf_model.blur_volume['embedder'].cuda()(coords)
        kernel_w = psf_model.blur_volume['mlpnet'].cuda()(torch.cat([coords, feat], dim=1))
        kernel_w = kernel_w.reshape(patch_size, patch_size, level, -1)
        psfV_left, psfV_right = kernel_w[..., :3], kernel_w[..., 3:]
        
        # visualize psfvolumes
        visualize_PSFVolume(psfV_left[..., 0].cpu().numpy(), min_depth, max_depth, Path(self.opts.paths.output_dir), 'left_r')
        visualize_PSFVolume(psfV_left[..., 1].cpu().numpy(), min_depth, max_depth, Path(self.opts.paths.output_dir), 'left_g')
        visualize_PSFVolume(psfV_left[..., 2].cpu().numpy(), min_depth, max_depth, Path(self.opts.paths.output_dir), 'left_b')
        visualize_PSFVolume(psfV_right[..., 0].cpu().numpy(), min_depth, max_depth, Path(self.opts.paths.output_dir), 'right_r')
        visualize_PSFVolume(psfV_right[..., 1].cpu().numpy(), min_depth, max_depth, Path(self.opts.paths.output_dir), 'right_g')
        visualize_PSFVolume(psfV_right[..., 2].cpu().numpy(), min_depth, max_depth, Path(self.opts.paths.output_dir), 'right_b')
    
    
    