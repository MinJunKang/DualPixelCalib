

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
from src.utils.base import create_dir
from src.utils.visualizer import visualize_PSFVolume, save_as_gif, visualize_corner_v2
from src.model.utils.common import masked_mean, masked_minmax, normalize
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
import torch.nn.functional as F
from torchvision.utils import save_image



# class to store camera related parameters
class CameraObject(object):
    
    def __init__(self, opts):
        
        self.opts = opts
        self.tag = int(10 * self.opts.calib.psfboard.target_focal)
        
        # camera parameters
        self.calib_data = self.loadCalibdata()
        
        # diffusion model for denoising patches
        if self.opts.calib.psfboard.use_diffusion_denoising:
            unet = Unet(
                dim = 64,
                dim_mults = (1, 2, 4, 8),
                flash_attn = True).cuda()
            self.diffusion = GaussianDiffusion(
                unet,
                image_size = self.opts.calib.psfboard.diffusion_input_res,
                timesteps = 1000,           # number of steps
                sampling_timesteps = self.opts.calib.psfboard.sampling_timesteps    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
            ).cuda().eval()
            self.diffusion_ckpt_path = self.psf_path / 'diffusion_ckpt.pt'
    
    # load calibration data with learned PSF model (optional)
    def loadCalibdata(self):
        calib_data = {}
        self.calib_path = create_dir(Path(self.opts.paths.calib_dir))
        self.psf_path = create_dir(self.calib_path.parent / 'psfcalib')
        
        # load intrinsic calibration data
        if (self.calib_path / 'calib_camera.pkl').is_file():
            calib_data.update({'camera': pickle.load(open(self.calib_path / 'calib_camera.pkl', 'rb'))})
        if (self.calib_path / 'calib_lidar.pkl').is_file():
            calib_data.update({'lidar': pickle.load(open(self.calib_path / 'calib_lidar.pkl', 'rb'))})
        if (self.calib_path / 'calib_stereo.pkl').is_file():
            calib_data.update({'stereo': pickle.load(open(self.calib_path / 'calib_stereo.pkl', 'rb'))})
        # load PSF calibration data
        if (self.psf_path / f'training_data_{self.tag}.pkl').is_file():
            calib_data.update({'training_data': pickle.load(open(self.psf_path / f'training_data_{self.tag}.pkl', 'rb'))})
                    
        return calib_data
        
    # run intrinsic calibration with all-in-focus images
    def runIntrinsicCalib(self, board, observations: dict, window_size: int = 34):
        device = observations['device']
        assert device in ['camera', 'lidar']
        
        # step1: detect boards
        observations = board.detectCalibBoard(observations, window_size=window_size)
        
        # step2: run initial intrinsic calibration
        calib_results, per_scene_results = board.runSingleCalib(observations)
        print(f'Coarse RE of {device}: ', sum(per_scene_results['ret']) / len(per_scene_results['ret']))
        
        # step3: propagate corners with view rejection
        observations = board.propagateCorners(observations, calib_results, per_scene_results)
        
        # step4: run intrinsic calibration with refined corners
        calib_results, per_scene_results = board.runSingleCalib(observations)
        print(f'Fine RE of {device}: ', sum(per_scene_results['ret']) / len(per_scene_results['ret']))
        
        # test plot
        # for n, idx in enumerate(per_scene_results['idx']):
        #     visualize_corner_v2(observations['images'][idx], per_scene_results['x'][n], f'tmp_{n}.png')
        print(calib_results)
        
        # step5: save calibration results
        self.calib_data[device] = calib_results
        pickle.dump(calib_results, open(self.calib_path / f'calib_{device}.pkl', 'wb'))
        
        # save per_scene_results to observations
        for idx, rvec, tvec in zip(per_scene_results['idx'], per_scene_results['rvecs'], per_scene_results['tvecs']):
            observations['corners'][idx]['rvec'] = rvec
            observations['corners'][idx]['tvec'] = tvec
        
    # run extrinsic calibration of camera and lidar
    def runExtrinsicCalib(self, board, observations_1: dict, observations_2: dict):
        
        # run stereo calibration
        calib_results, err1, err2 = board.runStereoCalib(observations_1, observations_2, self.calib_data)
        
        device_1, device_2 = observations_1['device'], observations_2['device']
        print(f'RE of {device_1} to {device_2}: ', err1)
        print(f'RE of {device_2} to {device_1}: ', err2)
        print(calib_results)
        
        self.calib_data['stereo'] = calib_results
        pickle.dump(calib_results, open(self.calib_path / f'calib_stereo.pkl', 'wb'))
    
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
        pickle.dump(training_data, open(self.psf_path / f'training_data_{self.tag}.pkl', 'wb'))
    
    # get meta data from observations
    def estimateMetadata(self, board, training_data: dict):
        fnumber = board.get_numerical_fnumber(self.opts.calib.psfboard.target_focal)
        focal_mm = self.opts.calib.observation.focal_mm
        aperture = max(self.calib_data['camera']['mtx'][0, 0], self.calib_data['camera']['mtx'][1, 1]) / fnumber
        
        # get depth range
        mask_start = (training_data['patch_uv'][..., 0] >= 0) & (training_data['patch_uv'][..., 1] >= 0)
        mask_end = (training_data['patch_uv'][..., 0] <= self.calib_data['camera']['image_size'][1] - 1) & (training_data['patch_uv'][..., 1] <= self.calib_data['camera']['image_size'][0] - 1)
        mask = torch.tensor(mask_start & mask_end & (training_data['patch_3d'][..., -1] > 0))
        depth = torch.tensor(training_data['patch_3d'][..., -1])
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
                self.diffusion.load_state_dict(checkpoint['model'], strict=True)
        
        # learn psf model
        datamodule = hydra.utils.instantiate(self.opts.get("data"), 
                                             model_cfg=self.opts.model.model_cfg, 
                                             calib_data=self.calib_data, 
                                             focal_distance=self.opts.calib.observation.focal_distance_mm,
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
        # self.savePSFImage(model)
        # self.savePSFVolume(model)
        # import pdb; pdb.set_trace()
        return model
    
    def trainDiffusionModel(self):
        self.diffusion.model.training = True
        batch_imgs = self.calib_data['training_data']['clean_patch']
        batch_imgs = torch.Tensor(batch_imgs).permute(0, 3, 1, 2)
        batch_imgs = torch.split(batch_imgs, self.opts.calib.psfboard.diffusion_batchsize)
        optim = torch.optim.Adam(self.diffusion.parameters(), lr=1e-4, betas=(0.9, 0.99))
        pbar = tqdm(range(self.opts.calib.psfboard.diffusion_epoch))
        for epoch in pbar:
            loss_diffusion = 0.0
            for batch in batch_imgs:
                resized_batch = F.interpolate(batch.cuda(), size=self.diffusion.image_size, mode='bilinear', align_corners=True)
                losses = self.diffusion(resized_batch / 255.0)
                losses.backward()
                optim.step()
                optim.zero_grad()
                loss_diffusion += float(losses.item())
            pbar.set_postfix({"loss": loss_diffusion / len(batch_imgs)})

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
        
    @torch.no_grad()
    def estimateDepthFromPSF(self, psf_model, imglg, imgrg, imgcg, idx, gt_depth=None, level=255, pad=60, resize_val=1.0):
        psf_model.eval()
        img_patch_size = 223
        stride = 67
        border = 51
        # if we do resize:
        if gt_depth is not None:
            gt_depth = cv2.resize(gt_depth, (0, 0), fx=resize_val, fy=resize_val)
        imglg = cv2.resize(imglg, (0, 0), fx=resize_val, fy=resize_val)
        imgrg = cv2.resize(imgrg, (0, 0), fx=resize_val, fy=resize_val)
        
        h_img, w_img = imglg.shape[:2]
        h_img_ori, w_img_ori = self.calib_data['camera']['image_size']
        h_ratio, w_ratio = h_img / h_img_ori, w_img / w_img_ori
        max_ratio = max(h_ratio, w_ratio)
        patch_size = int(psf_model.hparams.model_cfg.psf_volume.patch_uvsize * max_ratio)
        patch_size = patch_size + 1 if patch_size % 2 == 0 else patch_size
        img_patch_size = int(img_patch_size * max_ratio)
        stride = int(stride * max_ratio)
        border = int(border * max_ratio)
        img_patch_size = img_patch_size + 1 if img_patch_size % 2 == 0 else img_patch_size
        stride = stride + 1 if stride % 2 == 0 else stride
        border = border + 1 if border % 2 == 0 else border
        
        # new camera matrix
        umtx = self.calib_data['camera']['umtx']
        umtx[0, :] *= w_ratio
        umtx[1, :] *= h_ratio
        umtx = torch.from_numpy(np.float32(umtx)).cuda()

        # prepare patches
        img_l_patches = [imglg[y:y+img_patch_size, x:x+img_patch_size] 
                         for y in range(0, h_img - img_patch_size, stride) for x in range(0, w_img - img_patch_size, stride)]
        img_r_patches = [imgrg[y:y+img_patch_size, x:x+img_patch_size] 
                         for y in range(0, h_img - img_patch_size, stride) for x in range(0, w_img - img_patch_size, stride)]
        uv_coords = [torch.stack(torch.meshgrid(torch.arange(x, x + img_patch_size), torch.arange(y, y + img_patch_size), indexing='xy'), dim=-1).cuda()
                     for y in range(0, h_img - img_patch_size, stride) for x in range(0, w_img - img_patch_size, stride)]
        uv_coords = torch.stack(uv_coords)

        # uv coord normalization
        uv_coords_norm = uv_coords.float().clone()
        uv_coords_norm[..., 0] = (uv_coords_norm[..., 0] - umtx[0, 2]) / umtx[0, 0]
        uv_coords_norm[..., 1] = (uv_coords_norm[..., 1] - umtx[1, 2]) / umtx[1, 1]

        min_depth, max_depth = psf_model.min_depth.item() + pad, psf_model.max_depth.item() - pad
        depths = torch.linspace(min_depth, max_depth, level).cuda()
        depth_level = torch.zeros([h_img, w_img], dtype=torch.long)
        depth_cost = torch.zeros([h_img, w_img])
        
        # xyz approximation
        xy_coords = torch.linspace(-1/2*psf_model.bound_xy, 1/2*psf_model.bound_xy, patch_size).cuda()
        xyz_coords = torch.stack(torch.meshgrid(xy_coords, xy_coords, depths, indexing='xy'), dim=-1)  # [patch_size, patch_size, level, 3]

        for n in tqdm(range(uv_coords_norm.shape[0])):
            
            # get uv coord (center crop)
            half_patch_size, half_img_size = patch_size // 2, img_patch_size // 2
            uv_coords_norm_sampled = uv_coords_norm[n, half_img_size - half_patch_size:half_img_size + half_patch_size + 1, half_img_size - half_patch_size:half_img_size + half_patch_size + 1]
            uv_coords_norm_sampled = repeat(uv_coords_norm_sampled, 'x y c -> x y l c', l=level)

            # get psf volume
            feat = psf_model.blur_volume['featnet'].cuda()(xyz_coords.reshape(-1, 3))
            coords = torch.cat([uv_coords_norm_sampled.reshape(-1, 2), xyz_coords.reshape(-1, 3)], dim=1)
            if 'embedder' in psf_model.blur_volume:
                coords = psf_model.blur_volume['embedder'].cuda()(coords)
            kernel_w = psf_model.blur_volume['mlpnet'].cuda()(torch.cat([coords, feat], dim=1))
            kernel_w = rearrange(kernel_w, '(kh kw l) c -> l kh kw c', kh=patch_size, kw=patch_size)
            if 'psfgrid' in psf_model.blur_volume:
                kernel_w_coarse = rearrange(psf_model.blur_volume['psfgrid'].cuda()(rearrange(xyz_coords, 'kh kw l c -> 1 l kh kw c')), 'b c n kh kw -> b n kh kw c')
                kernel_w = kernel_w * kernel_w_coarse[0]
            psfk_left, psfk_right = kernel_w[..., :3], kernel_w[..., 3:]
            
            # normalization
            psfk_left = psfk_left / (psfk_left.reshape(level, -1, 3).sum(dim=1))[:, None, None] * 0.5  # [depth_level, kh, kw, 3]
            psfk_right = psfk_right / (psfk_right.reshape(level, -1, 3).sum(dim=1))[:, None, None] * 0.5
            
            # torch conv2d is actually cross-correlation, need to flip kernel to correctly perform conv
            psfk_left = torch.flip(psfk_left, [1, 2])
            psfk_right = torch.flip(psfk_right, [1, 2])
            
            # image patch normalize
            left_dp_img = torch.Tensor(img_l_patches[n] / 255.)[None,...].permute(0, 3, 1, 2).float().cuda()
            right_dp_img = torch.Tensor(img_r_patches[n] / 255.)[None,...].permute(0, 3, 1, 2).float().cuda()

            l = F.conv2d(left_dp_img, psfk_right.permute(0, 3, 1, 2), padding='same').squeeze()
            r = F.conv2d(right_dp_img, psfk_left.permute(0, 3, 1, 2), padding='same').squeeze()
            l = l[:,border:-border, border:-border]
            r = r[:,border:-border, border:-border]
            
            err = torch.sqrt(torch.mean((l - r) ** 2, dim=(1, 2)))
            fill_coords = uv_coords[n, 
                                    img_patch_size//2 - stride//2:img_patch_size//2+stride//2 + 1, 
                                    img_patch_size//2 - stride//2:img_patch_size//2+stride//2 + 1].reshape(-1, 2).permute(1, 0).cpu()
            conf = F.softmin(err, dim=0)
            fval, min_level = torch.max(conf, 0)
            depth_cost[fill_coords[1,:], fill_coords[0,:]] = fval
            depth_level[fill_coords[1,:], fill_coords[0,:]] = min_level
        
        # crop result
        depth_cost = (depth_cost/depth_cost.max() * 255).cpu().numpy().astype('uint8')
        depth_level = (depth_level/(level - 1) * 255).cpu().numpy().astype('uint8')
        depth_cost = cv2.resize(depth_cost, (0, 0), fx=0.5, fy=0.5)
        depth_level = cv2.resize(depth_level, (0, 0), fx=0.5, fy=0.5)
        depth_cost = depth_cost[img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+6), img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+4)]
        depth_level = depth_level[img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+6), img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+4)]
        
        print('done')    
        # make result img
        cv2.imwrite(f'output_cost_{idx}.png', depth_cost)
        cv2.imwrite(f'output_depth_{idx}.png', depth_level)
        
        # evaluation by depth
        if gt_depth is not None:
            gt_depth = gt_depth[img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+6), img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+4)]
            cv2.imwrite(f'gt_depth_{idx}.png', gt_depth)
            
    @torch.no_grad()
    def estimateDepthFromPSF_2stage(self, psf_model, imglg, imgrg, imgcg, idx, gt_depth=None, coarse_level=16, fine_level=9, pad=10, resize_val=1.0):
        psf_model.eval()
        img_patch_size = 111
        stride = 33
        border = 25
        # if we do resize:
        if gt_depth is not None:
            gt_depth = cv2.resize(gt_depth, (0, 0), fx=resize_val, fy=resize_val)
        imglg = cv2.resize(imglg, (0, 0), fx=resize_val, fy=resize_val)
        imgrg = cv2.resize(imgrg, (0, 0), fx=resize_val, fy=resize_val)
        
        h_img, w_img = imglg.shape[:2]
        h_img_ori, w_img_ori = self.calib_data['camera']['image_size']
        h_ratio, w_ratio = h_img / h_img_ori, w_img / w_img_ori
        max_ratio = max(h_ratio, w_ratio)
        patch_size = int(psf_model.hparams.model_cfg.psf_volume.patch_uvsize * max_ratio)
        patch_size = patch_size + 1 if patch_size % 2 == 0 else patch_size
        # img_patch_size = int(img_patch_size * max_ratio)
        # stride = int(stride * max_ratio)
        # border = int(border * max_ratio)
        # img_patch_size = img_patch_size + 1 if img_patch_size % 2 == 0 else img_patch_size
        # stride = stride + 1 if stride % 2 == 0 else stride
        # border = border + 1 if border % 2 == 0 else border
        
        # new camera matrix
        umtx = self.calib_data['camera']['umtx']
        umtx[0, :] *= w_ratio
        umtx[1, :] *= h_ratio
        umtx = torch.from_numpy(np.float32(umtx)).cuda()

        # prepare patches
        img_l_patches = [imglg[y:y+img_patch_size, x:x+img_patch_size] 
                         for y in range(0, h_img - img_patch_size, stride) for x in range(0, w_img - img_patch_size, stride)]
        img_r_patches = [imgrg[y:y+img_patch_size, x:x+img_patch_size] 
                         for y in range(0, h_img - img_patch_size, stride) for x in range(0, w_img - img_patch_size, stride)]
        uv_coords = [torch.stack(torch.meshgrid(torch.arange(x, x + img_patch_size), torch.arange(y, y + img_patch_size), indexing='xy'), dim=-1).cuda()
                     for y in range(0, h_img - img_patch_size, stride) for x in range(0, w_img - img_patch_size, stride)]
        uv_coords = torch.stack(uv_coords)

        # uv coord normalization
        uv_coords_norm = uv_coords.float().clone()
        uv_coords_norm[..., 0] = (uv_coords_norm[..., 0] - umtx[0, 2]) / umtx[0, 0]
        uv_coords_norm[..., 1] = (uv_coords_norm[..., 1] - umtx[1, 2]) / umtx[1, 1]

        min_depth, max_depth = psf_model.min_depth.item() + pad, psf_model.max_depth.item() - pad
        depths = torch.linspace(min_depth, max_depth, coarse_level).cuda()
        depth_level = torch.zeros([h_img, w_img], dtype=torch.long)
        depth_cost = torch.zeros([h_img, w_img])
        
        # xyz approximation
        xy_coords = torch.linspace(-1/2*psf_model.bound_xy, 1/2*psf_model.bound_xy, patch_size).cuda()
        xyz_coords = torch.stack(torch.meshgrid(xy_coords, xy_coords, depths, indexing='xy'), dim=-1)  # [patch_size, patch_size, level, 3]

        for n in tqdm(range(uv_coords_norm.shape[0])):
            
            # get uv coord (center crop)
            half_patch_size, half_img_size = patch_size // 2, img_patch_size // 2
            uv_coords_norm_sampled = uv_coords_norm[n, half_img_size - half_patch_size:half_img_size + half_patch_size + 1, half_img_size - half_patch_size:half_img_size + half_patch_size + 1]
            uv_coords_norm_sampled_coarse = repeat(uv_coords_norm_sampled, 'x y c -> x y l c', l=coarse_level)
            
            # image patch normalize
            left_dp_img = torch.Tensor(img_l_patches[n] / 255.)[None,...].permute(0, 3, 1, 2).float().cuda()
            right_dp_img = torch.Tensor(img_r_patches[n] / 255.)[None,...].permute(0, 3, 1, 2).float().cuda()
            
            ### Coarse Matching Step ###

            # get psf volume
            feat = psf_model.blur_volume['featnet'].cuda()(xyz_coords.reshape(-1, 3))
            coords = torch.cat([uv_coords_norm_sampled_coarse.reshape(-1, 2), xyz_coords.reshape(-1, 3)], dim=1)
            if 'embedder' in psf_model.blur_volume:
                coords = psf_model.blur_volume['embedder'].cuda()(coords)
            kernel_w = psf_model.blur_volume['mlpnet'].cuda()(torch.cat([coords, feat], dim=1))
            kernel_w = rearrange(kernel_w, '(kh kw l) c -> l kh kw c', kh=patch_size, kw=patch_size)
            if 'psfgrid' in psf_model.blur_volume:
                kernel_w_coarse = rearrange(psf_model.blur_volume['psfgrid'].cuda()(rearrange(xyz_coords, 'kh kw l c -> 1 l kh kw c')), 'b c n kh kw -> b n kh kw c')
                kernel_w = kernel_w * kernel_w_coarse[0]
            psfk_left, psfk_right = kernel_w[..., :3], kernel_w[..., 3:]
            
            # normalization
            psfk_left = psfk_left / (psfk_left.reshape(coarse_level, -1, 3).sum(dim=1))[:, None, None] * 0.5  # [depth_level, kh, kw, 3]
            psfk_right = psfk_right / (psfk_right.reshape(coarse_level, -1, 3).sum(dim=1))[:, None, None] * 0.5
            
            # torch conv2d is actually cross-correlation, need to flip kernel to correctly perform conv
            psfk_left = torch.flip(psfk_left, [1, 2])
            psfk_right = torch.flip(psfk_right, [1, 2])
            l = F.conv2d(left_dp_img, psfk_right.permute(0, 3, 1, 2), padding='same').squeeze()
            r = F.conv2d(right_dp_img, psfk_left.permute(0, 3, 1, 2), padding='same').squeeze()
            l = l[:,border:-border, border:-border]
            r = r[:,border:-border, border:-border]
            
            err = torch.sqrt(torch.mean((l - r) ** 2, dim=(1, 2)))
            fill_coords = uv_coords[n, 
                                    img_patch_size//2 - stride//2:img_patch_size//2+stride//2 + 1, 
                                    img_patch_size//2 - stride//2:img_patch_size//2+stride//2 + 1].reshape(-1, 2).permute(1, 0).cpu()
            conf = F.softmin(err, dim=0)
            fval_coarse, min_level_coarse = torch.max(conf, 0)
            
            ### Fine Matching Step ###
            eps_depth = (max_depth - min_depth) / coarse_level
            fine_min_depth, fine_max_depth = max(min_depth, depths[min_level_coarse].item() - eps_depth), min(max_depth, depths[min_level_coarse].item() + eps_depth)
            fine_depths = torch.linspace(fine_min_depth, fine_max_depth, fine_level).cuda()
            xyz_coords_fine = torch.stack(torch.meshgrid(xy_coords, xy_coords, fine_depths, indexing='xy'), dim=-1)
            uv_coords_norm_sampled_fine = repeat(uv_coords_norm_sampled, 'x y c -> x y l c', l=fine_level)
            
            # get psf volume
            feat = psf_model.blur_volume['featnet'].cuda()(xyz_coords_fine.reshape(-1, 3))
            coords = torch.cat([uv_coords_norm_sampled_fine.reshape(-1, 2), xyz_coords_fine.reshape(-1, 3)], dim=1)
            if 'embedder' in psf_model.blur_volume:
                coords = psf_model.blur_volume['embedder'].cuda()(coords)
            kernel_w = psf_model.blur_volume['mlpnet'].cuda()(torch.cat([coords, feat], dim=1))
            kernel_w = rearrange(kernel_w, '(kh kw l) c -> l kh kw c', kh=patch_size, kw=patch_size)
            if 'psfgrid' in psf_model.blur_volume:
                kernel_w_coarse = rearrange(psf_model.blur_volume['psfgrid'].cuda()(rearrange(xyz_coords_fine, 'kh kw l c -> 1 l kh kw c')), 'b c n kh kw -> b n kh kw c')
                kernel_w = kernel_w * kernel_w_coarse[0]
            psfk_left, psfk_right = kernel_w[..., :3], kernel_w[..., 3:]
            
            # normalization
            psfk_left = psfk_left / (psfk_left.reshape(fine_level, -1, 3).sum(dim=1))[:, None, None] * 0.5  # [depth_level, kh, kw, 3]
            psfk_right = psfk_right / (psfk_right.reshape(fine_level, -1, 3).sum(dim=1))[:, None, None] * 0.5
            
            # torch conv2d is actually cross-correlation, need to flip kernel to correctly perform conv
            psfk_left = torch.flip(psfk_left, [1, 2])
            psfk_right = torch.flip(psfk_right, [1, 2])
            l = F.conv2d(left_dp_img, psfk_right.permute(0, 3, 1, 2), padding='same').squeeze()
            r = F.conv2d(right_dp_img, psfk_left.permute(0, 3, 1, 2), padding='same').squeeze()
            l = l[:,border:-border, border:-border]
            r = r[:,border:-border, border:-border]
            
            err = torch.sqrt(torch.mean((l - r) ** 2, dim=(1, 2)))
            fill_coords = uv_coords[n, 
                                    img_patch_size//2 - stride//2:img_patch_size//2+stride//2 + 1, 
                                    img_patch_size//2 - stride//2:img_patch_size//2+stride//2 + 1].reshape(-1, 2).permute(1, 0).cpu()
            conf = F.softmin(err, dim=0)
            fval_fine, min_level_fine = torch.max(conf, 0)
            
            fval = fval_coarse * fval_fine
            min_level = min_level_coarse * fine_level + min_level_fine - fine_level // 2
            
            depth_cost[fill_coords[1,:], fill_coords[0,:]] = fval
            depth_level[fill_coords[1,:], fill_coords[0,:]] = min_level
        
        # crop result
        depth_cost = (depth_cost/depth_cost.max() * 255).cpu().numpy().astype('uint8')
        depth_level = (depth_level/(coarse_level * fine_level - 1) * 255).cpu().numpy().astype('uint8')
        # depth_cost = cv2.resize(depth_cost, (0, 0), fx=0.5, fy=0.5)
        # depth_level = cv2.resize(depth_level, (0, 0), fx=0.5, fy=0.5)
        depth_cost = depth_cost[img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+6), img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+4)]
        depth_level = depth_level[img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+6), img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+4)]
        
        print('done')    
        # make result img
        cv2.imwrite(f'output_cost_{idx}_2stage.png', depth_cost)
        cv2.imwrite(f'output_depth_{idx}_2stage.png', depth_level)
        
        # evaluation by depth
        if gt_depth is not None:
            gt_depth = gt_depth[img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+6), img_patch_size//2 - stride//2:-(img_patch_size//2-stride//2+4)]
            cv2.imwrite(f'gt_depth_{idx}.png', gt_depth)
    
    