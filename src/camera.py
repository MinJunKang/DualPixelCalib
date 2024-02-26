

import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
from pathlib import Path
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch.optim import Adam
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity
import torch.nn.functional as F


def PSNR(clean, noisy):
    if isinstance(clean, torch.Tensor):
        if clean.is_cuda:
            clean = clean.cpu().numpy()
        else:
            clean = clean.numpy()

    if isinstance(noisy, torch.Tensor):
        if noisy.is_cuda and noisy.requires_grad:
            noisy = noisy.detach().cpu().numpy()
        else:
            noisy = noisy.numpy()

    mse = np.mean((clean - noisy) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
    return psnr 

# class to store camera related parameters
class CameraObject(object):
    
    def __init__(self, opts, load_path=None, load_diff_path='/mnt/ssd4tb/jasonjeong/dual-pixel/DualPixelCalib/logs/240219_calib_F15/psfcalib/model-1000.pt'):
        
        self.opts = opts
        self.tag = int(10 * self.opts.calib.psfboard.target_focal)
        
        # camera parameters
        self.calib_data = self.loadCalibdata(load_path)

        self.unet = Unet(
                    dim = 64,
                    dim_mults = (1, 2, 4, 8),
                    flash_attn = True
                    ).cuda()

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size = 256,
            timesteps = 1000,           # number of steps
            # sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
            sampling_timesteps = 150    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        ).cuda().eval()

        self.diffusion.model.training = True
        
        self.PSF_volume_L = nn.Parameter(nn.init.zeros_(torch.zeros(32, 32, 20))).requires_grad_()
        self.PSF_volume_R = nn.Parameter(nn.init.zeros_(torch.zeros(32, 32, 20))).requires_grad_()

        self.optim = Adam(self.diffusion.parameters(), lr=1e-4, betas=(0.9, 0.99))
        self.optim_PSF = Adam([self.PSF_volume_L, self.PSF_volume_R], lr=1e-4, betas=(0.9, 0.99))

        if os.path.exists(load_diff_path):
            unet_state_dict = torch.load(load_diff_path,map_location="cuda")
            if 'model' in unet_state_dict.keys():
                self.diffusion.load_state_dict(unet_state_dict['model'])
            else:
                self.diffusion.load_state_dict(unet_state_dict)
            self.diffusion_requires_training=False
        else:
            self.diffusion_requires_training=True


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
        depth = training_data['patch_3d'][0][..., -1]
        depth_mean, depth_min, depth_max = depth.mean(), depth.min(), depth.max()
        
        return {'focal_mm': focal_mm, 'aperture': aperture, 'fnumber': fnumber, 'depth_mean': depth_mean, 'depth_min': depth_min, 'depth_max': depth_max}
    
    # train PSF calibration model
    def trainPSFCalibModel(self):
        
        # unet_state_dict = torch.load('/media/vilab/usb2/WSSS_Diff/denoising-diffusion-pytorch/results/model-140-pascal.pt',map_location="cuda")
        # diffusion.load_state_dict(unet_state_dict['model'])
        self.diffusion.model.training = True

        if self.diffusion_requires_training:
            batch_imgs = self.calib_data['training_data']['clean_patch'][0]
            # toTensor = transforms.ToTensor()
            batch_imgs = torch.Tensor(batch_imgs).cuda()[:, None, ...]
            batch_imgs = torch.cat([batch_imgs, batch_imgs, batch_imgs], axis=1)
            batch_imgs = torch.split(batch_imgs, 10)
            # batch_imgs = torch.cat(batch_imgs).cuda()

            for epoch in tqdm(range(10)):
                for batch in batch_imgs:
                    losses = self.diffusion(batch)
                    losses.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                print("%.2f" % float(losses.detach().cpu().numpy()))

                data = {
                    'step': epoch,
                    'model': self.diffusion.state_dict(),
                    'opt': self.optim.state_dict(),
                    'ema': self.ema.state_dict(),
                    'version': 'v0.1'
                }

                torch.save(data, '/mnt/ssd4tb/jasonjeong/dual-pixel/DualPixelCalib/logs/240219_calib_F15/psfcalib/model-1000.pt')

        noisy_patches = self.calib_data['training_data']['noisy_patch'][0]
        noisy_patches = torch.Tensor(noisy_patches).cuda()[:,None,...]
        noisy_patches = torch.cat([noisy_patches, noisy_patches, noisy_patches], axis=1)
        noisy_patch_split = torch.split(noisy_patches, 4)
        self.diffusion.model.training=False
        max_psnr = 0
        best_iter = 0
        best_ssim = 0

        max_dist = self.calib_data['training_data']['depth_max']
        min_dist = self.calib_data['training_data']['depth_min']
        dist = self.calib_data['training_data']['tvec'][0][:,-1,0]
        sample_psf_level = (2 * (dist - min_dist) / (max_dist - min_dist)) - 1

        psf_sample_H, psf_sample_W = torch.meshgrid((torch.linspace(-1, 1, 32), torch.Tensor(sample_psf_level)))
        grid_sample_HW = torch.cat([psf_sample_H[..., None], psf_sample_W[...,None]], axis=2)

        for epoch_psf in tqdm(range(20)):
            avg_loss = 0
            for idx, noisy_patch in enumerate(noisy_patch_split):

                iter_idx = range(idx*4, idx*4+len(noisy_patch))

                _, v, _ = self.diffusion.p_losses(noisy_patch, torch.Tensor([1]).long().cuda())
                img_out = self.diffusion.predict_start_from_v(noisy_patch, torch.Tensor([900]).long().cuda(), v).detach()

                lpatch = torch.Tensor(self.calib_data['training_data']['lpatch'][0][iter_idx]).cuda()[:, None, ...]
                lpatch = torch.cat([lpatch, lpatch, lpatch], axis=1)
                rpatch = torch.Tensor(self.calib_data['training_data']['rpatch'][0][iter_idx]).cuda()[:, None, ...]
                rpatch = torch.cat([rpatch, rpatch, rpatch], axis=1)
                _, v_L, _ = self.diffusion.p_losses(lpatch, torch.Tensor([1]).long().cuda())
                _, v_R, _ = self.diffusion.p_losses(rpatch, torch.Tensor([1]).long().cuda())
                img_out_L = self.diffusion.predict_start_from_v(lpatch, torch.Tensor([200]).long().cuda(), v_L)
                img_out_R = self.diffusion.predict_start_from_v(rpatch, torch.Tensor([200]).long().cuda(), v_R)

                psf_weight_L = F.grid_sample(self.PSF_volume_L[None, ...], grid_sample_HW[None, :,iter_idx,:], align_corners=False).permute(3, 0, 1, 2).cuda()
                psf_weight_R = F.grid_sample(self.PSF_volume_R[None, ...], grid_sample_HW[None, :,iter_idx,:], align_corners=False).permute(3, 0, 1, 2).cuda()

                pred_psf_L = F.conv2d(img_out[:,0:1,...], psf_weight_L, padding='same')
                pred_psf_R = F.conv2d(img_out[:,0:1,...], psf_weight_R, padding='same')

                pred_psf_L = torch.diagonal(pred_psf_L).permute(2, 0, 1)
                pred_psf_R = torch.diagonal(pred_psf_R).permute(2, 0, 1)

                loss_L = F.mse_loss(img_out_L[:,0,...].detach(), pred_psf_L)
                loss_R = F.mse_loss(img_out_R[:,0,...].detach(), pred_psf_R)

                norm_L = ((self.PSF_volume_L.sum(0).sum(0) - 1)**2).sum()
                norm_R = ((self.PSF_volume_R.sum(0).sum(0) - 1)**2).sum()
                if epoch_psf < 4:
                    loss_LR = loss_L + loss_R + 40*(norm_L + norm_R)
                else:
                    loss_LR = loss_L + loss_R + (norm_L + norm_R)

                loss_LR.backward()
                self.optim_PSF.step()
                self.optim_PSF.zero_grad()

                avg_loss += loss_LR.item()
                if idx % 10 == 0:                       
                    print(avg_loss / (idx+1))

            if epoch_psf % 5 == 0:
                print(loss_LR.item())
                torch.save(self.PSF_volume_L, '/mnt/ssd4tb/jasonjeong/dual-pixel/DualPixelCalib/logs/240219_calib_F15/psfcalib/PSF_L.pt')
                torch.save(self.PSF_volume_R, '/mnt/ssd4tb/jasonjeong/dual-pixel/DualPixelCalib/logs/240219_calib_F15/psfcalib/PSF_R.pt')
 
        for diff_iter in range(0, 980, 20):
            img_out = self.diffusion.predict_start_from_v(noisy_patch[None, ...], torch.Tensor([diff_iter]).long().cuda(), v)
            # normalize_img_out = (img_out - img_out.min())*255/(img_out.max()-img_out.min())
            psnr_out = PSNR(self.calib_data['training_data']['clean_patch'][0][idx], img_out)
            ssim_out = structural_similarity(self.calib_data['training_data']['clean_patch'][0][idx].astype(np.uint8), img_out[0,0].detach().cpu().numpy().astype(np.uint8), full=True)
            # psnr_out = PSNR(self.calib_data['training_data']['clean_patch'][0][idx], normalize_img_out)
            if psnr_out > max_psnr:
                max_psnr = psnr_out
                best_ssim = ssim_out[0]
                best_iter = diff_iter
        print(best_iter, max_psnr, best_ssim)

        max_psnr=0
        best_iter=0

        return None
        # import pdb; pdb.set_trace()
    
    def savePSFPatches(self, cfg):

        save_path = cfg['paths']['output_dir']
        directions = ['Left', 'Center', 'Right']
        scales = ['10','05','20']

        os.makedirs(save_path+'/lpatch', exist_ok=True)
        os.makedirs(save_path+'/rpatch', exist_ok=True)
        # for dir in directions:
        os.makedirs(save_path+'/clean_patch', exist_ok=True)
        os.makedirs(save_path+'/noisy_patch', exist_ok=True)
        
        for idx, scale in enumerate(scales):

            num_patches = self.calib_data['training_data']['clean_patch'][idx].shape[0]

            for patch in range(num_patches):
                clean_patch = self.calib_data['training_data']['clean_patch'][idx][patch,...]
                noisy_patch = self.calib_data['training_data']['noisy_patch'][idx][patch,...]
                lpatch = self.calib_data['training_data']['lpatch'][idx][patch,...]
                rpatch = self.calib_data['training_data']['rpatch'][idx][patch,...]
                mean_distance = np.linalg.norm(self.calib_data['training_data']['patch_3d'][idx][patch,...], axis=2).mean()
                save_name = 'F{:.1f}'.format(self.opts.calib.psfboard.target_focal) + \
                            '_Patch{:d}'.format(patch) + \
                            '_D{:.2f}'.format(mean_distance/1000) + \
                            '_S'+scale+'.png'
                cv2.imwrite(save_path+"/clean_patch/"+save_name, clean_patch.astype(np.uint8))
                cv2.imwrite(save_path+"/noisy_patch/"+save_name, noisy_patch.astype(np.uint8))
                cv2.imwrite(save_path+"/lpatch/"+save_name, lpatch.astype(np.uint8))
                cv2.imwrite(save_path+"/rpatch/"+save_name, rpatch.astype(np.uint8))
