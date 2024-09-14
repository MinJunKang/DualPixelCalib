from typing import Any, Dict, Optional, Tuple
from PIL import Image
import numpy as np
from pathlib import Path
from scipy.ndimage import convolve1d, gaussian_filter1d

import torch
from torchvision import transforms
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class DPCalloader(Dataset):
    
    def __init__(self, data, model_cfg, LDS_cfg, transform, precision):
        
        self.precision = precision
        self.transform = transform
        self.output_type = np.float16 if self.precision in ['16', '16-mixed'] else np.float32   # precision
        
        self.clean_patches = data['training_data']['clean_patch']
        self.left_patches = data['training_data']['lpatch']
        self.right_patches = data['training_data']['rpatch']
        self.aif_patches = data['training_data']['noisy_patch']
        self.points_patches = data['training_data']['patch_3d']
        self.uv_patches = data['training_data']['patch_uv']
        self.rvec = data['training_data']['rvec']
        self.tvec = data['training_data']['tvec']
        self.image_size = data['camera']['image_size']
        
        # Since the depth label of PSF is imbalanced, adapt "Delving into Deep Imbalanced Regression (ICML'21)" to regress imbalance
        # For LDS method, compute effective label density and re-weight
        if LDS_cfg.use_LDS:
            self.LDS_ratio = (model_cfg.level / LDS_cfg.LDS_step)
            self.weights_LDS, self.min_depth, self.max_depth = self.calc_reweight_LDS(data['training_data']['patch_3d'], model_cfg.level, LDS_cfg.LDS_step, LDS_cfg.LDS_sigma, LDS_cfg.LDS_ks)
        else:
            self.LDS_ratio = None
            self.weights_LDS, self.min_depth, self.max_depth = None, data['training_data']['depth_min'], data['training_data']['depth_max']
        
    def gauss1d_kernel_LDS(self, sigma, ks):
        half_ks = (ks - 1) // 2
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel = gaussian_filter1d(base_kernel, sigma=sigma)
        return kernel / max(kernel)
        
    def calc_reweight_LDS(self, depths, level, LDS_step, LDS_sigma, LDS_ks, eps=1e-3):
        depth_all = depths[depths > eps]
        bins_number, _ = np.histogram(depth_all, bins=int(level / LDS_step))
        bins_number = np.sqrt(bins_number)
        kernel_window = self.gauss1d_kernel_LDS(LDS_sigma, LDS_ks)
        smoothed_bins = convolve1d(bins_number, weights=kernel_window, mode='reflect')
        scaling = np.sum(bins_number) / np.sum(np.array(bins_number) / (np.array(smoothed_bins) + 1e-6))
        weights = np.float32(scaling / (smoothed_bins + 1e-6)).clip(0, 1)
        return weights, depth_all.min(), depth_all.max()
        
    def __getitem__(self, index):
        sample_out = dict()
        
        # preprocess data
        cleans = self.transform(self.output_type(self.clean_patches[index] / 255.0))
        lefts = self.transform(self.output_type(self.left_patches[index] / 255.0))
        rights = self.transform(self.output_type(self.right_patches[index] / 255.0))
        aifs = self.transform(self.output_type(self.aif_patches[index] / 255.0))
        uvs = self.transform(self.output_type(self.uv_patches[index]))
        points = self.transform(self.output_type(self.points_patches[index]))
        
        mask_uv_start = (self.uv_patches[index, ..., 0] >= 0) & (self.uv_patches[index, ..., 1] >= 0)
        mask_uv_end = (self.uv_patches[index, ..., 0] <= self.image_size[1] - 1) & (self.uv_patches[index, ..., 1] <= self.image_size[0] - 1)
        mask = (self.points_patches[index, ..., -1] > 0) & mask_uv_start & mask_uv_end
        mask = self.transform(self.output_type(mask))
        
        # Based on LDS, calc weight mask, to resolve imbalanced samples along depth
        depths = self.points_patches[index, ..., -1]
        if self.weights_LDS is not None:
            ind = (depths - self.min_depth) / (self.max_depth - self.min_depth) * (int(self.LDS_ratio) - 1)
            ind_0 = np.int64(ind)
            ind_1 = np.clip(ind_0 + 1, 0, int(self.LDS_ratio) - 1)
            val_0 = self.weights_LDS[ind_0]
            val_1 = self.weights_LDS[ind_1]
            weight = self.output_type(val_0 * (ind_1 - ind) + val_1 * (ind - ind_0))  # linear interpolation
        else:
            weight = np.ones_like(depths)
        weight = self.transform(self.output_type(weight))
        
        # convert to tensor
        sample_out['clean'] = cleans  # [C, H, W]
        sample_out['left'] = lefts  # [C, H, W]
        sample_out['right'] = rights  # [C, H, W]
        sample_out['aif'] = aifs  # [C, H, W]
        sample_out['mask'] = mask  # [1, H, W]
        sample_out['weight'] = weight  # [1, H, W]
        sample_out['uv_coord'] = uvs  # [2, H, W]
        sample_out['3d_coord'] = points  # [3, H, W]
        
        return sample_out
    
    def __len__(self):
        return len(self.clean_patches)


class DPPSFDataModule(LightningDataModule):
    
    def __init__(self,
                 model_cfg,
                 calib_data,
                 focal_distance,
                 LDS_cfg, 
                 batch_size: int,
                 num_workers: int,
                 pin_memory: bool,
                 precision: str,
                 train_val_split: Tuple[int, int] = (90_000, 10_000)):
        super().__init__()
        
        self.precision = precision
        self.calib_data = calib_data
        self.save_hyperparameters(logger=False, ignore='calib_data')
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        
        self.batch_size_per_device = batch_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor()  # Converts image to tensor
        ])
        minmax_depth = (calib_data['training_data']['depth_min'], calib_data['training_data']['depth_max'])
        
        self.meta_data = {'image_size': calib_data['camera']['image_size'],
                          'focal_mm': calib_data['training_data']['focal_mm'], 
                          'aperture': calib_data['training_data']['aperture'],
                          'fnumber': calib_data['training_data']['fnumber'],
                          'patchSize_px': calib_data['training_data']['patchSize_px'],
                          'px_ratio_max': calib_data['training_data']['px_ratio_max'],
                          'xy_ratio_max': calib_data['training_data']['xy_ratio_max'],
                          'focal_distance': focal_distance, 'depth_range': minmax_depth,
                          'mtx': calib_data['camera']['mtx'], 'umtx': calib_data['camera']['umtx'], 'dist': calib_data['camera']['dist']}
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            dataset = DPCalloader(self.calib_data, self.hparams.model_cfg, self.hparams.LDS_cfg, transform=self.transform, precision=self.precision)
            # train_val_split = [int(len(dataset) * split) for split in self.hparams.train_val_split]
            # train_val_split[-1] = len(dataset) - sum(train_val_split[:-1])
            # self.data_train, self.data_val = random_split(
            #     dataset=dataset,
            #     lengths=train_val_split,
            #     generator=torch.Generator().manual_seed(42),
            # )
            self.data_train, self.data_val = dataset, dataset
        
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )