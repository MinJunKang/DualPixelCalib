
import os
import cv2
import hydra
import pyrootutils
from pathlib import Path
from omegaconf import DictConfig
from src.camera import CameraObject

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True) 

@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="psfcalib")
def main(cfg: DictConfig):
    
    # define camera object
    camera_obj = CameraObject(cfg)
    
    if 'camera' in camera_obj.calib_data:
            
        # load trained PSF calibration model
        model = camera_obj.trainPSFCalibModel()
        
        # evaluation on benchmark
        public_benchmark_dir = Path('/workspace/dataset/ICCP2020_DP_dataset_processed')
        imglp = sorted([str(p) for p in (public_benchmark_dir / 'LEFT').glob('*.jpg')])
        imgrp = sorted([str(p) for p in (public_benchmark_dir / 'RIGHT').glob('*.jpg')])
        depthp = sorted([str(p) for p in (public_benchmark_dir / 'DEPTH').glob('*.TIF')])
        for imgl, imgr, depth in zip(imglp, imgrp, depthp):
            imglg = cv2.imread(imgl)
            imgrg = cv2.imread(imgr)
            depth = cv2.imread(depth, cv2.IMREAD_GRAYSCALE)
            img_name = os.path.basename(imgl).split('.')[0]
            camera_obj.estimateDepthFromPSF(model, imglg, imgrg, img_name, gt_depth=depth, resize_val=0.5)
    else:
        print('Intrinsic calibration is required before PSF calibration')


if __name__ == '__main__':
    main()