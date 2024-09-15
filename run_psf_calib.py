
import os
import hydra
import pyrootutils
from omegaconf import DictConfig
from src.calib import CalibBoard
from src.camera import CameraObject
from src.utils.io import read_observations, read_lidar_observations

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True) 

@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="psfcalib")
def main(cfg: DictConfig):
    
    # define camera object
    camera_obj = CameraObject(cfg)
    
    if 'camera' in camera_obj.calib_data:
        
        # prepare data for PSF calibration
        if not 'training_data' in camera_obj.calib_data:
            
            # define calibration board
            calib_board = CalibBoard(cfg)
    
            # read observations
            observations = read_observations(os.path.join(cfg.paths.psf_data_dir, cfg.calib.psfboard.path_rgb), 
                                             'psf', cfg.calib.observation.scale, [cfg.calib.psfboard.target_focal, 16.0])
            
            # prepare patches for PSF calibration
            camera_obj.preparePatches(calib_board, observations)
            
        # train PSF calibration model
        model = camera_obj.trainPSFCalibModel()
        
        # evaluate PSF calibration model with lidar depth (optional)
        if ('lidar' in camera_obj.calib_data) and cfg.evaluate_on_lidar and (cfg.calib.psfboard.path_lidar is not None):
            observations_lidar = read_lidar_observations(os.path.join(cfg.paths.psf_data_dir, cfg.calib.psfboard.path_lidar), 'sample')
            # import pdb; pdb.set_trace()
        
        # naive code : should be changed later
        import cv2
        depth_name = ['010', '040', '080', '090']
        for i, img_name in enumerate(['0101', '0102', '0103', '0104']):
            imglg = cv2.imread(f"/workspace/dataset/dual-pixel-defocus-disparity/Quantitative/{img_name}_L.png")
            imgrg = cv2.imread(f"/workspace/dataset/dual-pixel-defocus-disparity/Quantitative/{img_name}_R.png")
            imgcg = cv2.imread(f"/workspace/dataset/dual-pixel-defocus-disparity/Quantitative/{img_name}_B.png")
            depth = cv2.imread(f"/workspace/dataset/ICCP2020_DP_dataset_processed/DEPTH/{depth_name[i]}.TIF", cv2.IMREAD_GRAYSCALE)
            # result = cv2.imread(f"/workspace/dataset/dual-pixel-defocus-disparity/results/{img_name}.png")
            camera_obj.estimateDepthFromPSF(model, imglg, imgrg, imgcg, img_name, gt_depth=depth, level=64, pad=5, resize_val=1.0)
            # camera_obj.estimateDepthFromPSF_2stage(model, imglg, imgrg, imgcg, img_name, gt_depth=depth, coarse_level=31, fine_level=7, pad=110, resize_val=0.5)
        
        # imglg = cv2.imread(f"/workspace/dataset/testsample/DSC_0019_LEFT.TIF")
        # imgrg = cv2.imread(f"/workspace/dataset/testsample/DSC_0019_RIGHT.TIF")
        # camera_obj.estimateDepthFromPSF(model, imglg, imgrg, None, 'DSC_0019', level=41, pad=10, resize_val=1.0)
        # camera_obj.estimateDepthFromPSF_2stage(model, imglg, imgrg, None, 'DSC_0019', coarse_level=41, pad=10, resize_val=1.0)
            
        # for i, img_name in enumerate(['01', '02', '03', '04', '05']):
        #     imglg = cv2.imread(f"/workspace/dataset/sample_0524/{img_name}_L.TIF")
        #     imgrg = cv2.imread(f"/workspace/dataset/sample_0524/{img_name}_R.TIF")
        #     # camera_obj.estimateDepthFromPSF(model, imglg, imgrg, None, img_name, level=41, pad=10, resize_val=1.0)
        #     camera_obj.estimateDepthFromPSF_2stage(model, imglg, imgrg, None, img_name, coarse_level=21, pad=5, resize_val=1.0)
    else:
        print('Intrinsic calibration is required before PSF calibration')


if __name__ == '__main__':
    main()