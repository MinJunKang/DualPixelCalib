
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
        model = camera_obj.trainPSFCalibModel(vis_result=True)
        
        # evaluate PSF calibration model with lidar depth (optional)
        if ('lidar' in camera_obj.calib_data) and cfg.evaluate_on_lidar and (cfg.calib.psfboard.path_lidar is not None):
            observations_lidar = read_lidar_observations(os.path.join(cfg.paths.psf_data_dir, cfg.calib.psfboard.path_lidar), 'sample')
            # import pdb; pdb.set_trace()
    else:
        print('Intrinsic calibration is required before PSF calibration')


if __name__ == '__main__':
    main()