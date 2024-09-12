
import os
import hydra
import pyrootutils
from omegaconf import DictConfig
from src.calib import CalibBoard
from src.camera import CameraObject
from src.utils.io import read_observations, read_lidar_observations

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True) 

@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="intcalib")
def main(cfg: DictConfig):
    
    # define camera object
    camera_obj = CameraObject(cfg)
        
    # define calibration board
    calib_board = CalibBoard(cfg)
    
    # read observations
    observations = read_observations(os.path.join(cfg.paths.calib_data_dir, cfg.calib.intboard.path_rgb), 
                                        'calib', cfg.calib.observation.scale)
    if cfg.calib.intboard.path_lidar is not None:
        observations_lidar = read_lidar_observations(os.path.join(cfg.paths.calib_data_dir, cfg.calib.intboard.path_lidar), 'calib')
    else:
        observations_lidar = None
    
    # run intrinsic / extrinsic calibration
    camera_obj.runIntrinsicCalib(calib_board, observations)
    if observations_lidar is not None:
        camera_obj.runIntrinsicCalib(calib_board, observations_lidar, window_size=9)
        camera_obj.runExtrinsicCalib(calib_board, observations, observations_lidar)


if __name__ == '__main__':
    main()