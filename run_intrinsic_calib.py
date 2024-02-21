
import hydra
import pyrootutils
from omegaconf import DictConfig
from src.calib import CalibBoard
from src.camera import CameraObject
from src.utils.io import read_observations

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True) 

@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="intcalib")
def main(cfg: DictConfig):
    
    # define camera object
    camera_obj = CameraObject(cfg, load_path=cfg.paths.output_dir)
    
    if not 'camera' in camera_obj.calib_data:
        
        # define calibration board
        calib_board = CalibBoard(cfg)
        
        # read observations
        observations = read_observations(cfg.paths.calib_data_dir, 'calib', cfg.calib.observation.scale)
        
        # run intrinsic calibration
        camera_obj.runIntrinsicCalib(calib_board, observations)


if __name__ == '__main__':
    main()