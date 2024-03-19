
import hydra
import pyrootutils
from omegaconf import DictConfig
from src.calib import CalibBoard
from src.camera import CameraObject
from src.utils.io import read_observations
import cv2
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True) 

@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="psfcalib")
def main(cfg: DictConfig):
    
    # define camera object
    camera_obj = CameraObject(cfg, load_path=cfg.paths.output_dir)
    
    if 'camera' in camera_obj.calib_data:
        
        # prepare data for PSF calibration
        if not 'training_data' in camera_obj.calib_data:
            
            # define calibration board
            calib_board = CalibBoard(cfg)
    
            # read observations
            observations = read_observations(cfg.paths.psf_data_dir, 'psf', cfg.calib.observation.scale)
            
            # prepare patches for PSF calibration
            camera_obj.preparePatches(calib_board, observations)
            
        # train PSF calibration model
        camera_obj.trainPSFCalibModel()

        imglg = cv2.imread("./data/Punnappurath_ICCP_2020/010_L.jpg")
        imgrg = cv2.imread("./data/Punnappurath_ICCP_2020/010_R.jpg")
        imgcg = cv2.imread("./data/Punnappurath_ICCP_2020/010_B.jpg")
        
        camera_obj.estimateDepthFromPSF(imglg, imgrg, imgcg, level=255, splits=[8, 10], pad=60)

if __name__ == '__main__':
    main()