
import hydra
import pyrootutils
from omegaconf import DictConfig
from src.calib import CalibBoard
from src.camera import CameraObject
from src.utils.io import read_observations

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
        model = camera_obj.trainPSFCalibModel()
        
        import cv2
        imglg = cv2.imread("/workspace/dataset/ICCP2020_DP_dataset_processed/LEFT/010.jpg")
        imgrg = cv2.imread("/workspace/dataset/ICCP2020_DP_dataset_processed/RIGHT/010.jpg")
        imgcg = cv2.imread("/workspace/dataset/ICCP2020_DP_dataset_processed/LRSUM/010.jpg")
        camera_obj.estimateDepthFromPSF(model, imglg, imgrg, imgcg, level=32, pad=60)


if __name__ == '__main__':
    main()