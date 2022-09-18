
import pdb
import argparse
import warnings
warnings.filterwarnings("ignore")

from src.calib.psflearner import PSFLearner
from src.calib.detector import CalibBoard
from src.calib.calib import Calibration
from src.calib.utils.configure import Configuration

from pytorch_lightning import seed_everything

# parsing arguments
parser = argparse.ArgumentParser(description='Configuration : DualPixel Calibration')
parser.add_argument('--calibname', type=str, required=True, help='calibration dataset path')
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--config', type=str, default='config', required=False, help='model config path')
parser.add_argument('--mode', type=str, default='train', help='train or eval')
parser.add_argument('--no_verbose', action='store_false', help='verbose option')
parser.add_argument('--load_ckpt_name', type=str, default=None, help='ckpt to load (works only for resume or eval mode)')

args = parser.parse_args()
optioner = Configuration(args)
opt = optioner.get_config()


def main():

    # seed initialize : for reproducibility
    seed_everything(0)

    # read imgs and detect board (Initial)
    board = CalibBoard(opt)
    raw, data, if_update = board.detectBoard()
    
    # intrinsic calibrate of camera
    calib = Calibration(opt, board)
    data, if_update = calib.single_calib(raw, data, if_update, is_final=False)
    
    # propagate points based on initial calibration
    data, if_update = board.propagate_corners(raw, data, if_update)
    
    # run refined intrinsic calibration of camera
    data, if_update = calib.single_calib(raw, data, if_update, is_final=True)
    
    # stereo calibration between (L + R) and lidar
    data, if_update = calib.stereo_calib(raw, data, if_update)
        
    # Find actual focus distance g
    data = calib.estimate_gvalue(raw, data, if_update, visualize_graph=False)  # -> get gvalue from focal stack
    
    # Prepare patches for learning PSF 
    patches = calib.prepare_patches(raw, data, if_update=if_update)
    
    # Calibrate PSF Volume
    psflearner = PSFLearner(data[board.device], opt, board)
    psflearner.train(patches, ckpt=opt.load_ckpt_name)
    pdb.set_trace()


if __name__ == '__main__':
    main()

    