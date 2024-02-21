

import cv2
import math
import torch
import pickle
from pathlib import Path


# class to store camera related parameters
class CameraObject(object):
    
    def __init__(self, opts, load_path=None):
        
        self.opts = opts
        self.tag = int(10 * self.opts.calib.psfboard.target_focal)
        
        # camera parameters
        self.calib_data = self.loadCalibdata(load_path)
    
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
        import pdb; pdb.set_trace()
    
    
    
    