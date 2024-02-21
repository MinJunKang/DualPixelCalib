

import cv2
import math
import copy
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from createBoard import create_full_board
from createBoard_psf import create_full_board as create_full_board_psf
from src.utils.io import img2gray
from src.utils.calib_utils import SubPixRefinement
import src.utils.math as geomath
from einops import repeat, rearrange


# class to store calibration board parameters
class CalibBoard(object):
    
    def __init__(self, opts, verbose=False):
        
        # parameters
        self.opts = opts
        self.verbose = verbose
        self.calib_board = self.read_board(opts.paths.board_dir, 'calib', opts.calib.intboard.board_name)
        self.psf_board = self.read_board(opts.paths.board_dir, 'psf', opts.calib.psfboard.board_name)
        
        # fnumber lookup table
        self.fnumber_table = {'1.2': 0.7, '1.4': 1, '1.6': 1.3, '1.7': 1.5, '1.8': 1.7, 
                              '2.0': 2, '2.2': 2.3, '2.5': 2.7, '2.8': 3, 
                              '3.2': 3.3, '3.5': 3.6, '4.0': 4, '4.5': 4.3, 
                              '5.0': 4.7, '5.6': 5, '6.3': 5.3, '7.1': 5.7,
                              '8.0': 6, '9.0': 6.3, '10.0': 6.7, '11.0': 7, 
                              '13.0': 7.3, '14.0': 7.7, '16.0': 8, '18.0': 8.3, 
                              '20.0': 8.7}
        
    def read_board(self, board_path, board_type, board_name=None, scales=[1.0, 0.5, 2.0]):
        assert board_type in ['calib', 'psf']
        if board_name is None: return None
        board_path_full = Path(board_path) / board_name
        board_info = np.load(board_path_full / 'board_info.npy', allow_pickle=True).item()
        
        if board_type == 'calib':
            return create_full_board(board_info, self.opts.calib.intboard.board_size_mm, 1.0)
        else:
            # get multiscale board
            board_multiscale = []
            for scale in scales:
                board_info_scale = copy.deepcopy(board_info)
                board_multiscale.append(create_full_board_psf(board_info_scale, self.opts.calib.psfboard.board_size_mm, scale))
            return board_multiscale
        
    def get_numerical_fnumber(self, fnumber):
        # convert fnumber to float
        return np.sqrt(np.power(2, self.fnumber_table[str(fnumber)]))
        
    def detectCalibBoard(self, observations: dict, window_size=34, min_corner_num=6):
        assert observations['format'] == 'calib'
        
        # start detection
        observations['corners'] = []
        charucodetector = cv2.aruco.CharucoDetector(self.calib_board['board'])
        for image in tqdm(observations['images']):
            image = img2gray(image)
            charuco_corners, charuco_ids, _, _ = charucodetector.detectBoard(image)
            charuco_corners = np.squeeze(charuco_corners)
            corner_ids = np.squeeze(charuco_ids)
            
            if len(charuco_corners) >= min_corner_num:
                # subpixel refinement of corners
                refined_corners = SubPixRefinement(image, charuco_corners, window_size=window_size)
                inf_mask = (refined_corners[:, :2] == np.Infinity) | (refined_corners[:, :2] == np.NaN)
                refined_corners[:, :2][inf_mask] = 0.0
                refined_corners[:, :2] = ~inf_mask * refined_corners[:, :2] + inf_mask * charuco_corners
                
                # convert corner_ids to our convention
                corner_ids = (self.calib_board['num_x'] - 1) * (self.calib_board['num_y'] - 1) - 1 - corner_ids
                for i in range(len(corner_ids)):
                    rows = corner_ids[i] // (self.calib_board['num_x'] - 1)
                    cols = corner_ids[i] % (self.calib_board['num_x'] - 1)
                    corner_ids[i] = (self.calib_board['num_x'] - 1) * rows + (self.calib_board['num_x'] - 2) - cols
                multiplier = (np.floor(corner_ids / (self.calib_board['num_x'] - 1)) + 1) * (self.calib_board['num_x'] + 1)
                offset = np.mod(corner_ids, (self.calib_board['num_x'] - 1)) + 1
                corner_ids = multiplier.astype('uint64') + offset
                
                usage = True
            else:
                refined_corners = None
                corner_ids = None
                usage = False
            corner_info = {'corner_ids': corner_ids, 'refined_corners': refined_corners, 'usage': usage, 'winsize': window_size}
            observations['corners'].append(corner_info)
        
        return observations
    
    def runSingleCalib(self, observations: dict):
        
        calib_results = {'mtx': None, 'umtx': None, 'dist': None}
        per_scene_results = {'tvecs': [], 'rvecs': [], 'x': [], 'X': [], 'idx': []}
        
        # find pairs in observations
        world_pts = np.reshape(self.calib_board['rec3d_corners'], [-1, 3])
        for i, corner in enumerate(observations['corners']):
            if corner['usage']:
                corner_ids = corner['corner_ids'].astype('uint64')
                per_scene_results['x'].append(corner['refined_corners'][:, :2].astype('float32'))
                per_scene_results['X'].append(world_pts[corner_ids].astype('float32'))
                per_scene_results['idx'].append(i)
        
        # run calibration
        h, w = observations['image_size']
        numview = len(per_scene_results['idx'])
        print('intrinsic_calibrate view camera of (%d/%d) views' % (numview, len(observations['corners'])))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(per_scene_results['X'], per_scene_results['x'], (w, h), None, None)
        umtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        calib_results['mtx'] = mtx
        calib_results['umtx'] = umtx
        calib_results['dist'] = dist
        per_scene_results['tvecs'] = tvecs
        per_scene_results['rvecs'] = rvecs
        
        # get reprojection error
        reproj_err = []
        for i in range(numview):
            reproj, _ = cv2.projectPoints(per_scene_results['X'][i], rvecs[i], tvecs[i], mtx, dist)
            reproj_err.append(np.mean(np.linalg.norm(np.squeeze(reproj) - per_scene_results['x'][i], axis=1)))
        per_scene_results['ret'] = reproj_err
        
        return calib_results, per_scene_results
    
    def propagateCorners(self, observations: dict, calib_results: dict, per_scene_results: dict, px_threshold=1.6):
        
        world_pts = np.reshape(self.calib_board['rec3d_corners'], [-1, 3])
        world_pts_idx = np.array(range(len(world_pts)))
        for (tvec, rvec, idx, ret) in zip(per_scene_results['tvecs'], per_scene_results['rvecs'], per_scene_results['idx'], per_scene_results['ret']):
            if per_scene_results['ret'][idx] < px_threshold:
                
                # get propagated corners
                new_corners, _ = cv2.projectPoints(world_pts, rvec, tvec, calib_results['mtx'], calib_results['dist'])
                new_corners = np.squeeze(new_corners)
                refined_corners = SubPixRefinement(observations['images'][idx], new_corners, window_size=observations['corners'][idx]['winsize'])
                inf_mask = (refined_corners[:, :2] == np.Infinity) | (refined_corners[:, :2] == np.NaN)
                refined_corners[:, :2][inf_mask] = 0.0
                refined_corners[:, :2] = ~inf_mask * refined_corners[:, :2] + inf_mask * new_corners
                
                # update observations
                observations['corners'][idx]['refined_corners'] = refined_corners
                observations['corners'][idx]['corner_ids'] = world_pts_idx
                observations['corners'][idx]['usage'] = True
            else:
                observations['corners'][idx]['usage'] = False
        
        return observations
    
    def detectPSFBoard(self, observations: dict, calib_results: dict, min_corner_num=4):
        assert observations['format'] == 'psf'
        
        # start detection
        arucodetector = cv2.aruco.ArucoDetector(self.psf_board[0]['dictionary'])
        worldpoints_id = self.psf_board[0]['obj_corners_id']
        for idx, sample in enumerate(observations['samples']):
            observations['samples'][idx]['extrinsics'] = {}
            target_idx = sample['focals'].index(16.0)
            image = img2gray(sample['images_l'][target_idx])
            corners, marker_idx, rejected = arucodetector.detectMarkers(image)
            corners = np.squeeze(corners)
            marker_idx = np.squeeze(marker_idx)
            if len(corners) >= min_corner_num:
                cidx, idx1, idx2 = np.intersect1d(worldpoints_id, marker_idx, return_indices=True)
                corners, marker_idx = corners[idx2], marker_idx[idx2]
                
                observations['samples'][idx]['extrinsics']['rvec'] = []
                observations['samples'][idx]['extrinsics']['tvec'] = []
                for psfboard in self.psf_board:
                    # solve pnps
                    _, rvec, tvec = cv2.solvePnP(psfboard['obj_corners_3d'][idx1].reshape(-1, 3), corners.reshape(-1, 2), calib_results['mtx'], calib_results['dist'])
                    observations['samples'][idx]['extrinsics']['rvec'].append(rvec)
                    observations['samples'][idx]['extrinsics']['tvec'].append(tvec)
                observations['samples'][idx]['extrinsics']['usage'] = True
            else:
                observations['samples'][idx]['extrinsics']['usage'] = False
        return observations
    
    def extractPatches(self, observations: dict, calib_results: dict):
        
        # initialize data
        mask_patch_valid = []
        training_data = {'clean_patch': [], 'noisy_patch': [], 'lpatch': [], 'rpatch': [], 'patch_3d': [], 'patch_uv': [], 'patchSize': [], 'rvec': [], 'tvec': []}
        for psfboard in self.psf_board:
            circenters = psfboard['cir2d'].reshape(-1, 3)
            patchSize = round((psfboard['size_px'] / psfboard['num_grid'] * self.opts.calib.psfboard.patch_size_area) / self.opts.calib.psfboard.patch_size_mul) * self.opts.calib.psfboard.patch_size_mul
            training_data['patchSize'].append(patchSize)
            training_data['clean_patch'].append(np.zeros([len(observations['samples']), len(circenters), patchSize, patchSize], dtype=np.float32))
            training_data['noisy_patch'].append(np.zeros([len(observations['samples']), len(circenters), patchSize, patchSize], dtype=np.float32))
            training_data['lpatch'].append(np.zeros([len(observations['samples']), len(circenters), patchSize, patchSize], dtype=np.float32))
            training_data['rpatch'].append(np.zeros([len(observations['samples']), len(circenters), patchSize, patchSize], dtype=np.float32))
            training_data['patch_3d'].append(np.zeros([len(observations['samples']), len(circenters), patchSize, patchSize, 3], dtype=np.float32))
            training_data['patch_uv'].append(np.zeros([len(observations['samples']), len(circenters), patchSize, patchSize, 2], dtype=np.float32))
            training_data['rvec'].append(np.zeros([len(observations['samples']), len(circenters), 3, 1], dtype=np.float32))
            training_data['tvec'].append(np.zeros([len(observations['samples']), len(circenters), 3, 1], dtype=np.float32))
            mask_patch_valid.append(np.zeros([len(observations['samples']), len(circenters)], dtype=bool))
        mtx, umtx, dist = calib_results['mtx'], calib_results['umtx'], calib_results['dist']
        
        # start extraction
        blobParams = self.defineBlobDetector()
        target_focal = self.opts.calib.psfboard.target_focal
        for nidx, sample in enumerate(tqdm(observations['samples'])):
            aif_image = img2gray(sample['images_l'][sample['focals'].index(16.0)])
            image_l = img2gray(sample['images_l'][sample['focals'].index(target_focal)])
            image_r = img2gray(sample['images_r'][sample['focals'].index(target_focal)])
            if sample['extrinsics']['usage']:
                for idx, psfboard in enumerate(self.psf_board):
                    rvec, tvec = sample['extrinsics']['rvec'][idx], sample['extrinsics']['tvec'][idx]
                    
                    # template 3d coordinates
                    h_t, w_t = psfboard['template'].shape
                    uv, vv = np.meshgrid(np.linspace(0, w_t - 1, w_t), np.linspace(0, h_t - 1, h_t))
                    template_3d = geomath.convert2homography(np.stack([uv, vv], axis=-1), 0.0)
                    template_3d *= (self.opts.calib.psfboard.board_size_mm / psfboard['size_px'])
                    
                    # projected image
                    rotz = geomath.findrotz_euler(rvec)
                    homo = geomath.findhomography(rvec, tvec, umtx, psfboard['size_px'], psfboard['size_px'], self.opts.calib.psfboard.board_size_mm)
                    invhomo = np.linalg.inv(homo)
                    aif_pimg = cv2.warpPerspective(cv2.undistort(aif_image, mtx, dist, None, umtx), invhomo, (w_t, h_t))
                    pimg_l = cv2.warpPerspective(cv2.undistort(image_l, mtx, dist, None, umtx), invhomo, (w_t, h_t))
                    pimg_r = cv2.warpPerspective(cv2.undistort(image_r, mtx, dist, None, umtx), invhomo, (w_t, h_t))
                    
                    # patchify all_in_focus image using predefined coordinate
                    circenters, patchsize = psfboard['cir2d'].reshape(-1, 3), training_data['patchSize'][idx]
                    aif_patches = geomath.subpixel_cropper_batch(np.float32(aif_pimg), circenters, patchsize, mode='bilinear')
                    clean_patches = geomath.subpixel_cropper_batch(np.float32(psfboard['template']), circenters, patchsize, mode='bilinear')
                    
                    # detect circle and find fine-grained positions
                    offsets, valid_mask = self.optimizePatches(aif_patches, blobParams, psfboard['radius'].reshape(-1))
                    
                    mask_patch_valid[idx][nidx] = valid_mask
                    if valid_mask.any():
                        # using refined positions, get patches of left and right image
                        circenters_refined = circenters[:, :2] + offsets
                        patches_l = geomath.subpixel_cropper_batch(np.float32(pimg_l), circenters_refined, patchsize, mode='bilinear')
                        patches_r = geomath.subpixel_cropper_batch(np.float32(pimg_r), circenters_refined, patchsize, mode='bilinear')
                        patches_c = geomath.subpixel_cropper_batch(np.float32(aif_pimg), circenters_refined, patchsize, mode='bilinear')
                        coords_3d = geomath.get_3d_points(template_3d.reshape(-1, 3), rvec, tvec).reshape(h_t, w_t, 3)  # xyz
                        patches_x = geomath.subpixel_cropper_batch(coords_3d[..., 0], circenters_refined, patchsize)
                        patches_y = geomath.subpixel_cropper_batch(coords_3d[..., 1], circenters_refined, patchsize)
                        patches_z = geomath.subpixel_cropper_batch(coords_3d[..., 2], circenters_refined, patchsize)
                        patches_3d = np.stack([patches_x, patches_y, patches_z], axis=-1)
                        prj_uv, _ = cv2.projectPoints(template_3d.reshape(-1, 3), rvec, tvec, mtx, dist)  # uv coords
                        prj_uv = prj_uv.reshape([h_t, w_t, 2])
                        patches_u = geomath.subpixel_cropper_batch(prj_uv[..., 0], circenters_refined, patchsize)
                        patches_v = geomath.subpixel_cropper_batch(prj_uv[..., 1], circenters_refined, patchsize)
                        patches_uv = np.stack([patches_u, patches_v], axis=-1)
                        
                        # save data
                        training_data['rvec'][idx][nidx] = repeat(rvec, 'p q -> n p q', n=len(circenters))
                        training_data['tvec'][idx][nidx] = repeat(tvec, 'p q -> n p q', n=len(circenters))
                        training_data['clean_patch'][idx][nidx] = clean_patches
                        training_data['noisy_patch'][idx][nidx] = patches_c
                        training_data['lpatch'][idx][nidx] = patches_l
                        training_data['rpatch'][idx][nidx] = patches_r
                        training_data['patch_3d'][idx][nidx] = patches_3d
                        training_data['patch_uv'][idx][nidx] = patches_uv
        
        # aggregate data
        for key in training_data.keys():
            if key == 'patchSize':
                continue
            for idx, element in enumerate(training_data[key]):
                training_data[key][idx] = element[mask_patch_valid[idx]]
            
        return training_data
        
    def defineBlobDetector(self):
        # Setup SimpleBlobDetector parameters.
        blobParams = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        blobParams.minThreshold = int(255 * (1.0 - self.opts.calib.blobParams.maxthres))
        blobParams.maxThreshold = int(255 * (1.0 - self.opts.calib.blobParams.minthres))

        # Filter by Area.
        blobParams.filterByArea = self.opts.calib.blobParams.filterByArea

        # Filter by Circularity
        blobParams.filterByCircularity = self.opts.calib.blobParams.filterByCircularity
        blobParams.minCircularity = self.opts.calib.blobParams.minCircularity

        # Filter by Convexity
        blobParams.filterByConvexity = self.opts.calib.blobParams.filterByConvexity
        blobParams.minConvexity = self.opts.calib.blobParams.minConvexity

        # Filter by Inertia
        blobParams.filterByInertia = self.opts.calib.blobParams.filterByInertia
        blobParams.minInertiaRatio = self.opts.calib.blobParams.minInertiaRatio
        
        return blobParams
    
    def detectConic(self, patch, blobParams, minsize):
        
        blobParams.minArea = int(np.round(minsize))
        blobDetector = cv2.SimpleBlobDetector_create(blobParams)
        
        mask = patch > self.opts.calib.blobParams.minthres
        masked_patch = (255 - patch).astype('uint8') * mask
        keypoints = blobDetector.detect(masked_patch)
        
        if len(keypoints) > 1:  # too many circles are detected
            # masking (condition)
            metric_size = np.array([abs(key.size - minsize) for key in keypoints])
            metric_dist = np.array([np.linalg.norm(key.pt - np.array(patch.shape) / 2) for key in keypoints])
            keypoints = [keypoints[np.argmin(metric_size * metric_dist)]]
            
        if len(keypoints) == 0:
            return np.array(patch.shape) / 2, False
        else:
            return keypoints[0].pt, True
    
    # from detected points in PSF board of all-in-focus images, optimize to find points in blurred images 
    def optimizePatches(self, patches, blobParams, radius):
        num_patches = len(patches)
        offset = np.zeros((num_patches, 2))
        valid = np.zeros(num_patches, dtype=bool)
        for i in range(num_patches):
            # detect circle
            center_pos, valid[i] = self.detectConic(patches[i], blobParams, radius[i])
            offset[i] = center_pos - np.array(patches[i].shape) / 2  
        return offset, valid
    
    