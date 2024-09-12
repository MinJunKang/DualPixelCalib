

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
        
    def read_board(self, board_path, board_type, board_name=None):
        assert board_type in ['calib', 'psf']
        if board_name is None: return None
        board_path_full = Path(board_path) / board_name
        board_info = np.load(board_path_full / 'board_info.npy', allow_pickle=True).item()
        
        if board_type == 'calib':
            return create_full_board(board_info, self.opts.calib.intboard.board_size_mm, 1.0)
        else:
            board_info_scale = copy.deepcopy(board_info)   # in case for multiple scales
            return create_full_board_psf(board_info_scale, self.opts.calib.psfboard.board_size_mm, self.opts.calib.psfboard.template_scale)
        
    def get_numerical_fnumber(self, fnumber):
        # convert fnumber to float
        return np.sqrt(np.power(2, self.fnumber_table[str(fnumber)]))
        
    def detectCalibBoard(self, observations: dict, window_size=34, min_corner_num=16):
        assert observations['format'] == 'calib'
        
        # start detection
        observations['corners'] = []
        charucodetector = cv2.aruco.CharucoDetector(self.calib_board['board'])
        for image in tqdm(observations['images']):
            image = img2gray(image)
            charuco_corners, charuco_ids, _, _ = charucodetector.detectBoard(image)
            charuco_corners = np.squeeze(charuco_corners)
            corner_ids = np.squeeze(charuco_ids)
            
            try:
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
            except:
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
        calib_results['image_size'] = (h, w)
        per_scene_results['tvecs'] = tvecs
        per_scene_results['rvecs'] = rvecs
        
        # get reprojection error
        reproj_err = []
        for i in range(numview):
            reproj, _ = cv2.projectPoints(per_scene_results['X'][i], rvecs[i], tvecs[i], mtx, dist)
            reproj_err.append(np.mean(np.linalg.norm(np.squeeze(reproj) - per_scene_results['x'][i], axis=1)))
        per_scene_results['ret'] = reproj_err
        
        return calib_results, per_scene_results
    
    def runStereoCalib(self, observations_1: dict, observations_2: dict, intrinsics: dict):
        assert 'corners' in observations_1 and 'corners' in observations_2, 'Run intrinsic calib first !!'
        assert len(observations_1['images']) == len(observations_2['images']), 'Number of images do not match !!'
        
        # load intrinsics
        calib_data_1 = intrinsics[observations_1['device']]
        calib_data_2 = intrinsics[observations_2['device']]
        mtx1, dist1 = np.float32(calib_data_1['mtx']), np.float32(calib_data_1['dist'])
        mtx2, dist2 = np.float32(calib_data_2['mtx']), np.float32(calib_data_2['dist'])
        mtx1_l, mtx2_l = np.copy(mtx1), np.copy(mtx2)
        
        # resolution match
        image_height = min(calib_data_1['image_size'][0], calib_data_2['image_size'][0])
        image_width = min(calib_data_1['image_size'][1], calib_data_2['image_size'][1])
        ratio_m1 = np.diagflat([image_width / calib_data_1['image_size'][1], image_height / calib_data_1['image_size'][0], 1.0])
        ratio_m2 = np.diagflat([image_width / calib_data_2['image_size'][1], image_height / calib_data_2['image_size'][0], 1.0])
        mtx1_l = np.matmul(ratio_m1, mtx1_l)
        mtx2_l = np.matmul(ratio_m2, mtx2_l)
        
        # get paired 2d-3d from aggregate intersected views
        pairs_info = {'x1': [], 'x2': [], 'X': [], 'd1': [], 'd2': []}
        world_pts = np.float32(np.reshape(self.calib_board['rec3d_corners'], [-1, 3]))
        for corner1, corner2 in zip(observations_1['corners'], observations_2['corners']):
            if corner1['usage'] and corner2['usage']:
                pairs_info['x1'].append(corner1['refined_corners'][:, :2] * np.diag(ratio_m1)[None, :2])
                pairs_info['x2'].append(corner2['refined_corners'][:, :2] * np.diag(ratio_m2)[None, :2])
                pairs_info['X'].append(world_pts)
                pairs_info['d1'].append(geomath.get_plane_depth(np.float64(world_pts), np.float64(pairs_info['x1'][-1]), corner1['rvec'], corner1['tvec'], mtx1_l, dist1))
                pairs_info['d2'].append(geomath.get_plane_depth(np.float64(world_pts), np.float64(pairs_info['x2'][-1]), corner2['rvec'], corner2['tvec'], mtx2_l, dist2))
        X = np.float32(np.stack(pairs_info['X'], axis=0))
        x1 = np.float32(np.stack(pairs_info['x1'], axis=0))
        x2 = np.float32(np.stack(pairs_info['x2'], axis=0))
            
        # run stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(X, x1, x2, mtx1_l, dist1, mtx2_l, dist2, None, criteria=criteria_stereo, flags=flags)
        '''
            # Geometry Explanation
            R2 = RR1,    T2 = RT1 + T
            Rmat_c = cv2.Rodrigues(rvec_c)[0]
            Rmat_l = cv2.Rodrigues(rvec_l)[0]
            Rot_ = np.matmul(Rmat_l, np.linalg.inv(Rmat_c))
            T_ = tvec_l - np.matmul(Rot_, tvec_c)
        '''
        
        # get projective matrix
        Pmat = np.concatenate([np.concatenate([R, T], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
        invPmat = np.linalg.inv(Pmat)
        
        # measure reprojection error
        err_1, err_2 = [], []
        for x1, x2, d1, d2 in zip(pairs_info['x1'], pairs_info['x2'], pairs_info['d1'], pairs_info['d2']):
            x1_o = x1 / np.diag(ratio_m1)[None, :2]
            x2_o = x2 / np.diag(ratio_m2)[None, :2]
            p1 = cv2.undistortPoints(x1_o, mtx1, dist1)
            p2 = cv2.undistortPoints(x2_o, mtx2, dist2)
            up1 = geomath.convert2homography(np.squeeze(p1), 1.0) * d1
            up2 = geomath.convert2homography(np.squeeze(p2), 1.0) * d2
            rp2, _ = cv2.projectPoints(up1, cv2.Rodrigues(Pmat[:3, :3])[0], Pmat[:3, 3:], mtx2, dist2)
            rp1, _ = cv2.projectPoints(up2, cv2.Rodrigues(invPmat[:3, :3])[0], invPmat[:3, 3:], mtx1, dist1)
            err_1.append(cv2.norm(x1_o, np.squeeze(rp1), cv2.NORM_L2) / len(p1))
            err_2.append(cv2.norm(x2_o, np.squeeze(rp2), cv2.NORM_L2) / len(p2))
        err_1 = sum(err_1) / len(err_1)
        err_2 = sum(err_2) / len(err_2)
        
        calib_results = {'Emat': E, 'Fmat': F, 'Pmat': Pmat, 'device1': observations_1['device'], 'device2': observations_2['device']}
        
        return calib_results, err_1, err_2
    
    def propagateCorners(self, observations: dict, calib_results: dict, per_scene_results: dict, px_threshold=1.5):
        
        world_pts = np.reshape(self.calib_board['rec3d_corners'], [-1, 3])
        world_pts_idx = np.array(range(len(world_pts)))
        for (tvec, rvec, idx, ret) in zip(per_scene_results['tvecs'], per_scene_results['rvecs'], per_scene_results['idx'], per_scene_results['ret']):
            try:
                if ret < px_threshold:
                    
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
            except:
                import pdb; pdb.set_trace()
        
        return observations
    
    def detectPSFBoard(self, observations: dict, calib_results: dict, min_corner_num=4):
        assert observations['format'] == 'psf'
        
        # start detection
        arucodetector = cv2.aruco.ArucoDetector(self.psf_board['dictionary'])
        worldpoints_id = self.psf_board['obj_corners_id']
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
                
                # solve pnps
                _, rvec, tvec = cv2.solvePnP(self.psf_board['obj_corners_3d'][idx1].reshape(-1, 3), corners.reshape(-1, 2), calib_results['mtx'], calib_results['dist'])
                observations['samples'][idx]['extrinsics']['rvec'] = rvec
                observations['samples'][idx]['extrinsics']['tvec'] = tvec
                observations['samples'][idx]['extrinsics']['usage'] = True
            else:
                observations['samples'][idx]['extrinsics']['usage'] = False
        return observations
    
    def extractPatches(self, observations: dict, calib_results: dict):
        assert observations['device'] == 'camera'
        
        # initialize data
        training_data = {}
        circenters = self.psf_board['cir2d'].reshape(-1, 3)
        patchSize_px = math.ceil((self.psf_board['size_px'] / self.psf_board['num_grid'] * self.opts.calib.psfboard.patch_area) / self.opts.calib.psfboard.patch_mul) * self.opts.calib.psfboard.patch_mul
        patchSize_mm = self.opts.calib.intboard.board_size_mm / (self.psf_board['size_px'] / self.psf_board['num_grid']) * patchSize_px
        training_data['patchSize_px'] = patchSize_px
        training_data['patchSize_mm'] = patchSize_mm
        training_data['clean_patch'] = np.zeros([len(observations['samples']), len(circenters), patchSize_px, patchSize_px, 3], dtype=np.float32)
        training_data['noisy_patch'] = np.zeros([len(observations['samples']), len(circenters), patchSize_px, patchSize_px, 3], dtype=np.float32)
        training_data['lpatch'] = np.zeros([len(observations['samples']), len(circenters), patchSize_px, patchSize_px, 3], dtype=np.float32)
        training_data['rpatch'] = np.zeros([len(observations['samples']), len(circenters), patchSize_px, patchSize_px, 3], dtype=np.float32)
        training_data['patch_3d'] = np.zeros([len(observations['samples']), len(circenters), patchSize_px, patchSize_px, 3], dtype=np.float32)
        training_data['patch_uv'] = np.zeros([len(observations['samples']), len(circenters), patchSize_px, patchSize_px, 2], dtype=np.float32)
        training_data['rvec'] = np.zeros([len(observations['samples']), len(circenters), 3, 1], dtype=np.float32)
        training_data['tvec'] = np.zeros([len(observations['samples']), len(circenters), 3, 1], dtype=np.float32)
        mask_patch_valid = np.zeros([len(observations['samples']), len(circenters)], dtype=bool)
        mtx, umtx, dist = calib_results['mtx'], calib_results['umtx'], calib_results['dist']
        
        # start extraction
        blobParams = self.defineBlobDetector()
        target_focal = self.opts.calib.psfboard.target_focal
        for nidx, sample in enumerate(tqdm(observations['samples'])):
            aif_image = sample['images_l'][sample['focals'].index(16.0)]
            image_l = sample['images_l'][sample['focals'].index(target_focal)]
            image_r = sample['images_r'][sample['focals'].index(target_focal)]
            if sample['extrinsics']['usage']:
                h_t, w_t = self.psf_board['template'].shape
                rvec, tvec = sample['extrinsics']['rvec'], sample['extrinsics']['tvec']
                
                # template 3d coordinates
                uv, vv = np.meshgrid(np.linspace(0, w_t - 1, w_t), np.linspace(0, h_t - 1, h_t))
                template_3d = geomath.convert2homography(np.stack([uv, vv], axis=-1), 0.0)
                template_3d *= (self.opts.calib.psfboard.board_size_mm / self.psf_board['size_px'])
                
                # projected image
                homo = geomath.findhomography(rvec, tvec, umtx, self.psf_board['size_px'], self.psf_board['size_px'], self.opts.calib.psfboard.board_size_mm)
                invhomo = np.linalg.inv(homo)
                aif_pimg = cv2.warpPerspective(cv2.undistort(aif_image, mtx, dist, None, umtx), invhomo, (w_t, h_t))
                pimg_l = cv2.warpPerspective(cv2.undistort(image_l, mtx, dist, None, umtx), invhomo, (w_t, h_t))
                pimg_r = cv2.warpPerspective(cv2.undistort(image_r, mtx, dist, None, umtx), invhomo, (w_t, h_t))
                
                # patchify all_in_focus image using predefined coordinate
                circenters, patchsize = self.psf_board['cir2d'].reshape(-1, 3), training_data['patchSize_px']
                aif_patches = geomath.subpixel_cropper_batch(np.float32(img2gray(aif_pimg))[..., None], circenters, patchsize, mode='bilinear')
                
                # detect circle and find fine-grained positions
                offsets, valid_mask = self.optimizePatches(np.squeeze(aif_patches), blobParams, self.psf_board['radius'].reshape(-1))
                mask_patch_valid[nidx] = valid_mask
                
                # store patch data
                if valid_mask.any():
                    # using refined positions, get patches of left and right image
                    circenters_refined = circenters[:, :2] + offsets
                    patches_l = geomath.subpixel_cropper_batch(np.float32(pimg_l), circenters_refined, patchsize, mode='bilinear')
                    patches_r = geomath.subpixel_cropper_batch(np.float32(pimg_r), circenters_refined, patchsize, mode='bilinear')
                    patches_c = geomath.subpixel_cropper_batch(np.float32(aif_pimg), circenters_refined, patchsize, mode='bilinear')
                    clean_patches = geomath.subpixel_cropper_batch(np.float32(repeat(self.psf_board['template'], 'h w -> h w 3')), circenters, patchsize, mode='bilinear')
                    coords_3d = geomath.get_3d_points(template_3d.reshape(-1, 3), rvec, tvec).reshape(h_t, w_t, 3)  # xyz coords
                    patches_3d = geomath.subpixel_cropper_batch(coords_3d, circenters_refined, patchsize)
                    prj_uv, _ = cv2.projectPoints(template_3d.reshape(-1, 3), rvec, tvec, mtx, dist)  # uv coords
                    patches_uv = geomath.subpixel_cropper_batch(prj_uv.reshape(h_t, w_t, 2), circenters_refined, patchsize)
                    
                    # save data
                    training_data['rvec'][nidx] = repeat(rvec, 'p q -> n p q', n=len(circenters))
                    training_data['tvec'][nidx] = repeat(tvec, 'p q -> n p q', n=len(circenters))
                    training_data['clean_patch'][nidx] = clean_patches
                    training_data['noisy_patch'][nidx] = patches_c
                    training_data['lpatch'][nidx] = patches_l
                    training_data['rpatch'][nidx] = patches_r
                    training_data['patch_3d'][nidx] = patches_3d
                    training_data['patch_uv'][nidx] = patches_uv
        
        # aggregate data
        for key in training_data.keys():
            if 'patchSize' in key: continue
            training_data[key] = training_data[key][mask_patch_valid]
            
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
    
    