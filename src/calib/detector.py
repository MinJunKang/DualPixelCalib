
from curses import raw
import pdb
import os
import cv2
import math
from tqdm import tqdm
import flax
import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from pathlib import Path

import parmap
from multiprocessing import Manager, cpu_count

from src.calib.utils.file_manager import makedir_custom, read_img_raw, read_img
from src.calib.utils.geometry import convert2homography
from src.calib.createBoard import create_full_board
from src.calib.extern.single_demo import Run_MPRNet, Run_IFAN44, Run_DeepRFT


class CalibBoard(object):

    def __init__(self, args):
        
        # arguments
        self.device = args.device
        self.verbose = args.verbose
        self.board_name = args.board_name
        self.calibname = args.calibname
        self.focus_phone = args.focus_phone
        self.fnumber_dslr = args.fnumber_dslr
        self.size_square_mm = args.size_square_mm

        # board initialize
        self.board_info = self.board_init(args.model_cfg.scales)
        self.board_ref = self.board_info[args.model_cfg.scales.index(1.0)]
        self.rootpath = Path('./dataset') / self.calibname
        self.storepath = makedir_custom(self.rootpath / 'saved')
        self.tag = '%s_%d' % (self.device, int(self.fnumber_dslr * 10) if self.device == 'dslr' else int(self.focus_phone * 10))

    def board_init(self, scales=[0.5, 1.0, 2.0]):
        
        # read info_board
        info_board = np.load('boards/' + self.board_name + '/board_info.npy', allow_pickle=True).tolist()

        # get board with multiple scale
        board_scaled = []
        for scale in scales:
            board_scaled.append(create_full_board(info_board, self.size_square_mm, scale))
        
        return board_scaled
    
    def read_lidar(self):
        
        raw_data = {'lidar_img': [], 'lidar_depth': [], 'lidar_mask': [],
                    'view_num': 0}
        
        for path in sorted(self.rootpath.glob('sample*')):
            
            # read lidar information
            lidar_img = cv2.imread(str(path / 'LIDAR_IMG.png'))
            lidar_mask = np.load(str(path / 'LIDAR_MASK.npy'))
            lidar_depth = np.load(str(path / 'LIDAR_DEPTH.npy'))
            
            # mask is applied to img
            lidar_img *= np.repeat(np.expand_dims(lidar_mask, 2), 3, 2)
            
            raw_data['lidar_img'].append(lidar_img)
            raw_data['lidar_depth'].append(lidar_depth)
            raw_data['lidar_mask'].append((lidar_depth > 0) & lidar_mask)
            raw_data['view_num'] += 1

        return raw_data
    
    def read_data(self, device='dslr'):
        assert(device in ['dslr', 'phone'])
        
        raw_data = {'DPl_img': [], 'DPr_img': [], 'DPc_img': [], 
                    'view_num': 0, 'focal_mm': []}
        
        for path in tqdm(sorted(self.rootpath.glob('sample*')), desc='reading %s' % device):
        
            info_ = np.load(str(path / 'info.npy'), allow_pickle=True).item()
            
            if device == 'dslr':  # read dslr images
                index_ = info_['fnumber_dslr'].index(self.fnumber_dslr)
                focal_mm = info_['focal_dslr'][index_]
                
                img_name = info_['filename_dslr'][index_]
                img_l = read_img(str(path / 'LEFT' / img_name), scale=1.0)
                img_r = read_img(str(path / 'RIGHT' / img_name), scale=1.0)
                img_c = read_img(str(path / 'CENTER' / img_name), scale=0.5)
                
            else:  # read phone images
                index_ = info_['focus_phone'].index(self.focus_phone)
                focal_mm = 27.0
                
                img_name = info_['filename_phone'][index_]
                img_l = read_img_raw(str(path / 'LEFT' / img_name))
                img_r = read_img_raw(str(path / 'RIGHT' / img_name))
                img_c = read_img_raw(str(path / 'CENTER' / img_name))
                
            raw_data['DPl_img'].append(img_l)
            raw_data['DPr_img'].append(img_r)
            raw_data['DPc_img'].append(img_c)
            raw_data['focal_mm'].append(focal_mm)
            raw_data['view_num'] += 1
        
        return raw_data
    
    def detectBoard(self):
        '''
            board detection
            1. img_DSLR
            2. img_Phone
        '''
        
        # read raw data
        raw_data = dict()
        raw_data[self.device] = self.read_data(device=self.device)
        raw_data[self.device].update(self.read_lidar())
        
        savepath = self.storepath / ('rawdata_%s_%s.npy' % (self.device, self.tag))
        
        # load data if available
        if savepath.is_file():
            processed_data = np.load(str(savepath), allow_pickle=True).item()
            if_update = False
        else:
            # network to run
            # denoisenet = Run_MPRNet('./src/calib/extern/MPRNet', 'Denoising') 
            deblurnet = Run_DeepRFT('./src/calib/extern')
            deblurnet2 = Run_IFAN44('./src/calib/extern/IFAN')
            processor = [deblurnet, deblurnet2]
            
            # detect corners
            processed_data = dict()
            data_ = {'DPl_corner': [], 'DPr_corner': [], 'DPc_corner': [], 'lidar_corner': []}
            for i in tqdm(range(raw_data[self.device]['view_num']), desc='Detecting corners ...'):
                # multi-resolution aware aruco pattern's corner finder
                '''
                    left, right, center
                '''
                
                for tag in ['DPl', 'DPr', 'DPc', 'lidar']:
                    img_ = raw_data[self.device]['%s_img' % tag][i]
                    res_ = [1.0, 0.5, 0.25] if tag == 'lidar' else [1.0, 0.5, 0.25, 0.125]
                    data_['%s_corner' % tag].append(self.detectCorner(img=img_, 
                                                                      dictionary=self.board_ref['dictionary'],
                                                                      multi_res=res_,
                                                                      processor=processor))
            
            processed_data[self.device] = data_
            
            # deallocate memory for deblurnet, denoisenet
            for proc in processor:
                proc = []  # deallocation
            torch.cuda.empty_cache()
            np.save(str(savepath), processed_data)
            if_update = True
        
        return raw_data, processed_data, if_update
        
        
    def detectCorner(self, img, dictionary, multi_res=[1.0, 0.5, 0.25, 0.125], window_size=34, processor=[]):
        
        # hyperparameters
        brightness = 1.0
        brightness_inc = 0.25
        brightness_max = 2.0
        window_thres = 0.25
        
        patch_num = math.floor(self.board_ref['num_x'] * self.board_ref['num_y'] / 2)
        aruco_corners = np.zeros((patch_num, 4, 2), dtype=np.float32)
        aruco_marker_idx = np.zeros((patch_num, 1), dtype=np.int32)
        aruco_corners_mask_all = np.zeros((patch_num), dtype=bool)
        min_corner = min((self.board_ref['num_x'] - 1) * (self.board_ref['num_y'] - 1), 6)
        
        # convert to grayscale if colored
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        for res in multi_res:
            # resize img
            img_resized = resize(img, (img.shape[0] * res, img.shape[1] * res), anti_aliasing=True)
            
            # convert to uint8 type
            img_resized = (img_resized / img_resized.max() * 255).astype('uint8')
            
            # deblur for better corner detection (optional)
            for proc in processor:
                img_resized = proc.forward(img_resized)
            img_resized = cv2.bilateralFilter(img_resized, -1, 50, 10)
            
            # detect markers (brightness adaptation)
            while True:
                corners, marker_idx, rejected = cv2.aruco.detectMarkers(np.clip(img_resized * brightness, 0, 255).astype('uint8'), dictionary)
                corners = np.squeeze(corners)
                if corners.size > 0 or (brightness >= brightness_max):
                    break
                else:
                    brightness += brightness_inc
            
            if corners.size > 0:
                corners = np.reshape(corners, [-1, 4, 2])
                marker_idxi = np.squeeze(marker_idx)
                
                # check markers out of range
                if marker_idxi.tolist() != None:
                    marker_mask = marker_idxi <= (self.board_ref['num_x'] - 1) * (self.board_ref['num_y'] - 1)
                    marker_idxi = marker_idxi[marker_mask]
                if len(marker_idxi) == 0:
                    continue
                    
                # if idx is overlapped, reject candidates
                duplicated_idx = np.where(np.array(pd.DataFrame(marker_idxi).duplicated(keep=False)))[0]
                nonoverlapped_idx = np.setdiff1d(np.array(range(len(marker_idxi))), duplicated_idx)
                marker_idxi = marker_idxi[nonoverlapped_idx]
                corners = corners[nonoverlapped_idx]
                marker_idx = marker_idx[nonoverlapped_idx]
                
                # update mask
                aruco_corners_mask = np.zeros((patch_num), dtype=bool)
                aruco_corners_mask[marker_idxi] = True
                update_mask_detected = ~aruco_corners_mask_all[marker_idxi]
                update_mask = ~aruco_corners_mask_all & aruco_corners_mask
            
                if np.sum(update_mask) > 0:
                    aruco_corners[update_mask, :, :] = corners[update_mask_detected, :, :] / res
                    aruco_marker_idx[update_mask, :] = marker_idx[update_mask_detected, :]
                    aruco_corners_mask_all = aruco_corners_mask_all | aruco_corners_mask
                    
        corners = aruco_corners[aruco_corners_mask_all, :, :].tolist()
        corners_list = [np.expand_dims(np.array(corner, dtype=np.float32), axis=0) for corner in corners]
        markers_idx = aruco_marker_idx[aruco_corners_mask_all]
        img = (img * brightness).astype('float32')  # type: ignore
        
        # Subpix Refinement Process
        if len(corners_list) == 0:
            refined_corner = None
            corner_idx = None
            usage = False
        else:
            # determine window size based on marker detection (> 3px)
            for corner in corners_list:
                dist = min(np.linalg.norm(corner[:, 0] - corner[:, 1]), np.linalg.norm(corner[:, 1] - corner[:, 2]))
                window_size = min(window_size, int(dist * window_thres))
            window_size = max(4, window_size)
            
            # interpolate to find initial corners of checkerboard
            num_corner, corner_board, corner_idx = cv2.aruco.interpolateCornersCharuco(corners_list, markers_idx, img, self.board_ref['board'])
            
            if num_corner < min_corner:  # at least 4-points algorithm should work
                refined_corner = None
                corner_idx = None
                usage = False
            else:
                # subpixel refinement of corners
                refined_corner = self.SubPixRefinement(img, corner_board, window_size=window_size)
                inf_mask = (refined_corner[:, :2] == np.Infinity) | (refined_corner[:, :2] == np.NaN)
                refined_corner[:, :2][inf_mask] = 0.0
                refined_corner[:, :2] = ~inf_mask * refined_corner[:, :2] + inf_mask * np.squeeze(corner_board)
                
                # convert corner index to be aligned with template (detected corners' reference is at center)
                multiplier = (np.floor(corner_idx / (self.board_ref['num_x'] - 1)) + 1) * (self.board_ref['num_x'] + 1)
                offset = np.mod(corner_idx, (self.board_ref['num_x'] - 1)) + 1
                corner_idx = multiplier.astype('uint64') + offset
                usage = True
        
        corner_info = {'corner': refined_corner, 'corner_idx': corner_idx, 'usage': usage, 'winsize': window_size}
        return corner_info
    
    def SaddleInitialize(self, window_size):
        
        # initialize parameters
        window_size_full = window_size * 2 + 1
        saddlekernel = np.zeros((window_size_full, window_size_full), dtype=np.float32)
        maxVal = window_size + 1
        sum = 0
        cnt = 0
        for j in range(-window_size, window_size + 1):
            for i in range(-window_size, window_size + 1):
                iidx = i + window_size
                jidx = j + window_size
                saddlekernel[jidx, iidx] = maxVal - math.sqrt(i * i + j * j)
                if saddlekernel[jidx, iidx] > 0:
                    cnt += 1
                else:
                    saddlekernel[jidx, iidx] = 0
                sum += saddlekernel[jidx, iidx]
        # scale kernel
        saddlekernel /= sum
        
        Amat = np.zeros((cnt, 6), dtype=np.float32)
        valid_mask = saddlekernel > 0
        
        cnt = 0
        for j in range(-window_size, window_size + 1):
            for i in range(-window_size, window_size + 1):
                iidx = i + window_size
                jidx = j + window_size
                if valid_mask[jidx, iidx]:
                    Amat[cnt, 0] = i * i
                    Amat[cnt, 1] = j * j
                    Amat[cnt, 2] = i * j
                    Amat[cnt, 3] = j
                    Amat[cnt, 4] = i
                    Amat[cnt, 5] = 1.0
                    cnt += 1
                    
        At = cv2.transpose(Amat)
        invAtAAt = cv2.invert(np.matmul(At, Amat), cv2.DECOMP_SVD)[1]  # type: ignore
        invAtAAt = np.matmul(invAtAAt, At)
        
        return saddlekernel, invAtAAt, valid_mask

    def SubPixRefinement(self, img, corners, window_size=17, max_iter=100, step_threshold=0.001):
        '''
            Surface fitting to get subpixel accurate corners
            img : [H, W, C]
            corners : [N, 1, 2]
        '''
        
        corners = np.squeeze(corners)
        num_corner = len(corners)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        height, width = img.shape
        
        # initialize saddles
        saddlekernel, AMat, valid_mask = self.SaddleInitialize(window_size)
        
        # smooth circular filter
        img = cv2.filter2D(img, cv2.CV_64FC1, saddlekernel)
        
        # define saddle points
        saddle_points = np.zeros((num_corner, 6), dtype=np.float32)  # x, y, a1, a2, s, det
        saddle_points[:, :2] = corners.copy()
        bMat = np.zeros((AMat.shape[1], 1), dtype=np.float32)
        
        for n in range(num_corner):
            for iter in range(max_iter):
                px = saddle_points[n, 0]
                py = saddle_points[n, 1]
                if((px > window_size + 1) and (px < width - (window_size + 2)) and (py > window_size + 1) and (py < height - (window_size + 2))):
                    x0 = int(math.floor(px))
                    y0 = int(math.floor(py))
                    xw = px - x0
                    yw = py - y0
                    
                    # precompute bilinear interpolation weights
                    w00 = (1.0 - xw) * (1.0 - yw)
                    w01 = xw * (1.0 - yw)
                    w10 = (1.0 - xw) * yw
                    w11 = xw * yw
                    
                    # fit to local neighborhood = b vector
                    cnt = 0
                    for j in range(-window_size, window_size + 1):
                        y00 = y0 + j
                        y10 = y00 + 1
                        for i in range(-window_size, window_size + 1):
                            iidx = i + window_size
                            jidx = j + window_size
                            if valid_mask[jidx, iidx]:
                                x00 = x0 + i
                                x01 = x00 + 1
                                bMat[cnt, :] = img[y00, x00] * w00 + img[y00, x01] * w01 + img[y10, x00] * w10 + img[y10, x01] * w11
                                cnt += 1
                                
                    # fit quadric to surface by solving LSQ
                    p = np.matmul(AMat, bMat)  # [6, 1]
                    
                    # k5, k4, k3, k2, k1, k0
                    saddle_points[n, 5] = 4.0 * p[0] * p[1] - p[2] * p[2]  # update det
                    dx = (-2 * p[1] * p[4] + p[2] * p[3]) / saddle_points[n, 5]
                    dy = (p[2] * p[4] - 2 * p[0] * p[3]) / saddle_points[n, 5]
                    
                    saddle_points[n, 0] += dx  # update x
                    saddle_points[n, 1] += dy  # update y
                    dx = abs(dx)
                    dy = abs(dy)
                    
                    if (iter == max_iter) or ((step_threshold > dx) and (step_threshold > dy)):
                        # converged
                        k4mk5 = p[1] - p[0]
                        saddle_points[n, 4] = math.sqrt(p[2] * p[2] + k4mk5 * k4mk5)  # s
                        saddle_points[n, 2] = math.atan2(-p[2], k4mk5) / 2.0  # a1
                        saddle_points[n, 3] = math.acos((p[1] + p[0]) / saddle_points[n, 4]) / 2.0  # a2
                        break
                    else:
                        if (saddle_points[n, 5] > 0) or (abs(saddle_points[n, 0] - corners[n, 0]) > window_size) or (abs(saddle_points[n, 1] - corners[n, 1]) > window_size):
                            saddle_points[n, 0] = saddle_points[n, 1] = np.inf
                            # diverged
                            break
                else:
                    # too close to border
                    saddle_points[n, 0] = saddle_points[n, 1] = np.inf
                    break
        
        return saddle_points
    
    def propagate_corners(self, raw, data, if_update=False):
        
        # save path
        savepath = self.storepath / ('middata_%s_%s.npy' % (self.device, self.tag))
        if if_update and savepath.is_file():
            savepath.unlink()
        
        # load data if available
        if savepath.is_file():
            dev_data = np.load(str(savepath), allow_pickle=True).item()
            if_update = False
        else:
            repj_thres = 1.5
            dev_data = data[self.device].copy()
            dev_raw = raw[self.device]
            worldpts = np.reshape(self.board_ref['rec3d_world'], [-1, 3])
            worldpts_idx = np.expand_dims(np.array(range(len(worldpts))), 1)
            cnt_calib = {'DPl': 0, 'DPr': 0, 'DPc': 0, 'lidar': 0}
            
            # propagate corners
            for i in tqdm(range(dev_raw['view_num']), desc='Propagating corners'):
                
                for tag in ['DPl', 'DPr', 'DPc', 'lidar']:
                    
                    corner_obj = dev_data['%s_corner' % tag]
                    calib_obj = dev_data['%s_calib' % tag]
                    h, w = dev_raw['%s_img' % tag][i].shape[:2]
                    
                    # check usage
                    if corner_obj[i]['usage']:
                        
                        idx = cnt_calib['%s' % tag]
                        
                        # if reprojection error is too high, reject view
                        if calib_obj['ret'][idx] > repj_thres:
                            dev_data['%s_corner' % tag][i]['usage'] = False
                            dev_data['%s_corner' % tag][i]['corner'] = []
                            dev_data['%s_corner' % tag][i]['corner_idx'] = []
                            dev_data['%s_corner' % tag][i]['corner_mask'] = []
                        else:
                            # project template corner based on calibration
                            imgpoints, _ = cv2.projectPoints(worldpts, 
                                                            calib_obj['rvecs'][idx], 
                                                            calib_obj['tvecs'][idx], 
                                                            calib_obj['mtx'], 
                                                            calib_obj['dist'])
                            
                            # subpixel refinement of corners
                            refined_corner = self.SubPixRefinement(dev_raw['%s_img' % tag][i], 
                                                                   imgpoints, 
                                                                   window_size=int(corner_obj[i]['winsize'] * 0.5))
                            
                            inf_mask = (refined_corner[:, :2] == np.Infinity) | (refined_corner[:, :2] == np.NaN)
                            refined_corner[:, :2][inf_mask] = 0.0
                            refined_corner[:, :2] = ~inf_mask * refined_corner[:, :2] + inf_mask * np.squeeze(imgpoints)
                            
                            # check if the corners is out of image
                            mask_x = np.logical_and(refined_corner[:, 0] >= 0, refined_corner[:, 0] < w)
                            mask_y = np.logical_and(refined_corner[:, 1] >= 0, refined_corner[:, 1] < h)
                            mask = np.logical_and(mask_x, mask_y)
                            
                            dev_data['%s_corner' % tag][i]['corner'] = np.float32(refined_corner)
                            dev_data['%s_corner' % tag][i]['corner_idx'] = worldpts_idx
                            dev_data['%s_corner' % tag][i]['corner_mask'] = mask
                        cnt_calib['%s' % tag] += 1
                    else:
                        dev_data['%s_corner' % tag][i]['usage'] = False
                        dev_data['%s_corner' % tag][i]['corner'] = []
                        dev_data['%s_corner' % tag][i]['corner_idx'] = []
                        dev_data['%s_corner' % tag][i]['corner_mask'] = []
                        
            np.save(str(savepath), dev_data)
            if_update = True
            
        data[self.device].update(dev_data)
                        
        return data, if_update