
import pdb
import cv2
import math
from tqdm import tqdm
import torch

from functools import reduce
from operator import itemgetter

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from kornia.geometry.transform import rotate
from scipy.optimize import lsq_linear, least_squares


from src.calib.utils.geometry import (convert2homography, get_plane_depth, findrotz_euler,
                                      subpixel_cropper, findhomography, 
                                      grid_rotate, subpixel_cropper_batch)
from src.calib.utils.blobsetting import cfg as blob_cfg
from src.calib.utils.file_manager import makedir_custom


class Calibration(object):

    def __init__(self, args, board=None):

        # blob detector setting
        self.blob_cfg = blob_cfg
        
        # path parameters
        self.board = board
        self.board_scales = args.model_cfg.scales
        self.verbose = args.verbose
        if board is not None:
            self.rootpath = Path('./dataset') / board.calibname
            self.storepath = makedir_custom(self.rootpath / 'saved')
            self.resultpath = makedir_custom(self.rootpath / 'result')
            self.checkpath = makedir_custom(Path('checkpoints'))
        
        # fnumber setting
        self.fnumber_float = {'1.2': 0.7, '1.4': 1, '1.6': 1.3, '1.7': 1.5, '1.8': 1.7, 
                              '2.0': 2, '2.2': 2.3, '2.5': 2.7, '2.8': 3, 
                              '3.2': 3.3, '3.5': 3.6, '4.0': 4, '4.5': 4.3, 
                              '5.0': 4.7, '5.6': 5, '6.3': 5.3, '7.1': 5.7,
                              '8.0': 6, '9.0': 6.3, '10.0': 6.7, '11.0': 7, 
                              '13.0': 7.3, '14.0': 7.7, '16.0': 8, '18.0': 8.3, 
                              '20.0': 8.7}
        
    def fnumber_converter(self, fnumber):
        # convert fnumber to float
        return np.sqrt(np.power(2, self.fnumber_float[str(fnumber)]))
    
    def findpairs(self, data, worldpts, h, w, viewnum, key):
        pairs = {'x': [], 'X': [], 'idx': []}
        pairs['h'], pairs['w'] = h, w
        for i in range(viewnum):
            corner_obj = data['%s_corner' % key]
            if corner_obj[i]['usage']:
                corner_idx = corner_obj[i]['corner_idx'].astype('uint64')
                pairs['x'].append(corner_obj[i]['corner'][:, :2].astype('float32'))
                pairs['X'].append(np.squeeze(worldpts[corner_idx]).astype('float32'))
                pairs['idx'].append(i)
        return pairs
    
    def run_single_calibration(self, pairs, viewnum, type):
        
        h, w = pairs['h'], pairs['w']
        data = {'mtx': [], 'umtx': [], 'dist': [], 'tvecs': [], 'rvecs': [], 'idx': []}
        
        # calibration
        print('calibration of %s view camera with (%d/%d) views' % (type, len(pairs['idx']), viewnum))
        ret, data['mtx'], data['dist'], data['rvecs'], data['tvecs'] = cv2.calibrateCamera(pairs['X'], pairs['x'], (w, h), None, None)
        
        # save undistorted intrinsic
        # if we consider the points of undistorted observation, use this as intrinsics
        data['umtx'], _ = cv2.getOptimalNewCameraMatrix(data['mtx'], data['dist'], (w, h), 1, (w, h))
        
        # calculate reprojection error
        all_error = []
        for i in range(len(pairs['idx'])):
            imgpoints, _ = cv2.projectPoints(pairs['X'][i], data['rvecs'][i], data['tvecs'][i], data['mtx'], data['dist'])
            error = cv2.norm(np.expand_dims(pairs['x'][i], 1), imgpoints, cv2.NORM_L2)/len(imgpoints)
            all_error.append(error)
            data['idx'].append(pairs['idx'][i])
        data['ret'] = all_error
        
        return data
    
    def single_calib(self, raw, data, if_update=False, is_final=False):
        
        # save path
        prefix = 'final' if is_final else 'initial'
        savepath = self.storepath / ('single_calib_%s_%s_%s.npy' % (self.board.device, self.board.tag, prefix))
        if if_update and savepath.is_file():
            savepath.unlink()
        
        if savepath.is_file():
            calibdata = np.load(str(savepath), allow_pickle=True).item()
            if_update = False
        else:
            # 3d world points
            worldpts = np.reshape(self.board.board_ref['rec3d_world'], [-1, 3])
            
            calibdata = dict()
            dev_data = data[self.board.device]
            dev_raw = raw[self.board.device]
            viewnum = dev_raw['view_num']
            h_dev, w_dev = dev_raw['h_dev'], dev_raw['w_dev']
            h_lidar, w_lidar = dev_raw['h_lidar'], dev_raw['w_lidar']
            for tag in ['DPl', 'DPr', 'DPc', 'lidar']:  # calibration of DPl, DPr is needed only for propagating corners
                
                # 2d - 3d pairs
                (h, w) = (h_lidar, w_lidar) if tag =='lidar' else (h_dev, w_dev)
                pairs = self.findpairs(dev_data, worldpts, h, w, viewnum, tag)
                
                # calibration run with pairs
                calib = self.run_single_calibration(pairs, viewnum, tag)
                calibdata['%s_calib' % tag] = calib
                
            np.save(str(savepath), calibdata)
            if_update = True
            
        if self.verbose:
            for tag in ['DPl', 'DPr', 'DPc', 'lidar']:  # calibration of DPl, DPr is needed only for propagating corners
                reprojection_error = np.mean(np.array(calibdata['%s_calib' % tag]['ret']))
                print('Calibration %s from %s view camera with reprojection error of %f' % (prefix, tag, reprojection_error))
            
        data[self.board.device].update(calibdata)
        
        return data, if_update
    
    def stereo_calib(self, raw, data, if_update=False):
        '''
            Perform extrinsic calibration between camera and LIDAR
            1. Intrinsics are from single calibration step (L + R, LIDAR)
            2. Get common view between L + R and LIDAR
            3. Using common view's corners, perform stereo calibration
        '''
        
        # save path
        savepath = self.storepath / ('stereo_calib_%s_%s.npy' % (self.board.device, self.board.tag))
        if if_update and savepath.is_file():
            savepath.unlink()
        
        if savepath.is_file():
            calibdata = np.load(str(savepath), allow_pickle=True).item()
            err_c2l, err_l2c = calibdata['ret_c2l'], calibdata['ret_l2c']
            if_update = False
        else:
            # 3d world points
            worldpts = np.float32(np.reshape(self.board.board_ref['rec3d_world'], [-1, 3]))
            
            # load intrinsic data
            dev_data = data[self.board.device]
            dev_raw = raw[self.board.device]
            h_dev, w_dev = dev_raw['DPc_img'][0].shape[:2]
            h_lidar, w_lidar = dev_raw['lidar_img'][0].shape[:2]
            mtx_c, dist_c = dev_data['DPc_calib']['mtx'], dev_data['DPc_calib']['dist']
            mtx_l, dist_l = dev_data['lidar_calib']['mtx'], dev_data['lidar_calib']['dist']
            idx_c, idx_l = dev_data['DPc_calib']['idx'], dev_data['lidar_calib']['idx']
            
            # Get common view between L + R and LIDAR
            cidx, cidx_c, cidx_l = np.intersect1d(np.array(idx_c), np.array(idx_l), return_indices=True)
            
            # Get paired 2d - 3d points
            pairs = {'x1': [], 'x2': [], 'd1': [], 'd2': [], 'X': [], 'valid': []}
            for i, idx in enumerate(cidx.tolist()):
                idx_c_, idx_l_ = cidx_c[i], cidx_l[i]  # index of intersected calib data
                rvec_c, tvec_c = dev_data['DPc_calib']['rvecs'][idx_c_], dev_data['DPc_calib']['tvecs'][idx_c_]
                rvec_l, tvec_l = dev_data['lidar_calib']['rvecs'][idx_l_], dev_data['lidar_calib']['tvecs'][idx_l_]
                
                # check if the corners is out of image
                dev_corner = dev_data['DPc_corner'][idx]['corner'][:, :2]
                lidar_corner = dev_data['lidar_corner'][idx]['corner'][:, :2]
                dev_mask = dev_data['DPc_corner'][idx]['corner_mask']
                lidar_mask = dev_data['lidar_corner'][idx]['corner_mask']
                
                mask_common = np.logical_and(dev_mask, lidar_mask)
                
                # check overlapping
                if np.sum(mask_common) < 6:
                    pairs['valid'].append(False)
                else:
                    pairs['valid'].append(True)
                    pairs['x1'].append(dev_corner[mask_common])
                    pairs['x2'].append(lidar_corner[mask_common])
                    pairs['X'].append(worldpts[mask_common])
                    pairs['d1'].append(get_plane_depth(pairs['X'][-1], pairs['x1'][-1], rvec_c, tvec_c, mtx_c, dist_c))
                    pairs['d2'].append(get_plane_depth(pairs['X'][-1], pairs['x2'][-1], rvec_l, tvec_l, mtx_l, dist_l))
                    
            
            # from src.calib.utils.visualizer import visualize_corner, visualize_corner_v2
            
            # devpath = makedir_custom(Path('dev'))
            # lidarpath = makedir_custom(Path('lidar'))
            
            # for i, idx in enumerate(cidx.tolist()):
            #     img_dev = dev_raw['DPc_img'][idx]
            #     img_lidar = dev_raw['lidar_img'][idx]
            #     corner_dev = pairs['x1'][i]
            #     corner_lidar = pairs['x2'][i]
            #     visualize_corner(img_dev.copy(), corner_dev, list(range(len(corner_dev))), str(devpath / ('dev_%d.png' % i)), sizet=4)
            #     visualize_corner(img_lidar.copy(), corner_lidar, list(range(len(corner_lidar))), str(lidarpath / ('lidar_%d.png' % i)), sizet=1)
                
            # run stereo calibration
            flags = cv2.CALIB_FIX_INTRINSIC
            criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            retS, _, _, _, _, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(pairs['X'], pairs['x1'], pairs['x2'], mtx_c, dist_c, mtx_l, dist_l, (w_dev, h_dev), criteria_stereo, flags)

            '''
                R2 = RR1,    T2 = RT1 + T
                Rmat_c = cv2.Rodrigues(rvec_c)[0]
                Rmat_l = cv2.Rodrigues(rvec_l)[0]
                Rot_ = np.matmul(Rmat_l, np.linalg.inv(Rmat_c))
                T_ = tvec_l - np.matmul(Rot_, tvec_c)
            '''
            
            # get projective matrix
            Pmat = np.concatenate([np.concatenate([Rot, Trns], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
            invPmat = np.linalg.inv(Pmat)
            
            # measure reprojection error
            err_c2l, err_l2c, cnt_valid = [], [], []
            for i in range(len(cidx)):
                idx_valid = sum(cnt_valid)
                if pairs['valid'][i]:
                    # undistortion
                    undistored_DSLR = cv2.undistortPoints(pairs['x1'][idx_valid], mtx_c, dist_c)
                    undistored_LIDAR = cv2.undistortPoints(pairs['x2'][idx_valid], mtx_l, dist_l)
                    
                    # unprojection
                    unprojected_DSLR = convert2homography(np.squeeze(undistored_DSLR), 1.0) * pairs['d1'][idx_valid]
                    unprojected_LIDAR = convert2homography(np.squeeze(undistored_LIDAR), 1.0) * pairs['d2'][idx_valid]
                    
                    # projection from DSLR to LIDAR
                    rvec_DSLR2LIDAR = cv2.Rodrigues(Pmat[:3, :3])[0]
                    tvec_DSLR2LIDAR = Pmat[:3, 3:]
                    imgpoints_LIDAR, _ = cv2.projectPoints(unprojected_DSLR, rvec_DSLR2LIDAR, tvec_DSLR2LIDAR, mtx_l, dist_l)
                    
                    # projection from LIDAR to DSLR
                    rvec_LIDAR2DSLR = cv2.Rodrigues(invPmat[:3, :3])[0]
                    tvec_LIDAR2DSLR = invPmat[:3, 3:]
                    imgpoints_DSLR, _ = cv2.projectPoints(unprojected_LIDAR, rvec_LIDAR2DSLR, tvec_LIDAR2DSLR, mtx_c, dist_c)
                    
                    # From DSLR to LIDAR (reprojection error)
                    error_c2l = cv2.norm(np.expand_dims(pairs['x2'][idx_valid], 1), imgpoints_LIDAR.astype('float32'), cv2.NORM_L2)/len(imgpoints_LIDAR)
                    
                    # From LIDAR to DSLR (reprojection error)
                    error_l2c = cv2.norm(np.expand_dims(pairs['x1'][idx_valid], 1), imgpoints_DSLR.astype('float32'), cv2.NORM_L2)/len(imgpoints_DSLR)
                    
                    err_c2l.append(error_c2l)
                    err_l2c.append(error_l2c)
                cnt_valid.append(pairs['valid'][i])
            
            # save calibration results
            calibdata = {'Pmat_c2l': Pmat, 'Emat_c2l': Emat, 'Fmat_c2l': Fmat, 'ret_c2l': err_c2l, 'ret_l2c': err_l2c, 'cnt_valid': cnt_valid}
            np.save(str(savepath), calibdata)
            if_update = True
        
        rpj_error1 = np.mean(np.array(err_c2l))
        rpj_error2 = np.mean(np.array(err_l2c))
        if self.verbose:
            print('Stereo calib rpj error: (%f) [cam to lidar] (%f) [lidar to cam]' % (rpj_error1, rpj_error2))
        data[self.board.device].update(calibdata)
        
        return data, if_update
    
    def estimate_gvalue(self, raw, data, if_update=False, visualize_graph=False):
        
        # save path
        savepath = self.storepath / ('metadata_%s_%s.npy' % (self.board.device, self.board.tag))
        if if_update and savepath.is_file():
            savepath.unlink()
        
        if savepath.is_file():
            meta_data = np.load(str(savepath), allow_pickle=True).item()
        else:
            worldpts = np.float32(np.reshape(self.board.board_ref['rec3d_world'], [-1, 3]))
            
            dev_data = data[self.board.device]
            
            idx_l, idx_r, idx_c = dev_data['DPl_calib']['idx'], dev_data['DPr_calib']['idx'], dev_data['DPc_calib']['idx']
            
            # get common index
            cidx = reduce(np.intersect1d, (idx_l, idx_r, idx_c)).tolist()
            _, _, cidx_c = np.intersect1d(cidx, np.array(idx_c), return_indices=True)
            _, _, cidx_l = np.intersect1d(cidx, np.array(idx_l), return_indices=True)
            _, _, cidx_r = np.intersect1d(cidx, np.array(idx_r), return_indices=True)
            
            disp_stacked, depth_stacked = [], []
            
            # get disp between L and R
            for i, idx in enumerate(cidx):
                idx_c, idx_l, idx_r = cidx_c[i], cidx_l[i], cidx_r[i]
                
                # corners of center, left and right
                ccorner = dev_data['DPc_corner'][idx]['corner']
                lcorner = dev_data['DPl_corner'][idx]['corner']
                rcorner = dev_data['DPr_corner'][idx]['corner']
                
                mtx_c, dist_c = dev_data['DPc_calib']['mtx'], dev_data['DPc_calib']['dist']
                rvec_c, tvec_c = dev_data['DPc_calib']['rvecs'][idx_c], dev_data['DPc_calib']['tvecs'][idx_c]
                
                # get disparity
                real_lrdisp = rcorner[:, 0] - lcorner[:, 0]
                real_clinliermask = np.abs(rcorner[:, 1] - lcorner[:, 1]) < 0.1
                
                # get depth of center
                real_depth = get_plane_depth(worldpts, ccorner[:, :2].astype('float32'), rvec_c, tvec_c, mtx_c, dist_c)
                
                # output
                disp_stacked += real_lrdisp[real_clinliermask].tolist()
                depth_stacked += real_depth[real_clinliermask].tolist()
            
            # get gvalue from (L and R)
            focal_mm = raw[self.board.device]['focal_mm'][0]
            gvalue, kvalue, bias, tterm, aperture = self.find_calibparams(disp_stacked, depth_stacked, focal_mm, dev_data['DPc_calib']['mtx'][0][0])
            print('Camera focus roughly locates at %f [mm]' % gvalue)
            mindepth, maxdepth = min(depth_stacked), max(depth_stacked)
            mindisp, maxdisp = min(disp_stacked), max(disp_stacked)
            depth_range = [mindepth, maxdepth]
            disp_range = [mindisp, maxdisp]
            meta_data = {'igvalue': gvalue, 'kvalue': kvalue, 'bias': bias, 
                        'tterm': tterm, 'focal_mm': focal_mm, 'aperture': aperture, 
                        'depth_range': depth_range, 'disp_range': disp_range}
            np.save(str(savepath), meta_data)
        
        data[self.board.device]['meta_data'] = meta_data
        
        if visualize_graph:
            yvalue = np.array(disp_stacked)
            xvalue = np.array(depth_stacked)
            depthf_ones = convert2homography(xvalue, 1.0)
            res = lsq_linear(1.0 / depthf_ones, -yvalue)
            res_lsq = least_squares((lambda x, A, b: A * x[0] + x[1] - b), res.x, loss='soft_l1', f_scale=0.1, args=(1.0 / xvalue[:, 0], -yvalue))
            
            # plot graph
            plt.plot(1 / xvalue, 1 / xvalue * res_lsq.x[0] + res_lsq.x[1], c='b', linewidth=2)
            plt.scatter(1 / xvalue, -yvalue, s=8, c='g')
            plt.title('Focus distance g at %.1fmm with #F=%.1f' % (gvalue, self.board.fnumber), fontsize=15)
            plt.xlabel('Inverse depth [mm-1]', fontsize=15)
            plt.ylabel('Disparity [px]', fontsize=15)
            plt.savefig('%s/graph.png' % str(self.storepath))
            plt.close()
        return data
    
    def find_calibparams(self, disp, depth, focal_mm, fx):
        
        # inputs
        dispf = np.array(disp)
        depthf = np.array(depth)
        if self.board.device == 'dslr':
            fnumber = np.sqrt(np.power(2, self.fnumber_float[str(self.board.fnumber_dslr)]))
        else:
            fnumber = np.sqrt(np.power(2, self.fnumber_float['1.7']))
        
        depthf_ones = convert2homography(depthf, 1.0)
        res = lsq_linear(1.0 / depthf_ones, dispf)
        res_lsq = least_squares((lambda x, A, b: A * x[0] + x[1] - b), res.x, loss='soft_l1', 
                                f_scale=0.1, args=(1.0 / depthf[:, 0], dispf))
        
        # gvalue
        gvalue = -res_lsq.x[0] / res_lsq.x[1]
        aperture = fx / fnumber
        
        # kvalue
        kvalue = -res_lsq.x[0] / ((focal_mm / (gvalue - focal_mm)) * gvalue * aperture)
        
        # bias
        bias = (res_lsq.x[1] - (aperture * focal_mm / (gvalue - focal_mm)) * kvalue)
        tterm = bias * 2 / fx
        
        return gvalue, kvalue, bias, tterm, aperture
    
    def prepare_patches(self, raw, data, multi=True, if_update=False):
        '''
            Strictly, left and right dual-pixel's intrinsic should be calibrated individually along depth.
        '''
        
        # save path
        savepath = self.storepath / ('patchdata_%s_%s.npy' % (self.board.device, self.board.tag))
        if if_update and savepath.is_file():
            savepath.unlink()
        
        if savepath.is_file():
            patchdata = np.load(str(savepath), allow_pickle=True).item()
        else:
            dev_data = data[self.board.device]
            dev_raw = raw[self.board.device]
            data_all = [dev_raw, dev_data]
            board_num = len(self.board.board_info) if multi else 1
            
            # find common indexes of all
            idx_l, idx_r, idx_c = dev_data['DPl_calib']['idx'], dev_data['DPr_calib']['idx'], dev_data['DPc_calib']['idx']
            idx_lidar = dev_data['lidar_calib']['idx']
            idx_common = reduce(np.intersect1d, (idx_l, idx_r, idx_c, idx_lidar)).tolist()
            _, _, idxes_l = np.intersect1d(np.array(idx_common), np.array(idx_l), return_indices=True)
            _, _, idxes_r = np.intersect1d(np.array(idx_common), np.array(idx_r), return_indices=True)
            _, _, idxes_c = np.intersect1d(np.array(idx_common), np.array(idx_c), return_indices=True)
            _, _, idxes_lidar = np.intersect1d(np.array(idx_common), np.array(idx_lidar), return_indices=True)
            
            # multi-res board assign
            indexes = {'DPl': idxes_l.tolist(), 'DPr': idxes_r.tolist(), 'DPc': idxes_c.tolist(), 'lidar': idxes_lidar.tolist()}
            patchdata = {'DPl_patch': [], 'DPr_patch': [], 'DPc_patch': [], 'lidar_patch': [], 'board_num': board_num}
            
            # get patches
            infos_focus = self.calc_circles(self.board.board_ref, data_all, 'DPc', 'Focus', idx_common, indexes['DPc'])
            for tag in ['DPl', 'DPr', 'DPc', 'lidar']:
                dindex = None if tag == 'lidar' else indexes['DPc']
                infos_ = None if tag == 'lidar' else infos_focus
                patchdata['%s_patch' % tag] = self.calc_patches(data_all, tag, idx_common, indexes[tag], board_num, dindex, infos_)
            
            np.save(str(savepath), patchdata)
        
        return patchdata
    
    def calc_circles(self, board, data, type, dtype, indexes, calib_indexes):
        
        # calculate circles, patch locations
        patch_prop = {'circle': [], 'mask': []}
        
        # board info
        template = np.float32(board['template'])
        size_template_px = board['size_px']
        size_template_py = board['size_py']
        size_patch = int(min(size_template_px, size_template_py) / board['num_grid'] * 0.65)
        size_patch = (size_patch + 1) if size_patch % 2 == 0 else size_patch  # odd number
        
        radius = board['radius'].reshape(-1, 1)
        patches_corner = board['patch2d_template'].reshape(-1, 4, 2)
        
        data_raw, data_calib = data[0], data[1]
        
        for i, idx in tqdm(enumerate(indexes), desc='Calculating circles for %s' % dtype):
            calibidx = calib_indexes[i]
            
            # load calib info
            rvec = data_calib['%s_calib' % type]['rvecs'][calibidx]
            tvec = data_calib['%s_calib' % type]['tvecs'][calibidx]
            umtx = data_calib['%s_calib' % type]['umtx']
            mtx, dist = data_calib['%s_calib' % type]['mtx'], data_calib['%s_calib' % type]['dist']
            
            # projected image
            rotz = torch.tensor(-findrotz_euler(rvec) * 180 / math.pi)
            homo = findhomography(rvec, tvec, umtx, size_template_px, size_template_py, self.board.size_square_mm)
            img = data_raw['%s_img' % dtype][idx]
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            uimg = cv2.undistort(img, mtx, dist, None, umtx)  # type: ignore
            pimg = cv2.warpPerspective(uimg, np.linalg.inv(homo), template.shape[::-1])
            
            # detect circles from blur patch
            circles, mask, _ = self.detect_circle_from_img(pimg, radius, patches_corner, size_patch, rotz)
            
            patch_prop['circle'].append(circles)
            patch_prop['mask'].append(mask)
            
        return patch_prop
    
    def calc_patches(self, data, type, indexes, calib_indexes, board_num, didx=None, infos_focus=None):
        '''
            Todo : 
            1. surface normal
            2. K value included
            3. h, w included
        '''
        
        # calc basic informations
        infos = self.calc_circles(self.board.board_ref, data, type, type, indexes, calib_indexes)
        
        # multi-res patch assign
        data_raw, data_calib = data[0], data[1]
        result = {'uv_coord': [], 'sharp': [], 'depth': [], 'blur': [],
                  'normal': [], 'Kmat': [], 'scale': [], 'idx': [], 'idx_board': [], 'size_patch': []}
        if infos_focus is not None:
            result['focus'] = []
        
        for bidx in range(board_num):
            
            # infos
            board_info_res = self.board.board_info[bidx]
            template = np.float32(board_info_res['template'])
            worldpts = np.float32(np.reshape(board_info_res['rec3d_world'], [-1, 3]))
            size_template_px = board_info_res['size_px']
            size_template_py = board_info_res['size_py']
            circles_center = board_info_res['cir2d_template'].reshape(-1, 2)
            size_patch = int(min(size_template_px, size_template_py) / board_info_res['num_grid'] * 0.65)
            size_patch = (size_patch + 1) if size_patch % 2 == 0 else size_patch  # odd number

            # pts3d
            h_t, w_t = template.shape
            uv, vv = np.meshgrid(np.linspace(0, w_t - 1, w_t), np.linspace(0, h_t - 1, h_t))
            pts3d = convert2homography(np.stack([uv, vv], axis=-1), 0.0)
            pts3d[:, :, 0] *= (self.board.size_square_mm / size_template_px)
            pts3d[:, :, 1] *= (self.board.size_square_mm / size_template_py)
            pts3d = np.reshape(pts3d, [-1, 3])
            
            # patchify
            uv_coord, sharps_, depths_, blurs_, focuses_, indexes_, indexes_board_, normals_, Ks_, scales_ = [], [], [], [], [], [], [], [], [], []
        
            for i, idx in tqdm(enumerate(indexes), desc='Creating patches for %s (%d/%d)' % (type, bidx + 1, board_num)):
                calibidx = calib_indexes[i]
            
                rvec = data_calib['%s_calib' % type]['rvecs'][calibidx]
                tvec = data_calib['%s_calib' % type]['tvecs'][calibidx]
                umtx = data_calib['%s_calib' % type]['umtx']
                mtx, dist = data_calib['%s_calib' % type]['mtx'], data_calib['%s_calib' % type]['dist']

                if didx is not None:
                    idx_center = didx[i]
                    rvec_c, tvec_c = data_calib['DPc_calib']['rvecs'][idx_center], data_calib['DPc_calib']['tvecs'][idx_center]
                    umtx_c, mtx_c, dist_c = data_calib['DPc_calib']['umtx'], data_calib['DPc_calib']['mtx'], data_calib['DPc_calib']['dist']
                else:
                    rvec_c, tvec_c = rvec, tvec
                    umtx_c, mtx_c, dist_c = umtx, mtx, dist
                
                # projected points, uv coords
                prj_uv, _ = cv2.projectPoints(pts3d, rvec, tvec, mtx, dist)
                prj_uv = prj_uv.reshape([template.shape[0], template.shape[1], 2])
                
                # depth and normal
                prj, _ = cv2.projectPoints(pts3d, rvec_c, tvec_c, mtx_c, dist_c)
                depth, normal = get_plane_depth(worldpts, prj, rvec_c, tvec_c, mtx_c, dist_c, get_normal=True)
                depth = depth.reshape(template.shape[::-1])
            
                # projected image
                rotz = findrotz_euler(rvec)
                homo = findhomography(rvec, tvec, umtx, size_template_px, size_template_py, self.board.size_square_mm)
                invhomo = np.linalg.inv(homo)
                img = data_raw['%s_img' % type][idx]
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                uimg = cv2.undistort(img, mtx, dist, None, umtx)  # type: ignore
                pimg = cv2.warpPerspective(uimg, invhomo, template.shape[::-1])  # image to projective space
                scale_term = max(np.abs(invhomo[0, 0] + invhomo[0, 1]), np.abs(invhomo[1, 0] + invhomo[1, 1]))  # to determine patch_size in projective space
                # since invhomo[2, 0] and invhomo[2, 1] is very small (<< 1), we ignore that terms in computing patch_size.
                
                # focus image
                if infos_focus is not None:
                    rotz_f = findrotz_euler(rvec_c)
                    homo_f = findhomography(rvec_c, tvec_c, umtx_c, size_template_px, size_template_py, self.board.size_square_mm)
                    img_f = data_raw['Focus_img'][idx]
                    if img_f.ndim == 3:
                        img_f = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)
                    uimg_f = cv2.undistort(img_f, mtx_c, dist_c, None, umtx_c)  # type: ignore
                    pimg_f = cv2.warpPerspective(uimg_f, np.linalg.inv(homo_f), template.shape[::-1])
                else:
                    rotz_f, pimg_f = None, None
            
                # clean patches' patch
                mask_all = np.logical_and(infos['mask'][i], infos_focus['mask'][i]) if infos_focus is not None else infos['mask'][i]
                count_mask = np.sum(mask_all)
                cleans = subpixel_cropper_batch(template, circles_center, size_patch, mode='bilinear')
                cleans = np.stack(cleans, axis=0)[mask_all]
            
                if count_mask > 1:
                
                    # masking circles
                    circles = itemgetter(*np.where(mask_all)[0])(infos['circle'][i])
            
                    # centers from detected circles
                    blur_center = np.stack([np.array(circle.pt) for circle in circles if circle != []], axis=0)
                    blur_center *= self.board_scales[bidx]
                    
                    # rotated depth (rotate whole depth first, then crop for continuous mapping)
                    rotated_pdepth = ndimage.rotate(input=depth, angle=-rotz * 180 / math.pi, order=0)
                    blur_center_ = grid_rotate(blur_center, -rotz, template.shape[::-1], rotated_pdepth.shape[::-1])
                    rotated_depths = subpixel_cropper_batch(rotated_pdepth, blur_center_, size_patch)
                    
                    # rotated pimg
                    angle_torch = torch.tensor(-rotz * 180 / math.pi)
                    blurs = subpixel_cropper_batch(np.float32(pimg), blur_center, size_patch, mode='bilinear', as_numpy=False)
                    rotated_blurs = rotate(blurs[:, None], angle=angle_torch.repeat(len(blurs)), 
                                           mode='bilinear', padding_mode='border').squeeze().numpy()

                    # based on circles and radius, create patch coord
                    rotated_uu = ndimage.rotate(input=np.float32(prj_uv[:, :, 0]), angle=-rotz * 180 / math.pi, order=0)
                    rotated_vv = ndimage.rotate(input=np.float32(prj_uv[:, :, 1]), angle=-rotz * 180 / math.pi, order=0)
                    patch_uu = subpixel_cropper_batch(rotated_uu, blur_center_, size_patch)
                    patch_vv = subpixel_cropper_batch(rotated_vv, blur_center_, size_patch)
                    patch_uv = np.stack([patch_uu, patch_vv], -1)
                    
                    # save paired all-in-focus image
                    if infos_focus is not None:
                        circles_f = itemgetter(*np.where(mask_all)[0])(infos_focus['circle'][i])
                        focus_center = np.stack([np.array(circle.pt) for circle in circles_f if circle != []], axis=0)
                        focus_center *= self.board_scales[bidx]
                        
                        angle_torch_f = torch.tensor(-rotz_f * 180 / math.pi)
                        focuses = subpixel_cropper_batch(np.float32(pimg_f), focus_center, size_patch, mode='bilinear', as_numpy=False)
                        rotated_f = rotate(focuses[:, None], angle=angle_torch_f.repeat(len(focuses)), 
                                                 mode='bilinear', padding_mode='border').squeeze().numpy()
                        focuses_ += np.split(rotated_f, len(rotated_f))
                
                    # save data
                    uv_coord += np.split(patch_uv, len(patch_uv))
                    sharps_ += np.split(cleans, len(cleans))
                    depths_ += np.split(rotated_depths, len(rotated_depths))
                    blurs_ += np.split(rotated_blurs, len(rotated_blurs))
                    normals_ += [normal for _ in range(count_mask)]
                    Ks_ += [umtx for _ in range(count_mask)]
                    scales_ += [scale_term for _ in range(count_mask)]
                    indexes_ += [i for _ in range(count_mask)]
                    indexes_board_ += [idx for _ in range(count_mask)]
                
            # save data
            result['uv_coord'].append(np.array(uv_coord))
            result['depth'].append(np.array(depths_))
            result['normal'].append(np.array(normals_))
            result['sharp'].append(np.array(sharps_))
            result['blur'].append(np.array(blurs_))
            result['Kmat'].append(np.array(Ks_))
            result['scale'].append(np.array(scales_))
            result['idx'].append(np.array(indexes_))
            result['idx_board'].append(np.array(indexes_board_))
            result['size_patch'].append(size_patch)
            if infos_focus is not None:
                result['focus'].append(np.array(focuses_))
                
        return result
    
    def detect_circle_from_img(self, img, minsize, corners, size_patch, rotz):
        
        circles = []
        max_radius = 0
        numpatch = len(corners)
        nimg = (img - img.min()) / (img.max() - img.min()) * 255
        
        # Setup SimpleBlobDetector parameters.
        blobParams = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        blobParams.minThreshold = int(255 * (1.0 - self.blob_cfg.blobParams.maxthres))
        blobParams.maxThreshold = int(255 * (1.0 - self.blob_cfg.blobParams.minthres))

        # Filter by Area.
        blobParams.filterByArea = self.blob_cfg.blobParams.filterByArea
        blobParams.maxArea = self.blob_cfg.blobParams.maxArea  # maxArea may be adjusted to suit for your experiment

        # Filter by Circularity
        blobParams.filterByCircularity = self.blob_cfg.blobParams.filterByCircularity
        blobParams.minCircularity = self.blob_cfg.blobParams.minCircularity

        # Filter by Convexity
        blobParams.filterByConvexity = self.blob_cfg.blobParams.filterByConvexity
        blobParams.minConvexity = self.blob_cfg.blobParams.minConvexity

        # Filter by Inertia
        blobParams.filterByInertia = self.blob_cfg.blobParams.filterByInertia
        blobParams.minInertiaRatio = self.blob_cfg.blobParams.minInertiaRatio
        
        # crop to get patch
        corners_patch = np.mean(corners, axis=1)  # [N, 2]
        patch = subpixel_cropper_batch(np.float32(nimg), corners_patch, size_patch, mode='bilinear', as_numpy=False)
        patch = rotate(patch[:, None], angle=rotz.repeat(len(patch)), mode='bilinear', padding_mode='border').squeeze().numpy()
        startpoint = np.stack([corners_patch[:, 0] - size_patch / 2, corners_patch[:, 1] - size_patch / 2], axis=1)
        
        for i in range(numpatch):
            
            # masking by threshold
            patch_ = (patch[i] + np.fliplr(patch[i])) * 0.5
            mask_patch = patch_ > self.blob_cfg.blobParams.minthres
            
            # Create a detector with the parameters
            blobParams.minArea = int(np.round(minsize[i]))  # minArea may be adjusted to suit for your experiment
            blobDetector = cv2.SimpleBlobDetector_create(blobParams)

            # detect circles in gray image
            masked_patch = (255 - patch_).astype('uint8') * mask_patch
            keypoints = blobDetector.detect(masked_patch)
            scale = 1.0
            if len(keypoints) == 0:  # consider the case because of blur
                rsize_patch = int(size_patch * 0.25)
                rsize_patch = rsize_patch + 1 if rsize_patch % 2 == 0 else rsize_patch
                masked_patch = cv2.resize(masked_patch, dsize=(rsize_patch, rsize_patch), interpolation=cv2.INTER_LANCZOS4)
                scale = size_patch / rsize_patch
                keypoints = blobDetector.detect(masked_patch)
            
            # from skimage.draw import circle_perimeter
            # circy, circx = circle_perimeter(int(keypoints[0].pt[1]), int(keypoints[0].pt[0]), int(keypoints[0].size), shape=patch.shape)
            # patch[circy, circx] = (220)
            # cv2.imwrite('ex7.png', patch)
            
            if len(keypoints) > 1:  # too many circles are detected
                # masking (condition)
                masker_size = np.array([key.size * scale > minsize[i] for key in keypoints])
                masker_pt = np.array([np.linalg.norm(key.pt * np.array([scale, scale]) + startpoint[i] - corners_patch[i]) <= 20 for key in keypoints])
                masker = np.logical_and(masker_size, masker_pt[:, None])
                keypoints = self.refine_detection(keypoints, masker)
                
            if len(keypoints) == 0:
                circles.append([])
            else:
                keypoints[0].pt *= np.array([scale, scale])
                keypoints[0].pt += startpoint[i]
                keypoints[0].size *= scale
                circles.append(keypoints[0])
                max_radius = max(max_radius, keypoints[0].size)
                
        mask = [(circle != []) for circle in circles]

        return circles, mask, max_radius
    
    def refine_detection(self, keypoints, masker):
        
        nkeypoints = []
        for idx, point in enumerate(keypoints):
            if masker[idx]:
                nkeypoints.append(point)

        return nkeypoints
        
        
        

        
