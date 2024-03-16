
import cv2
import math
import torch
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from kornia.geometry.transform import crop_by_boxes
from scipy.optimize import lsq_linear


def convert2homography(coord, mul=1.0):
    adder = np.ones(coord.shape[:-1] + (1,)) * mul
    return np.concatenate([coord, adder], axis=-1)


def get_3d_points(worldpts, rvec, tvec):
    # world coordinate
    Rmat = cv2.Rodrigues(rvec)[0]
    P = np.concatenate([Rmat, tvec], axis=1)
    P = np.concatenate([P, np.array([[0, 0, 0, 1]])], axis=0)
    X_h = convert2homography(worldpts, 1.0)
    W_h = np.matmul(P, np.transpose(X_h))
    return np.transpose(W_h)[:, :3]


def get_plane_depth(worldpts, pts2d, rvec, tvec, mtx, dist, get_normal=False):
        
    # world coordinate
    Rmat = cv2.Rodrigues(rvec)[0]
    P = np.concatenate([Rmat, tvec], axis=1)
    P = np.concatenate([P, np.array([[0, 0, 0, 1]])], axis=0)
    X_h = convert2homography(worldpts, 1.0)
    W_h = np.matmul(P, np.transpose(X_h))
    Amat = np.transpose(W_h[:3, :])
    bdata = np.ones(W_h[0, :].shape)
    res = lsq_linear(Amat, bdata)
    
    # depth
    unprojected = convert2homography(np.squeeze(cv2.undistortPoints(pts2d, mtx, dist)), 1.0)
    depth = np.expand_dims(1.0 / np.matmul(res.x, np.transpose(unprojected)), axis=1)
    
    # test error
    # unprojected2 = unprojected * depth
    # test = np.transpose(np.matmul(np.linalg.inv(P), np.transpose(convert2homography(unprojected2, 1.0))))
    # pdb.set_trace()
    
    if get_normal:
        return depth, res.x
    else:
        return depth
    
    
def findrotz_euler(rvec):
    '''
        refer to https://learnopencv.com/rotation-matrix-to-euler-angles/
    '''
    Rmat = cv2.Rodrigues(rvec)[0]
    
    sy = math.sqrt(Rmat[0, 0] * Rmat[0, 0] + Rmat[1, 0] * Rmat[1, 0])
    
    if sy < 1e-6: # singularity at north/south pole
        return 0
    else:
        return math.atan2(Rmat[1, 0], Rmat[0, 0])
    

def findhomography(rvec, tvec, newmtx, px_size, py_size, mm_size):
    
    # find homography matrix from Rotation matrix and translation matrix
    '''
    https://medium.com/analytics-vidhya/using-homography-for-pose-estimation-in-opencv-a7215f260fdd
    '''
    Rmat = cv2.Rodrigues(rvec)[0]
    M = np.concatenate([Rmat[:, :2], tvec], axis=1)
    homo = np.matmul(newmtx, M)
    homo /= homo[2, 2]
    scaler = np.eye(3)
    scaler[0, 0] *= px_size / mm_size  # scaling (mm to px)
    scaler[1, 1] *= py_size / mm_size  # scaling (mm to px)
    homo = np.matmul(homo, np.linalg.inv(scaler))
    
    return homo


def subpixel_cropper(img, center, size_patch):
    patch = cv2.getRectSubPix(img, (size_patch, size_patch), center)
    startpoint = center - (size_patch / 2.0, size_patch / 2.0)
    return patch, startpoint


def subpixel_cropper_batch(img, center, size_patch, mode='nearest', as_numpy=True):
    '''
        img : [H, W, C], numpy array
        center : [B, 2], numpy array
        size_patch (P) : a single value
        mode : nearest or bilinear
    '''
    batch_num = len(center)
    center_x, center_y = center[:, 0], center[:, 1]
    img_ = repeat(torch.from_numpy(img), 'h w c -> b c h w', b=batch_num)
    
    # create src_box grid
    src_box_grid_x = np.stack([center_x - size_patch / 2, center_x + size_patch / 2, center_x + size_patch / 2, center_x - size_patch / 2], axis=-1)
    src_box_grid_y = np.stack([center_y - size_patch / 2, center_y - size_patch / 2, center_y + size_patch / 2, center_y + size_patch / 2], axis=-1)
    src_box_grid = torch.from_numpy(np.stack([src_box_grid_x, src_box_grid_y], axis=-1))  # [B, 4, 2]
    
    # create dst_box grid
    dst_box_grid_x = np.stack([0.0, size_patch - 1, size_patch - 1, 0.0], axis=-1)
    dst_box_grid_y = np.stack([0.0, 0.0, size_patch - 1, size_patch - 1], axis=-1)
    dst_box_grid = repeat(torch.from_numpy(np.stack([dst_box_grid_x, dst_box_grid_y], axis=-1)), 'n m -> b n m', b=batch_num)  # [B, 4, 2]
    
    # do crop
    cropped_img = crop_by_boxes(img_, src_box_grid, dst_box_grid, mode=mode)
    if as_numpy:
        return rearrange(cropped_img.numpy(), 'b c h w -> b h w c')  # [B, P, P, C]
    else:
        return cropped_img  # [B, C, P, P]


def grid_rotate(pts, angle, old_size, new_size):
    '''
        pts: [N, 2]
        angle : radians
        old_size, new_size : (h, w)
    '''
    pts_ = pts.copy()
    pts_[:, 0] -= old_size[1] / 2
    pts_[:, 1] -= old_size[0] / 2
    
    npts_x = np.cos(angle) * pts_[:, 0] + np.sin(angle) * pts_[:, 1]
    npts_y = -np.sin(angle) * pts_[:, 0] + np.cos(angle) * pts_[:, 1]
    npts = np.stack([npts_x, npts_y], -1)
    
    npts[:, 0] += new_size[1] / 2
    npts[:, 1] += new_size[0] / 2
    
    return npts